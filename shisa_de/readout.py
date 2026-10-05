"""The raw readout: render a question, read the answer codes, normalize.

This module is the whole contract. Everything else in the package is a
convenience layer over it. `docs/READOUT.md` documents the DE-1 wire format, the
rendered prompt, the letter slots, and the fallback path, with full samples;
`docs/READOUT-DE2.md` documents what DE-2 changes.

The readout makes one vLLM request per question:

1. Render the state and the question with the checkpoint's own chat template.
2. ``POST /v1/completions`` with ``max_tokens: 1``, ``temperature: 0`` and
   ``logprobs: 20``, and read the first generated token's distribution.
3. Take the logprob of each option letter. A letter outside the returned top-k
   costs one more request, ``prompt + letter`` with ``prompt_logprobs: 0``.
4. Softmax over the letters. The argmax is the answer.

The image readout is the same read of the same slots over a different request.
An image cannot travel through the text path: only the server can expand an
image placeholder into image tokens, so `Readout.read_image` posts messages to
``/v1/chat/completions`` and lets the server render the prompt. Two things
change as a result. The distribution arrives as a list of token objects under
``choices[0].logprobs.content[0].top_logprobs`` rather than as a map, and the
boundary check cannot compare against a client-side rendering, so it requires
the last prompt token to be the generation-prompt terminator instead. There is
no letter fallback on that path: a letter outside the returned top-k is an
error rather than an extra request, because appending a letter to a chat
request does not reproduce the answer boundary.

DE-2 keeps the scaffold and changes the read (`docs/READOUT-DE2.md`): options
beyond `Z` take codes from a pinned codebook, every candidate's logprob is
requested by token id in one request (`Readout.read_codes`), the user turn may
be rendered twice, and `Readout.think` generates a bounded thought that the
codes are then read after. `shisa_de/policy.py` sequences those reads.

Apart from that bounded thought, nothing here generates text, and nothing
outside the option codes is part of the answer.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from threading import Lock
from typing import Any

import httpx

from .images import validate_image_url
from .questions import MAX_OPTIONS, Question, QuestionError

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Serialize cold imports and loads across Readout instances, not just workers
# sharing one client. Initialized tokenizers do not acquire this lock.
_TOKENIZER_INIT_LOCK = Lock()

#: The system line the measured scaffold uses. Changing it changes answers, so
#: it is part of the readout identity rather than a preference.
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

#: The generation-prompt terminator the checkpoint's chat template ends on, so
#: the answer letter is the token after it. The image readout cannot render the
#: prompt itself, so it confirms the answer boundary by requiring the server's
#: last prompt token to be this one.
ANSWER_PREFIX = "<channel|>"

#: Identity of the answer-producing contract, not the client build. Bump only
#: for changes to rendering, request shape, slot handling, or scoring semantics;
#: existing thresholds must then be revalidated. Implementation-only fixes
#: (such as initialization locking) must retain this version.
READOUT_VERSION = "de1-letter-slots-v3"

#: The DE-2 contract: the same scaffold, read through the 256-code codebook with
#: the user turn written twice and an optional bounded thought. It is a separate
#: identity rather than a `v4` because DE-1 answers are still produced by the
#: version above; the two contracts coexist.
DE2_READOUT_VERSION = "de2-codebook-v1"

#: The one place readout identities live: one current version per family.
READOUT_VERSIONS = {"de1": READOUT_VERSION, "de2": DE2_READOUT_VERSION}

#: For each current version, the earlier versions whose direct text read is
#: answer-identical to it, so a calibration fitted against one of them still
#: describes the answers this client returns. A version bump that changes the
#: direct text read must not list its predecessors here; one that only adds a
#: path (v2 added images, v3 added overflow orchestration) should.
CALIBRATION_COMPATIBLE: dict[str, tuple[str, ...]] = {
    READOUT_VERSION: ("de1-letter-slots-v1", "de1-letter-slots-v2"),
    DE2_READOUT_VERSION: (),
}


def calibration_compatible(fitted: str, current: str) -> bool:
    """Whether a calibration fitted against `fitted` applies under `current`."""
    return fitted == current or fitted in CALIBRATION_COMPATIBLE.get(current, ())


#: The system line for the DE-2 thinking read: the direct line with the ban on
#: reasoning replaced by a request for it. Part of the DE-2 readout identity.
THINK_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Reason through the question step by step before you answer. When your reasoning is complete, "
    "respond with only its uppercase letter."
)

#: The line between the two copies of the user turn in the DE-2 repeated read.
INPUT_REPEAT = "\n\nRead the same input again before answering:\n"

CODEBOOK_FILE = "data/codebook-de2.json"

#: The most options one DE-2 prompt can carry: the length of the codebook.
MAX_CODES = 256

#: vLLM accepts at most this many ids in `logprob_token_ids`, so a wider
#: question reads its candidates in consecutive requests over the same prompt.
CANDIDATE_ID_LIMIT = 128


@lru_cache(maxsize=1)
def codebook() -> tuple[str, ...]:
    """The pinned DE-2 answer codes, in order: `A` to `Z`, then uppercase pairs."""
    text = resources.files("shisa_de").joinpath(CODEBOOK_FILE).read_text(encoding="utf-8")
    codes = tuple(json.loads(text)["codes"])
    if codes[:len(LETTERS)] != tuple(LETTERS) or len(set(codes)) != len(codes) or len(codes) != MAX_CODES:
        raise ReadoutError(f"{CODEBOOK_FILE} is not the pinned {MAX_CODES}-code codebook")
    return codes


def codes_for(count: int) -> list[str]:
    """The answer codes for a question with `count` options, in option order.

    Up to 26 options these are the letters, so a small question renders the same
    under either contract.
    """
    if count <= len(LETTERS):
        return list(LETTERS[:count])
    if count > MAX_CODES:
        raise QuestionError(f"{count} answers exceeds the {MAX_CODES} codes the readout has")
    return list(codebook()[:count])


class ReadoutError(RuntimeError):
    """The readout could not produce an answer."""


class ReadoutHTTPError(ReadoutError):
    """The endpoint answered with an error status."""

    def __init__(self, message: str, status: int, detail: str) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Slot:
    """One option's answer position: the letter, its token id, and its text."""

    letter: str
    token_id: int
    token_text: str


@dataclass
class Thought:
    """One bounded thought: the prompt it followed and the tokens generated."""

    prompt_ids: list[int]
    token_ids: list[int]
    closed: bool
    prompt_tokens: int = 0
    text: str | None = None


@dataclass
class LetterRead:
    """What one question's readout cost and returned."""

    logprobs: dict[str, float]
    probabilities: dict[str, float]
    requests: int = 0
    prompt_tokens: int = 0
    missing_from_top: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)
    sampled: str | None = None
    prompt: str | None = None
    top_logprobs: dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    #: The messages sent for an image read, when ``debug`` asked for them. The
    #: server renders the prompt on that path, so there is no local `prompt`.
    messages: list[dict[str, Any]] | None = None
    #: Image reads only: whether the server rendered the system turn the way the
    #: text scaffold does. `string` when it did, `differs` when it did not,
    #: `unverified` when the response did not carry enough to tell.
    system_render: str | None = None

    def answer(self) -> str:
        """The letter with the highest probability."""
        return max(self.probabilities, key=self.probabilities.get)


class Readout:
    """The letter-slot readout over an OpenAI-compatible completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer: str | None = None,
        *,
        tokenizer_revision: str | None = None,
        local_files_only: bool = False,
        timeout: float = 120.0,
        system_prompt: str = DIRECT_SYSTEM,
        max_logprobs: int = 20,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer_source = tokenizer or model
        self.tokenizer_revision = tokenizer_revision
        self.local_files_only = local_files_only
        self.system_prompt = system_prompt
        self.max_logprobs = max_logprobs
        self._tokenizer = None
        self._slot_cache: dict[str, Slot] = {}
        self._boundary_checked: set[str] = set()
        # Cleared the first time a server rejects `logprob_token_ids`, after which
        # candidates are read from the top-k with the per-code fallback.
        self._candidate_ids = True
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers, transport=transport)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Readout":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- tokenizer and slots ----------------------------------------------

    def ensure_tokenizer(self):
        """Load the tokenizer used for rendering, once.

        Only the tokenizer is needed, so `transformers` runs without torch.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        with _TOKENIZER_INIT_LOCK:
            if self._tokenizer is not None:
                return self._tokenizer
            from transformers import AutoTokenizer  # imported lazily: not needed for a stubbed readout

            kwargs: dict[str, Any] = {}
            if self.tokenizer_revision:
                kwargs["revision"] = self.tokenizer_revision
            if self.local_files_only:
                kwargs["local_files_only"] = True
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_source, **kwargs)
            return self._tokenizer

    def slot(self, letter: str) -> Slot:
        """Resolve one option letter to its token.

        Letters must be single tokens for the readout to work, and the ids are
        checkpoint-specific, so they are resolved from the served tokenizer
        rather than hardcoded.
        """
        if letter in self._slot_cache:
            return self._slot_cache[letter]
        tokenizer = self.ensure_tokenizer()
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ReadoutError(
                f"answer letter {letter!r} is not one token for tokenizer {self.tokenizer_source!r}; "
                "this checkpoint cannot be read with letter slots"
            )
        resolved = Slot(letter=letter, token_id=encoded[0], token_text=letter)
        self._slot_cache[letter] = resolved
        return resolved

    def slots(self, count: int) -> list[Slot]:
        if count > MAX_OPTIONS:
            raise QuestionError(f"{count} answers exceeds the {MAX_OPTIONS} the readout supports")
        return [self.slot(LETTERS[index]) for index in range(count)]

    def code_slots(self, count: int) -> list[Slot]:
        """The DE-2 answer slots: the first `count` codes of the codebook."""
        return [self.slot(code) for code in codes_for(count)]

    # -- rendering ---------------------------------------------------------

    def _payload(self, state: Any, question: Question,
                 max_options: int = MAX_OPTIONS) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
        """The user JSON object and the option pairs for one question.

        Both readouts send this same object. The text readout renders it into the
        prompt itself; the image readout sends it as the text part beside the
        image and lets the server render.
        """
        question.validate(max_options)
        options = question.options()
        payload = {
            "evidence": state,
            "criterion": question.instructions,
            "options": [
                {"letter": code, "description": description}
                for code, (_, description) in zip(codes_for(len(options)), options)
            ],
        }
        return payload, options

    def render(self, state: Any, question: Question, *, repeat: int = 1, thinking: bool = False,
               max_options: int = MAX_OPTIONS) -> str:
        """Render one question against one state into the prompt the model sees.

        The defaults are the DE-1 scaffold. DE-2 passes `repeat=2` to write the
        user turn twice, `thinking=True` for the prompt a thought is generated
        from, and `max_options=MAX_CODES` to admit codebook-width questions.
        """
        payload, _ = self._payload(state, question, max_options)
        tokenizer = self.ensure_tokenizer()
        user = json.dumps(payload, ensure_ascii=False)
        if repeat > 1:
            user = INPUT_REPEAT.join([user] * repeat)
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": THINK_SYSTEM if thinking else self.system_prompt},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

    def check_boundary(self, prompt: str, slots: list[Slot]) -> None:
        """Verify that appending a letter adds exactly that letter's token.

        The answer is read at the first generated position, so the prompt plus
        the letter must tokenize as the prompt tokens plus the slot token. A
        mismatch means the client and the server disagree about the tokenizer,
        which would silently score the wrong distribution.
        """
        tokenizer = self.ensure_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for slot in slots:
            combined = tokenizer.encode(prompt + slot.letter, add_special_tokens=False)
            if combined != prompt_ids + [slot.token_id]:
                raise ReadoutError(
                    f"the answer boundary moves for {slot.letter!r}: prompt + letter does not "
                    f"tokenize as the prompt plus token {slot.token_id}. The tokenizer "
                    f"({self.tokenizer_source!r}) does not match the served model ({self.model!r})."
                )

    # -- requests ----------------------------------------------------------

    def list_models(self) -> list[str]:
        """The model ids the endpoint serves."""
        response = self._client.get("/v1/models")
        response.raise_for_status()
        return [entry.get("id") for entry in (response.json().get("data") or [])]

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """One POST that returns a JSON object, with the readout's error conventions."""
        try:
            response = self._client.post(path, json=body)
        except httpx.HTTPError as exc:  # network, timeout, DNS
            raise ReadoutError(f"request to {self.base_url}{path} failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:400]
            raise ReadoutHTTPError(
                f"{self.base_url}{path} returned HTTP {response.status_code}: {detail}",
                response.status_code, detail,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ReadoutError(f"{self.base_url}{path} did not return JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ReadoutError(
                f"{self.base_url}{path} returned a {type(data).__name__}, expected a JSON object"
            )
        return data

    def _completions(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._post("/v1/completions", body)

    def _letter_logprob(self, prompt: str | list[int], slot: Slot) -> tuple[float | None, int | None, int]:
        """One fallback request: read a single letter's logprob at the answer boundary."""
        data = self._completions(
            {
                "model": self.model,
                "prompt": prompt + slot.letter if isinstance(prompt, str) else prompt + [slot.token_id],
                "max_tokens": 1,
                "temperature": 0,
                "prompt_logprobs": 0,
            }
        )
        entries = (data["choices"][0].get("prompt_logprobs") or [None])[-1]
        prompt_tokens = int((data.get("usage") or {}).get("prompt_tokens") or 0)
        if not entries:
            return None, None, prompt_tokens
        entry = next(iter(entries.values()))
        return entry.get("logprob"), entry.get("rank"), prompt_tokens

    def read(self, prompt: str, option_count: int, *, debug: bool = False) -> LetterRead:
        """Read the option letters' distribution for a rendered prompt."""
        slots = self.slots(option_count)
        self.check_boundary(prompt, slots)
        data = self._completions(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": self.max_logprobs,
            }
        )
        choice = data["choices"][0]
        top = ((choice.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        usage = data.get("usage") or {}
        result = LetterRead(
            logprobs={},
            probabilities={},
            requests=1,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            sampled=choice.get("text"),
            prompt=prompt if debug else None,
            top_logprobs=dict(top) if debug else {},
        )
        for slot in slots:
            if slot.token_text in top:
                result.logprobs[slot.letter] = top[slot.token_text]
        missing = [slot for slot in slots if slot.letter not in result.logprobs]
        result.missing_from_top = [slot.letter for slot in missing]
        for slot in missing:
            logprob, rank, prompt_tokens = self._letter_logprob(prompt, slot)
            result.requests += 1
            result.prompt_tokens += prompt_tokens
            if logprob is None:
                raise ReadoutError(
                    f"letter {slot.letter!r} is missing from the top {self.max_logprobs} and the "
                    "prompt_logprobs fallback returned nothing for it"
                )
            result.logprobs[slot.letter] = logprob
            if rank is not None:
                result.ranks[slot.letter] = rank
        result.probabilities = softmax(result.logprobs)
        return result

    def evaluate(self, state: Any, question: Question, *, debug: bool = False) -> tuple[LetterRead, list[tuple[str, Any]]]:
        """Render and read one question. Returns the read and the option pairs."""
        started = time.perf_counter()
        prompt = self.render(state, question)
        options = question.options()
        read = self.read(prompt, len(options), debug=debug)
        read.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return read, options

    # -- the DE-2 reads ----------------------------------------------------

    def check_code_boundary(self, prompt_ids: list[int], slots: list[Slot]) -> None:
        """The DE-2 form of the boundary check, for a prompt that is already tokens.

        The prompt has to end on the generation-prompt terminator, and each code
        appended to that terminator has to add exactly its own token. The
        terminator is a control token, so what follows it tokenizes the same
        whatever precedes it, which lets the per-code half be checked once per
        code instead of once per code per question: a 256-option question would
        otherwise re-tokenize its whole prompt 256 times.
        """
        tokenizer = self.ensure_tokenizer()
        prefix_id = self.answer_prefix_id()
        if not prompt_ids or prompt_ids[-1] != prefix_id:
            raise ReadoutError(
                f"the prompt does not end on {ANSWER_PREFIX!r} (token {prefix_id}), so the answer "
                f"is not at the first generated position. The tokenizer "
                f"({self.tokenizer_source!r}) does not match the served model ({self.model!r})."
            )
        for slot in slots:
            if slot.letter in self._boundary_checked:
                continue
            if tokenizer.encode(ANSWER_PREFIX + slot.letter, add_special_tokens=False) != [prefix_id, slot.token_id]:
                raise ReadoutError(
                    f"the answer boundary moves for {slot.letter!r}: appended to {ANSWER_PREFIX!r} it "
                    f"does not tokenize as token {slot.token_id}. The tokenizer "
                    f"({self.tokenizer_source!r}) does not match the served model ({self.model!r})."
                )
            self._boundary_checked.add(slot.letter)

    def _check_prompt_tokens(self, data: dict[str, Any], expected: int) -> int:
        """Require the server to have counted the prompt the way the client tokenized it."""
        reported = int((data.get("usage") or {}).get("prompt_tokens") or 0)
        if reported and reported != expected:
            raise ReadoutError(
                f"the server counted {reported} prompt tokens where the client tokenized "
                f"{expected}: the tokenizer ({self.tokenizer_source!r}) does not match the "
                f"served model ({self.model!r})"
            )
        return reported

    def read_codes(self, prompt: str | list[int], slots: list[Slot], *, debug: bool = False) -> LetterRead:
        """Read the distribution over `slots` at the end of a prompt, by token id.

        One request names every candidate in `logprob_token_ids`, so each code's
        logprob comes back whatever its rank and no code needs the fallback.
        vLLM caps that list at `CANDIDATE_ID_LIMIT`, so a wider question sends the
        same prompt once per block of candidates. The logprobs are the server's
        raw ones over the whole vocabulary either way, so blocks combine under
        one softmax. A server without `logprob_token_ids` is read from its top-k,
        with the same one-request-per-missing-code fallback the DE-1 read uses.

        `prompt` is the rendered text, or token ids for the read that follows a
        thought, where the thought has to be continued token for token.
        """
        tokenizer = self.ensure_tokenizer()
        is_text = isinstance(prompt, str)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if is_text else list(prompt)
        self.check_code_boundary(prompt_ids, slots)
        result = LetterRead(logprobs={}, probabilities={}, prompt=prompt if debug and is_text else None)
        start = 0
        while start < len(slots):
            block = slots[start:start + CANDIDATE_ID_LIMIT] if self._candidate_ids else slots
            body: dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": self.max_logprobs,
                "return_tokens_as_token_ids": True,
            }
            if self._candidate_ids:
                body["logprob_token_ids"] = [slot.token_id for slot in block]
            try:
                data = self._completions(body)
            except ReadoutHTTPError as exc:
                if self._candidate_ids and exc.status in (400, 422) and "logprob_token_ids" in exc.detail:
                    self._candidate_ids = False
                    continue
                raise
            choice = data["choices"][0]
            top = ((choice.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
            result.requests += 1
            result.prompt_tokens += self._check_prompt_tokens(data, len(prompt_ids))
            if start == 0:
                result.sampled = choice.get("text")
            if debug:
                result.top_logprobs.update(top)
            for slot in block:
                value = top.get(f"token_id:{slot.token_id}", top.get(slot.token_text))
                if value is not None:
                    result.logprobs[slot.letter] = value
            start += len(block)
        missing = [slot for slot in slots if slot.letter not in result.logprobs]
        result.missing_from_top = [slot.letter for slot in missing]
        for slot in missing:
            logprob, rank, prompt_tokens = self._letter_logprob(prompt, slot)
            result.requests += 1
            result.prompt_tokens += prompt_tokens
            if logprob is None:
                raise ReadoutError(
                    f"code {slot.letter!r} was not returned for its token id and the "
                    "prompt_logprobs fallback returned nothing for it"
                )
            result.logprobs[slot.letter] = logprob
            if rank is not None:
                result.ranks[slot.letter] = rank
        # Option order, not arrival order: the fallback fills gaps out of sequence.
        result.logprobs = {slot.letter: result.logprobs[slot.letter] for slot in slots}
        result.probabilities = softmax(result.logprobs)
        return result

    def think(self, prompt: str, budget: int, *, debug: bool = False) -> Thought:
        """Generate one greedy thought of at most `budget` tokens.

        `prompt` is a `render(..., thinking=True)` prompt. Generation stops at the
        token that closes the thought, which is the same token every answer is
        read after, or at the budget. The tokens come back as ids so the read
        that follows continues exactly what the model produced; a thought cut by
        the budget is returned unclosed and the caller closes it.
        """
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ReadoutError(f"the thinking budget must be a positive integer, got {budget!r}")
        tokenizer = self.ensure_tokenizer()
        close_id = self.answer_prefix_id()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        data = self._completions(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": budget,
                "temperature": 0,
                "stop_token_ids": [close_id],
                "return_token_ids": True,
            }
        )
        choice = data["choices"][0]
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list) or any(isinstance(t, bool) or not isinstance(t, int) for t in token_ids):
            raise ReadoutError(
                "the completion carries no token_ids, so the thought cannot be continued token "
                "for token; the thinking read needs a server that honours return_token_ids"
            )
        closed = close_id in token_ids
        if closed:
            token_ids = token_ids[:token_ids.index(close_id)]
        return Thought(
            prompt_ids=prompt_ids,
            token_ids=list(token_ids),
            closed=closed,
            prompt_tokens=self._check_prompt_tokens(data, len(prompt_ids)),
            text=choice.get("text") if debug else None,
        )

    # -- the image readout -------------------------------------------------

    def answer_prefix_id(self) -> int:
        """The token id of the generation-prompt terminator, resolved locally.

        The image readout cannot render the prompt itself, so it confirms the
        answer position from the server's side instead: the last prompt token
        must be this token. It has to be exactly one token for that check to
        mean anything, which is the same single-token requirement the option
        letters carry.
        """
        tokenizer = self.ensure_tokenizer()
        encoded = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != ANSWER_PREFIX:
            raise ReadoutError(
                f"the generation-prompt terminator {ANSWER_PREFIX!r} is not one token for "
                f"tokenizer {self.tokenizer_source!r}, so the image readout cannot confirm "
                "the answer boundary with this tokenizer"
            )
        return encoded[0]

    def _messages(self, payload: dict[str, Any], image_url: str) -> list[dict[str, Any]]:
        """The multimodal messages the server renders: the image, then the JSON."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                ],
            },
        ]

    @staticmethod
    def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
        """The first choice of a chat response, or an error naming what arrived."""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ReadoutError("the chat response carries no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ReadoutError(
                f"the chat response's first choice is a {type(choice).__name__}, expected an object"
            )
        return choice

    @staticmethod
    def _content_top_logprobs(choice: dict[str, Any]) -> dict[str, float]:
        """The answer position's token distribution from a chat response.

        A chat response carries top logprobs as a list of token objects, one per
        generated token, where the text path carries a map at the same position.
        """
        logprobs = choice.get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        if not isinstance(content, list) or not content:
            raise ReadoutError(
                "the chat response carries no logprobs.content, so there is no answer "
                "distribution to read"
            )
        first = content[0]
        entries = first.get("top_logprobs") if isinstance(first, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ReadoutError(
                "the chat response's logprobs.content[0] carries no top_logprobs, so there is "
                "no answer distribution to read"
            )
        top: dict[str, float] = {}
        for entry in entries:
            token = entry.get("token") if isinstance(entry, dict) else None
            logprob = entry.get("logprob") if isinstance(entry, dict) else None
            if (not isinstance(token, str) or not isinstance(logprob, (int, float))
                    or isinstance(logprob, bool) or not math.isfinite(logprob)):
                raise ReadoutError(
                    f"the chat response has a malformed top_logprobs entry: {entry!r}"
                )
            # First occurrence wins: a repeated token cannot carry two logprobs.
            top.setdefault(token, float(logprob))
        return top

    def _check_image_boundary(self, data: dict[str, Any], prefix_id: int) -> None:
        """Confirm the server read the distribution at the answer boundary.

        `check_boundary` does this for the text path by tokenizing the prompt it
        rendered itself. The image readout has no local prompt to tokenize, so the
        substitute is the server's last prompt token: it must be the exact token
        the local tokenizer resolves the generation-prompt terminator to.
        """
        prompt_logprobs = data.get("prompt_logprobs")
        if not isinstance(prompt_logprobs, list) or not prompt_logprobs:
            raise ReadoutError(
                "the chat response carries no root-level prompt_logprobs, so the answer "
                "boundary cannot be confirmed; the server must return prompt_logprobs"
            )
        last = prompt_logprobs[-1]
        if not isinstance(last, dict) or not last:
            raise ReadoutError(
                "the chat response's prompt_logprobs carries no token at the answer boundary"
            )
        token_id, entry = next(iter(last.items()))
        decoded = entry.get("decoded_token") if isinstance(entry, dict) else None
        if str(token_id) != str(prefix_id) or decoded != ANSWER_PREFIX:
            raise ReadoutError(
                f"the last prompt token is {decoded!r} (id {token_id}), not {ANSWER_PREFIX!r} "
                f"(id {prefix_id}); the distribution was not read at the answer boundary"
            )

    def _system_render(self, data: dict[str, Any], state: Any, question: Question) -> str:
        """Whether the server rendered the system turn as the text scaffold does.

        A chat server that hands the template content parts instead of strings
        can render the system turn differently: Gemma 4's template writes a space
        after a system message that arrives as parts. The answer boundary still
        checks out, so this compares the server's prompt tokens with the local
        render through the token that follows the system line.

        Returns `string` when they agree, `differs` when they do not, and
        `unverified` when the response does not carry the prompt tokens.
        """
        tokenizer = self.ensure_tokenizer()
        local = self.render(state, question)
        at = local.find(self.system_prompt)
        entries = data.get("prompt_logprobs")
        if at < 0 or not isinstance(entries, list):
            return "unverified"
        local_ids = tokenizer.encode(local, add_special_tokens=False)
        through_system = tokenizer.encode(local[:at + len(self.system_prompt)], add_special_tokens=False)
        want = local_ids[:len(through_system) + 1]
        if want[:-1] != through_system or len(entries) < len(want) or len(want) < 2:
            return "unverified"
        served = []
        for entry in entries[1:len(want)]:
            if not isinstance(entry, dict) or not entry:
                return "unverified"
            served.append(str(next(iter(entry))))
        # The first prompt position carries no logprob, so it is not compared.
        return "string" if served == [str(token) for token in want[1:]] else "differs"

    def probe_chat_render(self, state: Any, question: Question) -> str:
        """Ask the chat endpoint, without an image, how it renders the system turn.

        Returns `string`, `differs`, `unverified`, or `unavailable: ...`. The
        image readout goes through the chat endpoint, so this is what an image
        question will be rendered with.
        """
        payload, _ = self._payload(state, question)
        try:
            data = self._post(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    "max_tokens": 1,
                    "temperature": 0,
                    "prompt_logprobs": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except ReadoutError as exc:
            return f"unavailable: {exc}"
        return self._system_render(data, state, question)

    def read_image(
        self,
        state: Any,
        question: Question,
        image_url: str,
        *,
        top_logprobs: int = 20,
        debug: bool = False,
        require_string_system: bool = False,
    ) -> LetterRead:
        """Read one question's option letters with one image attached.

        The image is sent as an ``image_url`` content part and the question as the
        JSON object `render` would have rendered, because only the server can turn
        the image into image tokens. `image_url` must already be a URL: a local
        file name is rejected rather than forwarded, so call `prepare_image` first.

        Unlike `read`, there is no letter fallback. Appending a letter to a chat
        request does not reproduce the answer boundary, so a letter outside the
        returned top-k raises `ReadoutError` instead of costing a second request,
        and an incomplete distribution is never renormalized into an answer.

        The read records, as `system_render`, whether the server rendered the system
        turn the way the text scaffold does. `require_string_system` turns a
        different rendering into an error. `DecisionModel` does not set it: a
        stock vLLM renders the space on every image request and rejects images
        outright when told to pass strings (`docs/READOUT.md`, section 12), so the
        strict form is for a server whose template has been fixed.
        """
        started = time.perf_counter()
        image_url = validate_image_url(image_url)
        if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, int) or top_logprobs < 1:
            raise ReadoutError(f"top_logprobs must be a positive integer, got {top_logprobs!r}")
        payload, options = self._payload(state, question)
        slots = self.slots(len(options))
        if len(slots) > top_logprobs:
            raise ReadoutError(
                f"{len(slots)} options cannot be read from a top-{top_logprobs} distribution: "
                f"the server returns at most {top_logprobs} tokens, so at least "
                f"{len(slots) - top_logprobs} option letters would be missing and the image "
                f"readout has no letter fallback; pass top_logprobs={len(slots)} or higher and "
                f"serve the model with --max-logprobs {len(slots)} or higher"
            )
        prefix_id = self.answer_prefix_id()
        messages = self._messages(payload, image_url)
        data = self._post(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": top_logprobs,
                "prompt_logprobs": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        choice = self._first_choice(data)
        self._check_image_boundary(data, prefix_id)
        system_render = self._system_render(data, state, question)
        if require_string_system and system_render != "string":
            raise ReadoutError(
                "the server did not render the system turn the way the text scaffold does "
                f"(system render: {system_render}). Gemma 4's chat template writes a space after "
                "a system message that arrives as content parts, which is how vLLM passes "
                "every message of an image request"
            )
        top = self._content_top_logprobs(choice)
        missing = [slot.letter for slot in slots if slot.token_text not in top]
        if missing:
            raise ReadoutError(
                f"option letters {''.join(missing)} are missing from the returned top "
                f"{top_logprobs} logprobs and the image readout has no letter fallback; raise "
                f"the server's --max-logprobs above {top_logprobs} or pass a higher "
                "top_logprobs"
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        message = choice.get("message")
        sampled = message.get("content") if isinstance(message, dict) else None
        result = LetterRead(
            logprobs={slot.letter: top[slot.token_text] for slot in slots},
            probabilities={},
            requests=1,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            sampled=sampled if isinstance(sampled, str) else None,
            top_logprobs=dict(top) if debug else {},
            messages=messages if debug else None,
            system_render=system_render,
        )
        result.probabilities = softmax(result.logprobs)
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return result


def softmax(logprobs: dict[str, float]) -> dict[str, float]:
    """Turn a letter's logprobs into a distribution over the letters.

    Only the option letters take part: tokens outside the option list are
    discarded even when they outrank a valid option.
    """
    if not logprobs:
        raise ReadoutError("no letter logprobs to normalize")
    peak = max(logprobs.values())
    weights = {letter: math.exp(value - peak) for letter, value in logprobs.items()}
    total = sum(weights.values())
    return {letter: weight / total for letter, weight in weights.items()}


__all__ = [
    "ANSWER_PREFIX",
    "CALIBRATION_COMPATIBLE",
    "CANDIDATE_ID_LIMIT",
    "DE2_READOUT_VERSION",
    "DIRECT_SYSTEM",
    "INPUT_REPEAT",
    "LETTERS",
    "MAX_CODES",
    "READOUT_VERSION",
    "READOUT_VERSIONS",
    "THINK_SYSTEM",
    "LetterRead",
    "Readout",
    "ReadoutError",
    "ReadoutHTTPError",
    "Slot",
    "Thought",
    "calibration_compatible",
    "codebook",
    "codes_for",
    "softmax",
]
