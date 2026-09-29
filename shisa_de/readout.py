"""The raw DE-1 readout: render a question, read the answer letters, normalize.

This module is the whole contract. Everything else in the package is a
convenience layer over it. `docs/READOUT.md` documents the wire format, the
rendered prompt, the letter slots, and the fallback path, with full samples.

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

Nothing here generates text, and nothing outside the option letters is part of
the answer.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .images import validate_image_url
from .questions import MAX_OPTIONS, Question, QuestionError

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

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

#: The readout version. Any change to rendering, request shape, or slot handling
#: must bump this: thresholds fitted against one version do not transfer.
READOUT_VERSION = "de1-letter-slots-v2"


class ReadoutError(RuntimeError):
    """The readout could not produce an answer."""


@dataclass(frozen=True)
class Slot:
    """One option's answer position: the letter, its token id, and its text."""

    letter: str
    token_id: int
    token_text: str


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

    # -- rendering ---------------------------------------------------------

    def _payload(self, state: Any, question: Question) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
        """The user JSON object and the option pairs for one question.

        Both readouts send this same object. The text readout renders it into the
        prompt itself; the image readout sends it as the text part beside the
        image and lets the server render.
        """
        question.validate()
        options = question.options()
        payload = {
            "evidence": state,
            "criterion": question.instructions,
            "options": [
                {"letter": LETTERS[index], "description": description}
                for index, (_, description) in enumerate(options)
            ],
        }
        return payload, options

    def render(self, state: Any, question: Question) -> str:
        """Render one question against one state into the prompt the model sees."""
        payload, _ = self._payload(state, question)
        tokenizer = self.ensure_tokenizer()
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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
            raise ReadoutError(
                f"{self.base_url}{path} returned HTTP {response.status_code}: {detail}"
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

    def _letter_logprob(self, prompt: str, slot: Slot) -> tuple[float | None, int | None, int]:
        """One fallback request: read a single letter's logprob at the answer boundary."""
        data = self._completions(
            {
                "model": self.model,
                "prompt": prompt + slot.letter,
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

    def read_image(
        self,
        state: Any,
        question: Question,
        image_url: str,
        *,
        top_logprobs: int = 20,
        debug: bool = False,
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
    "DIRECT_SYSTEM",
    "LETTERS",
    "READOUT_VERSION",
    "LetterRead",
    "Readout",
    "ReadoutError",
    "Slot",
    "softmax",
]
