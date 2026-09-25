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

from .questions import MAX_OPTIONS, Question, QuestionError

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: The system line the measured scaffold uses. Changing it changes answers, so
#: it is part of the readout identity rather than a preference.
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

#: The readout version. Any change to rendering, request shape, or slot handling
#: must bump this: thresholds fitted against one version do not transfer.
READOUT_VERSION = "de1-letter-slots-v1"


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

    def render(self, state: Any, question: Question) -> str:
        """Render one question against one state into the prompt the model sees."""
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

    def _completions(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post("/v1/completions", json=body)
        except httpx.HTTPError as exc:  # network, timeout, DNS
            raise ReadoutError(f"request to {self.base_url}/v1/completions failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:400]
            raise ReadoutError(
                f"{self.base_url}/v1/completions returned HTTP {response.status_code}: {detail}"
            )
        return response.json()

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
    "DIRECT_SYSTEM",
    "LETTERS",
    "READOUT_VERSION",
    "LetterRead",
    "Readout",
    "ReadoutError",
    "Slot",
    "softmax",
]
