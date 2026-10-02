"""Text choice overflow via balanced chunks and one finalist per chunk.

The per-prompt readout stays capped at 26 letters. Returned scores are the
final-round softmax conditional on finalist selection, not calibrated global
probabilities. Eliminated options retain keys with score zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .questions import Choice, MAX_OPTIONS, QuestionError
from .readout import LETTERS, LetterRead, Readout, ReadoutError

OVERFLOW_STRATEGY = "finalist-top1"
# One finalist per chunk must still fit into a single final read.
MAX_OVERFLOW_OPTIONS = MAX_OPTIONS * MAX_OPTIONS


def balanced_chunks(options: list[tuple[str, Any]]) -> list[list[tuple[str, Any]]]:
    count = len(options)
    blocks = (count + MAX_OPTIONS - 1) // MAX_OPTIONS
    if count <= MAX_OPTIONS or count > MAX_OVERFLOW_OPTIONS:
        raise QuestionError(f"overflow choice needs 27..{MAX_OVERFLOW_OPTIONS} options, got {count}")
    base, extra = divmod(count, blocks)
    result, start = [], 0
    for index in range(blocks):
        size = base + (index < extra)
        result.append(options[start:start + size])
        start += size
    return result


def validate_overflow(question: Choice) -> None:
    if not question.instructions or not str(question.instructions).strip():
        raise QuestionError("instructions must be a non-empty question")
    balanced_chunks(question.options())


@dataclass
class OverflowRead:
    probabilities: dict[str, float]
    finalists: list[str]
    components: list[tuple[list[str], LetterRead]]

    @property
    def requests(self) -> int:
        return sum(read.requests for _, read in self.components)

    @property
    def prompt_tokens(self) -> int:
        return sum(read.prompt_tokens for _, read in self.components)

    @property
    def logical_reads(self) -> int:
        return len(self.components)

    @property
    def missing_from_top(self) -> list[str]:
        return [f"read-{index}:{letter}" for index, (_, read) in enumerate(self.components)
                for letter in read.missing_from_top]

    def raw(self, debug: bool) -> dict[str, Any]:
        stages = []
        for index, (keys, read) in enumerate(self.components):
            item = {"stage": "final" if index == len(self.components) - 1 else "chunk",
                    "options": keys, "logprobs": read.logprobs,
                    "probabilities": read.probabilities, "requests": read.requests,
                    "prompt_tokens": read.prompt_tokens,
                    "missing_from_top": read.missing_from_top, "ranks": read.ranks}
            if debug:
                item["prompt"] = read.prompt
                item["top_logprobs"] = read.top_logprobs
            stages.append(item)
        return {"strategy": OVERFLOW_STRATEGY, "score_semantics": "conditional-on-finalists",
                "probabilities": self.probabilities, "finalists": self.finalists,
                "requests": self.requests, "prompt_tokens": self.prompt_tokens,
                "logical_reads": self.logical_reads, "components": stages}


def _check_read(read: LetterRead, count: int) -> None:
    letters = set(LETTERS[:count])
    if set(read.probabilities) != letters or set(read.logprobs) != letters:
        raise ReadoutError("overflow component returned an incomplete option distribution")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in read.probabilities.values()):
        raise ReadoutError("overflow component returned invalid probabilities")
    if not math.isclose(sum(read.probabilities.values()), 1.0, abs_tol=1e-9):
        raise ReadoutError("overflow component probabilities do not sum to one")


def read_overflow(readout: Readout, state: Any, question: Choice, *, debug: bool = False) -> OverflowRead:
    """Run the frozen finalist-top1 rule without using cross-chunk confidence."""
    validate_overflow(question)
    options = question.options()
    components = []
    finalists = []
    # Sequential within a head: DecisionModel's existing head pool bounds
    # concurrency. Do not create a nested pool for each wide question.
    for block in balanced_chunks(options):
        built = Choice(question.instructions, dict(block))
        read, _ = readout.evaluate(state, built, debug=debug)
        _check_read(read, len(block))
        keys = [key for key, _ in block]
        components.append((keys, read))
        best = max(range(len(block)), key=lambda i: read.probabilities[LETTERS[i]])
        finalists.append(block[best])
    # Blocks are consecutive, so their winners already have original order.
    final_read, _ = readout.evaluate(state, Choice(question.instructions, dict(finalists)), debug=debug)
    _check_read(final_read, len(finalists))
    keys = [key for key, _ in finalists]
    components.append((keys, final_read))
    scores = {key: 0.0 for key, _ in options}
    scores.update({key: final_read.probabilities[LETTERS[i]] for i, key in enumerate(keys)})
    return OverflowRead(scores, keys, components)
