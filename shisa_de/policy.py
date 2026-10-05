"""The DE-2 inference policy: read the question twice, think when unsure.

DE-2 answers from a sequence of reads over the same weights rather than from
one. `docs/READOUT-DE2.md` is the contract; this module sequences the reads
`shisa_de/readout.py` provides.

1. Render the scaffold with the user turn written twice and read the codes.
2. If the top probability of that read is below `THINK_GATE` and the question
   has at most `THINK_OPTION_CAP` options, render the question once more with
   thinking on, generate a thought of at most `THINK_BUDGET` tokens, close it,
   and read the codes after it. Otherwise the first read is the answer.

`repeat` stops after step 1 and `direct` reads the scaffold once. The gate reads
the raw distribution, before any calibration, as the measured policy did.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .questions import MAX_OPTIONS, Question
from .readout import MAX_CODES, LetterRead, Readout, Thought

POLICIES = ("direct", "repeat", "repeat-think")

#: The policies each family's contract defines. DE-1 was measured on one read.
FAMILY_POLICIES = {"de1": ("direct",), "de2": POLICIES}
DEFAULT_POLICY = {"de1": "direct", "de2": "repeat-think"}

#: The adopted gate: think when the repeated read's top probability is below this.
THINK_GATE = 0.7
#: Questions with more options than this never think.
THINK_OPTION_CAP = 26
#: The most tokens one thought may run to.
THINK_BUDGET = 1024

#: `Answer.strategy` for each way a policy can end.
DIRECT, REPEAT, REPEAT_THINK = "direct", "repeat2", "repeat2-think"


@dataclass
class PolicyRead:
    """Every read one question cost under a policy; the last one is the answer."""

    strategy: str
    components: list[tuple[str, LetterRead]]
    thought: Thought | None = None
    elapsed_ms: float = 0.0

    @property
    def read(self) -> LetterRead:
        return self.components[-1][1]

    @property
    def requests(self) -> int:
        return sum(read.requests for _, read in self.components) + (self.thought is not None)

    @property
    def prompt_tokens(self) -> int:
        thought = self.thought.prompt_tokens if self.thought else 0
        return sum(read.prompt_tokens for _, read in self.components) + thought

    @property
    def logical_reads(self) -> int:
        return len(self.components)

    @property
    def thought_tokens(self) -> int:
        return len(self.thought.token_ids) if self.thought else 0

    @property
    def missing_from_top(self) -> list[str]:
        return list(self.read.missing_from_top)

    def raw(self, debug: bool) -> dict[str, Any]:
        stages = []
        for name, read in self.components:
            item = {"stage": name, "logprobs": read.logprobs, "probabilities": read.probabilities,
                    "requests": read.requests, "prompt_tokens": read.prompt_tokens,
                    "missing_from_top": read.missing_from_top, "ranks": read.ranks}
            if debug:
                item["prompt"] = read.prompt
                item["top_logprobs"] = read.top_logprobs
            stages.append(item)
        final = self.read
        out = {"strategy": self.strategy, "logprobs": final.logprobs,
               "probabilities": final.probabilities, "requests": self.requests,
               "prompt_tokens": self.prompt_tokens, "logical_reads": self.logical_reads,
               "missing_from_top": final.missing_from_top, "ranks": final.ranks,
               "sampled": final.sampled, "components": stages}
        if self.thought is not None:
            out["thought_tokens"] = self.thought_tokens
            out["thought_closed"] = self.thought.closed
            if debug:
                out["thought"] = self.thought.text
        return out


def read_policy(
    readout: Readout,
    state: Any,
    question: Question,
    *,
    policy: str = "repeat-think",
    think_gate: float = THINK_GATE,
    think_budget: int = THINK_BUDGET,
    debug: bool = False,
) -> PolicyRead:
    """Answer one text question under a DE-2 policy."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {list(POLICIES)}")
    started = time.perf_counter()
    # Only a text choice goes past the letters; a noul or a score keeps the 26 cap.
    limit = MAX_CODES if question.type == "choice" else MAX_OPTIONS
    question.validate(limit)
    slots = readout.code_slots(len(question.options()))

    def done(strategy: str, components: list[tuple[str, LetterRead]], thought: Thought | None = None) -> PolicyRead:
        return PolicyRead(strategy, components, thought,
                          elapsed_ms=round((time.perf_counter() - started) * 1000, 2))

    if policy == "direct":
        prompt = readout.render(state, question, max_options=limit)
        return done(DIRECT, [(DIRECT, readout.read_codes(prompt, slots, debug=debug))])

    prompt = readout.render(state, question, repeat=2, max_options=limit)
    first = readout.read_codes(prompt, slots, debug=debug)
    unsure = max(first.probabilities.values()) < think_gate
    if policy == "repeat" or not unsure or len(slots) > THINK_OPTION_CAP:
        return done(REPEAT, [(REPEAT, first)])

    thought = readout.think(readout.render(state, question, thinking=True), think_budget, debug=debug)
    # The model's own close token may be missing when the budget cut the thought,
    # so the close is always written here and the codes are read after it.
    after = thought.prompt_ids + thought.token_ids + [readout.answer_prefix_id()]
    return done(REPEAT_THINK, [(REPEAT, first), ("think", readout.read_codes(after, slots, debug=debug))], thought)


__all__ = ["DEFAULT_POLICY", "DIRECT", "FAMILY_POLICIES", "POLICIES", "PolicyRead", "REPEAT",
           "REPEAT_THINK", "THINK_BUDGET", "THINK_GATE", "THINK_OPTION_CAP", "read_policy"]
