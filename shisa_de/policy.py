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

`ReadOptions` holds the settings one question is read under, and
`resolve_options` layers them: the family's defaults, then the model's, then
one call's. The option names (`reads`, `reasoning`, `compound`) select among the
reads the two contracts define; a combination neither defines is an error
rather than an unmeasured read.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .overflow import OVERFLOW_STRATEGY
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

READS = ("single", "double")
OVERFLOWS = (OVERFLOW_STRATEGY, "error")
#: Each policy as (reads, reasoning). A single read that reasons is not one of them.
_POLICY_PARTS = {"direct": ("single", False), "repeat": ("double", False), "repeat-think": ("double", True)}
_PARTS_POLICY = {parts: name for name, parts in _POLICY_PARTS.items()}


@dataclass(frozen=True)
class ReadOptions:
    """The settings one question is read under."""

    family: str
    policy: str
    think_gate: float = THINK_GATE
    think_budget: int = THINK_BUDGET
    overflow: str = OVERFLOW_STRATEGY

    @property
    def reads(self) -> str:
        """`single` or `double`: how many times the user turn is written."""
        return _POLICY_PARTS[self.policy][0]

    @property
    def reasoning(self) -> bool:
        """Whether an unsure read is followed by a thought and a second read."""
        return _POLICY_PARTS[self.policy][1]

    @property
    def reasoning_prob(self) -> float:
        return self.think_gate

    @property
    def reasoning_len(self) -> int:
        return self.think_budget

    @property
    def compound(self) -> bool:
        """Whether a DE-1 text choice above 26 options is read in two rounds."""
        return self.family == "de1" and self.overflow == OVERFLOW_STRATEGY


def _one_of(old_name: str, old: Any, new_name: str, new: Any) -> Any:
    """The value of a setting that has two names; both may be given if they agree."""
    if old is not None and new is not None and old != new:
        raise ValueError(f"{new_name}={new!r} and {old_name}={old!r} set the same thing; pass one")
    return new if new is not None else old


def resolve_options(
    family: str,
    base: ReadOptions | None = None,
    *,
    policy: str | None = None,
    reads: str | None = None,
    reasoning: bool | None = None,
    reasoning_prob: float | None = None,
    reasoning_len: int | None = None,
    compound: bool | None = None,
    think_gate: float | None = None,
    think_budget: int | None = None,
    overflow: str | None = None,
) -> ReadOptions:
    """Layer one level of settings over `base`, or over the family's defaults.

    Anything left as `None` is inherited. `policy` names a whole read; `reads`
    and `reasoning` change one part of the inherited one. `reasoning=True`
    reads twice first and `reads="single"` does not reason, because the
    contract's only thinking read follows a repeated one. `reasoning_prob` /
    `think_gate`, `reasoning_len` / `think_budget` and `compound` / `overflow`
    are the same settings under two names.
    """
    if base is None:
        base = ReadOptions(family, DEFAULT_POLICY[family])
    if policy is not None and policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {list(POLICIES)}")
    if reads is not None and reads not in READS:
        raise ValueError(f"reads must be 'single' or 'double', got {reads!r}")
    for name, flag in (("reasoning", reasoning), ("compound", compound)):
        if flag is not None and not isinstance(flag, bool):
            raise ValueError(f"{name} must be a boolean")
    if reads == "single" and reasoning:
        raise ValueError("reasoning=True follows a repeated read; it cannot be combined with reads='single'")

    count, thinks = _POLICY_PARTS[policy or base.policy]
    if reasoning is not None:
        thinks = reasoning
        count = "double" if reasoning else count
    if reads is not None:
        count = reads
        thinks = thinks and reads == "double"
    resolved = _PARTS_POLICY[(count, thinks)]
    if policy is not None and resolved != policy:
        given = ", ".join(f"{name}={value!r}" for name, value in (("reads", reads), ("reasoning", reasoning))
                          if value is not None)
        raise ValueError(f"policy={policy!r} disagrees with {given}; pass one or the other")
    if resolved not in FAMILY_POLICIES[family]:
        raise ValueError(
            f"policy {resolved!r} is not defined for {family}; "
            f"expected one of {list(FAMILY_POLICIES[family])}"
        )

    gate = _one_of("think_gate", think_gate, "reasoning_prob", reasoning_prob)
    if gate is None:
        gate = base.think_gate
    elif isinstance(gate, bool) or not isinstance(gate, (int, float)) or not 0.0 <= gate <= 1.0:
        raise ValueError("reasoning_prob (think_gate) must be a probability")
    budget = _one_of("think_budget", think_budget, "reasoning_len", reasoning_len)
    if budget is None:
        budget = base.think_budget
    elif isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("reasoning_len (think_budget) must be a positive integer")

    if overflow is not None and overflow not in OVERFLOWS:
        raise ValueError("overflow must be 'finalist-top1' or 'error'")
    if compound and family != "de1":
        raise ValueError(
            f"compound=True is not defined for {family}: it reads a text choice of up to "
            f"{MAX_CODES} options in one prompt"
        )
    wide = _one_of("overflow", overflow, "compound",
                   None if compound is None else OVERFLOWS[0] if compound else OVERFLOWS[1])
    return ReadOptions(family, resolved, float(gate), budget, wide or base.overflow)


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


__all__ = ["DEFAULT_POLICY", "DIRECT", "FAMILY_POLICIES", "POLICIES", "PolicyRead", "READS", "REPEAT",
           "REPEAT_THINK", "ReadOptions", "THINK_BUDGET", "THINK_GATE", "THINK_OPTION_CAP", "read_policy",
           "resolve_options"]
