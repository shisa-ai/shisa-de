"""Question objects: the three System One primitives, in DE-1's terms.

Each object serializes to the wire shape the TypeSafe API documents, so the same
question can be sent to hosted Jev or to a DE-1 endpoint. See
``docs/READOUT.md`` for how each one is rendered and read.

    >>> Choice("Which queue?", {"billing": "Charges and refunds"}).to_wire()
    {'type': 'choice', 'instructions': 'Which queue?', 'criteria': {'billing': 'Charges and refunds'}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")

#: DE-1 reads one answer letter per question, so a question can offer at most
#: this many options or levels. TypeSafe's hosted API allows 255 options and 10
#: levels; the readout is the binding limit here.
MAX_OPTIONS = 26


class QuestionError(ValueError):
    """A question cannot be rendered or is outside what the readout supports."""


@dataclass
class Question:
    """Base class for the three question types."""

    instructions: str
    type: str = field(init=False, default="")

    def to_wire(self) -> dict[str, Any]:
        raise NotImplementedError

    def options(self) -> list[tuple[str, Any]]:
        """(answer key, rendered description) pairs, in presentation order.

        The readout assigns letters in this order, so the first pair is `A`.
        """
        raise NotImplementedError

    def validate(self) -> None:
        if not self.instructions or not str(self.instructions).strip():
            raise QuestionError("instructions must be a non-empty question")
        count = len(self.options())
        if count < 2:
            raise QuestionError(f"{self.type} question needs at least two answers, got {count}")
        if count > MAX_OPTIONS:
            raise QuestionError(
                f"{self.type} question offers {count} answers; DE-1 reads one letter per "
                f"answer and supports at most {MAX_OPTIONS} per prompt. "
                "Use DecisionModel for text choice overflow."
            )


@dataclass
class Noul(Question):
    """A yes/no question. The answer is the probability that it is yes."""

    criteria: dict[str, str] | None = None
    type: str = field(init=False, default="noul")

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            wire["criteria"] = dict(self.criteria)
        return wire

    def options(self) -> list[tuple[str, Any]]:
        return [("yes", "Yes"), ("no", "No")]


@dataclass
class Choice(Question):
    """One option out of a set. The answer is the chosen option plus the distribution."""

    criteria: dict[str, Any] = field(default_factory=dict)
    type: str = field(init=False, default="choice")

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": {key: value for key, value in self.criteria.items()},
        }

    def options(self) -> list[tuple[str, Any]]:
        return [(key, render_option(key, value)) for key, value in self.criteria.items()]


@dataclass
class Score(Question):
    """A position along ordered levels. The answer is a weighted position plus the distribution."""

    criteria: list[Any] = field(default_factory=list)
    type: str = field(init=False, default="score")

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }

    def options(self) -> list[tuple[str, Any]]:
        return [(str(index), level) for index, level in enumerate(self.criteria)]


def render_option(key: str, description: Any) -> Any:
    """What the model is shown for one option.

    Some label sets carry the label in the key and no description. Showing an
    empty description leaves the model with indistinguishable options, so the
    key is shown instead. This follows the renderer fix recorded in the
    research notes, where discarding keys cost 0.129 to 0.948 accuracy on
    `dbpedia14`.
    """
    if description is None:
        return key
    if isinstance(description, str) and not description.strip():
        return key
    return description


__all__ = ["Choice", "MAX_OPTIONS", "Noul", "QUESTION_TYPES", "Question", "QuestionError", "Score", "render_option"]
