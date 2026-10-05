"""Question objects: the three System One primitives, in DE terms.

Each object serializes to the wire shape the TypeSafe API documents, so the same
question can be sent to hosted Jev or to a DE-1 or DE-2 endpoint. See
``docs/READOUT.md`` and ``docs/READOUT-DE2.md`` for how each one is rendered and
read.

    >>> Choice("Which queue?", {"billing": "Charges and refunds"}).to_wire()
    {'type': 'choice', 'instructions': 'Which queue?', 'criteria': {'billing': 'Charges and refunds'}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")

#: The letter-slot limit: one answer letter per option, so at most this many
#: options or levels in one prompt. It binds every DE-1 prompt, and every DE-2
#: noul, score and image question; DE-2 text choices go up to the 256-code
#: codebook (`shisa_de.readout.MAX_CODES`). TypeSafe's hosted API allows 255
#: options and 10 levels.
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

    def validate(self, max_options: int = MAX_OPTIONS) -> None:
        """Reject a question the readout cannot render.

        `max_options` is the per-prompt limit of the contract in use: 26 letters
        by default, the 256-code codebook when DE-2 reads a text choice.
        """
        if not self.instructions or not str(self.instructions).strip():
            raise QuestionError("instructions must be a non-empty question")
        count = len(self.options())
        if count < 2:
            raise QuestionError(f"{self.type} question needs at least two answers, got {count}")
        if count > max_options:
            raise QuestionError(
                f"{self.type} question offers {count} answers; the readout binds one code per "
                f"answer and supports at most {max_options} per prompt here. "
                "Use DecisionModel for wide text choices."
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
