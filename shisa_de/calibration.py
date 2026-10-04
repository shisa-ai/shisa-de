"""Calibration and confidence for DE answers.

Two numbers are kept apart on purpose:

- **confidence** is the statistic TypeSafe returns on Choice and Score answers,
  ``(K * p_max - 1) / (K - 1)``, so the same code can read hosted Jev and DE.
  It is a convenience, not a calibration guarantee.
- **calibrated probability** is the tempered distribution. DE-1 arrives
  near-calibrated and temperature scaling sharpens it further; the temperatures
  shipped here were fitted on dev splits of the committed suites.

One record ships per family, because a temperature fitted on one checkpoint
provides no evidence about another. Every answer says which calibration was
applied, including the model it was fitted on, because a threshold fitted
against one readout, one serving shape, or one checkpoint does not transfer to
another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .family import model_family

CALIBRATION_FILE = "data/calibration.json"

#: One record per family. A temperature fitted on one checkpoint does not
#: transfer to another, so the record a caller gets has to match the model.
CALIBRATION_FILES = {"de1": CALIBRATION_FILE, "de2": "data/calibration-de2.json"}


def temper_binary(p: float, temperature: float) -> float:
    """Binary temperature scaling: ``p^(1/T) / (p^(1/T) + (1-p)^(1/T))``."""
    if temperature <= 0 or temperature == 1:
        return p
    a = max(p, 1e-12) ** (1.0 / temperature)
    b = max(1.0 - p, 1e-12) ** (1.0 / temperature)
    return a / (a + b)


def temper_distribution(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    """Multiclass temperature scaling: ``p_i^(1/T)`` renormalized. The argmax is unchanged."""
    if temperature <= 0 or temperature == 1:
        return dict(probabilities)
    powered = {key: max(value, 1e-12) ** (1.0 / temperature) for key, value in probabilities.items()}
    total = sum(powered.values()) or 1.0
    return {key: value / total for key, value in powered.items()}


def confidence(probabilities: dict[str, float]) -> float:
    """The confidence statistic hosted Jev returns, derived from the distribution.

    ``(K * p_max - 1) / (K - 1)``, clipped to 0..1: 1.0 when all mass is on one
    option, 0 when the distribution is even. TypeSafe documents this formula in
    its confidence page and states that callers are not locked into it, which is
    why the raw ``probabilities`` are always returned beside it.
    """
    if not probabilities:
        return 0.0
    k = len(probabilities)
    if k < 2:
        return 1.0
    peak = max(probabilities.values())
    return max(0.0, min(1.0, (k * peak - 1.0) / (k - 1.0)))


@dataclass(frozen=True)
class Calibration:
    """A fitted temperature set, tied to a readout version and a serving shape."""

    temperatures: dict[str, float]
    readout_version: str
    model: str
    serving_shape: str
    fitted_on: str
    test_ece: dict[str, Any]
    source: str
    family: str = "de1"
    note: str = ""

    @property
    def id(self) -> str:
        """A short identifier recorded on every answer.

        Carries the model and the readout version, not just the temperatures.
        The fitted model is the field that decides whether a record applies to
        the endpoint in front of it, so leaving it out of the identifier is how
        a DE-1 record came to be applied to DE-2 answers without anyone seeing
        it. A reader of one answer can now tell which checkpoint the number
        beside it was fitted on.
        """
        parts = ",".join(f"{key}={value}" for key, value in sorted(self.temperatures.items()))
        return f"{self.model}|{self.readout_version}|{self.serving_shape}[{parts}]"

    def temperature_for(self, question_type: str) -> float:
        """The temperature for a question type. Score distributions use the choice fit."""
        key = "noul" if question_type == "noul" else "choice"
        return float(self.temperatures.get(key, 1.0))

    def matches(self, model: str, readout_version: str) -> tuple[bool, list[str]]:
        """Whether this record applies to a served model under a readout.

        Returns `(ok, reasons)`. The reasons name each field that disagrees, so
        a caller can report the mismatch instead of a bare false.
        """
        reasons: list[str] = []
        if self.family != model_family(model):
            reasons.append(
                f"fitted for family {self.family!r} but {model!r} is {model_family(model)!r}"
            )
        if self.readout_version != readout_version:
            reasons.append(
                f"fitted against readout {self.readout_version!r} but this client reads "
                f"{readout_version!r}"
            )
        return (not reasons), reasons


def load_calibration(name: str = CALIBRATION_FILE) -> Calibration:
    """Load a calibration record shipped inside the package."""
    text = resources.files("shisa_de").joinpath(name).read_text(encoding="utf-8")
    return calibration_from_dict(json.loads(text), origin=name)


def load_calibration_file(path: str | Path) -> Calibration:
    """Load a calibration record from an arbitrary filesystem path."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"no calibration record at {target}")
    return calibration_from_dict(json.loads(target.read_text(encoding="utf-8")), origin=str(target))


def calibration_from_dict(raw: dict[str, Any], origin: str = "") -> Calibration:
    """Build a record from its JSON form, defaulting the family from the model."""
    model = raw["model"]
    return Calibration(
        temperatures=dict(raw["temperatures"]),
        readout_version=raw["readout_version"],
        model=model,
        serving_shape=raw["serving_shape"],
        fitted_on=raw["fitted_on"],
        test_ece=dict(raw.get("test_ece") or {}),
        source=raw.get("source", ""),
        family=raw.get("family") or model_family(model),
        note=raw.get("note", ""),
    )


def calibration_for(family: str) -> Calibration:
    """The shipped record for a family."""
    if family not in CALIBRATION_FILES:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(CALIBRATION_FILES)}")
    return load_calibration(CALIBRATION_FILES[family])


def resolve_calibration(spec: str | None) -> Calibration | None:
    """Resolve a `--calibration` argument.

    Accepts a family name (`de1`, `de2`) or a path to a record. `None` or
    `"auto"` means "let the client pick from the served model id".
    """
    if spec is None or spec == "auto":
        return None
    if spec in CALIBRATION_FILES:
        return calibration_for(spec)
    return load_calibration_file(spec)


__all__ = [
    "CALIBRATION_FILES",
    "Calibration",
    "calibration_for",
    "calibration_from_dict",
    "confidence",
    "load_calibration",
    "load_calibration_file",
    "resolve_calibration",
    "temper_binary",
    "temper_distribution",
]
