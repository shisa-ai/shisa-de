"""Calibration and confidence for DE-1 answers.

Two numbers are kept apart on purpose:

- **confidence** is the statistic TypeSafe returns on Choice and Score answers,
  ``(K * p_max - 1) / (K - 1)``, so the same code can read hosted Jev and DE-1.
  It is a convenience, not a calibration guarantee.
- **calibrated probability** is the tempered distribution. DE-1 arrives
  near-calibrated and temperature scaling sharpens it further; the temperatures
  shipped here were fitted on dev splits of the committed suites.

Every answer says which calibration was applied, because a threshold fitted
against one readout or one serving shape does not transfer to another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

CALIBRATION_FILE = "data/calibration.json"


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

    @property
    def id(self) -> str:
        """A short identifier recorded on every answer."""
        parts = ",".join(f"{key}={value}" for key, value in sorted(self.temperatures.items()))
        return f"{self.serving_shape}[{parts}]"

    def temperature_for(self, question_type: str) -> float:
        """The temperature for a question type. Score distributions use the choice fit."""
        key = "noul" if question_type == "noul" else "choice"
        return float(self.temperatures.get(key, 1.0))


def load_calibration(name: str = CALIBRATION_FILE) -> Calibration:
    """Load the shipped calibration record."""
    text = resources.files("shisa_de").joinpath(name).read_text(encoding="utf-8")
    raw = json.loads(text)
    return Calibration(
        temperatures=dict(raw["temperatures"]),
        readout_version=raw["readout_version"],
        model=raw["model"],
        serving_shape=raw["serving_shape"],
        fitted_on=raw["fitted_on"],
        test_ece=dict(raw.get("test_ece") or {}),
        source=raw.get("source", ""),
    )


__all__ = [
    "Calibration",
    "confidence",
    "load_calibration",
    "temper_binary",
    "temper_distribution",
]
