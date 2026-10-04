"""Shisa DE: typed questions in, typed answers out.

    >>> from shisa_de import DecisionModel
    >>> de = DecisionModel.from_pretrained("shisa-ai/shisa-de-1")
    >>> de.classify({"ticket": "I was charged twice."}, {"intent": ["refund", "cancel"]})
    {'intent': 'refund'}

The readout contract, with full samples, is in `docs/READOUT.md`. The model is
served rather than loaded, so nothing here downloads weights.

DE-1 and DE-2 share the letter-slot readout but not a calibration. The client
picks the record from the served model id: an id with an explicit DE-1 slug is
DE-1, anything else is treated as DE-2. Pass `calibration=` to override.
"""

from .calibration import (
    CALIBRATION_FILES,
    Calibration,
    calibration_for,
    calibration_from_dict,
    confidence,
    load_calibration,
    load_calibration_file,
    resolve_calibration,
    temper_binary,
    temper_distribution,
)
from .client import DEFAULT_ENDPOINT, DEFAULT_MODEL, Answer, Decision, DecisionModel
from .family import DEFAULT_FAMILY, FAMILIES, family_is_explicit, model_family
from .images import ImageError
from .questions import MAX_OPTIONS, Choice, Noul, Question, QuestionError, Score
from .readout import DIRECT_SYSTEM, LETTERS, READOUT_VERSION, LetterRead, Readout, ReadoutError, Slot, softmax

__version__ = "0.3.0"

__all__ = [
    "Answer",
    "CALIBRATION_FILES",
    "Calibration",
    "Choice",
    "DEFAULT_ENDPOINT",
    "DEFAULT_FAMILY",
    "DEFAULT_MODEL",
    "DIRECT_SYSTEM",
    "Decision",
    "DecisionModel",
    "FAMILIES",
    "ImageError",
    "LETTERS",
    "LetterRead",
    "MAX_OPTIONS",
    "Noul",
    "Question",
    "QuestionError",
    "READOUT_VERSION",
    "Readout",
    "ReadoutError",
    "Score",
    "Slot",
    "__version__",
    "calibration_for",
    "calibration_from_dict",
    "confidence",
    "family_is_explicit",
    "load_calibration",
    "load_calibration_file",
    "model_family",
    "resolve_calibration",
    "softmax",
    "temper_binary",
    "temper_distribution",
]
