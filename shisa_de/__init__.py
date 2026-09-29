"""Shisa DE-1: typed questions in, typed answers out.

    >>> from shisa_de import DecisionModel
    >>> de = DecisionModel.from_pretrained("shisa-ai/shisa-de-1")
    >>> de.classify({"ticket": "I was charged twice."}, {"intent": ["refund", "cancel"]})
    {'intent': 'refund'}

The readout contract, with full samples, is in `docs/READOUT.md`. The model is
served rather than loaded, so nothing here downloads weights.
"""

from .calibration import Calibration, confidence, load_calibration, temper_binary, temper_distribution
from .client import DEFAULT_ENDPOINT, DEFAULT_MODEL, Answer, Decision, DecisionModel
from .images import ImageError
from .questions import MAX_OPTIONS, Choice, Noul, Question, QuestionError, Score
from .readout import DIRECT_SYSTEM, LETTERS, READOUT_VERSION, LetterRead, Readout, ReadoutError, Slot, softmax

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Calibration",
    "Choice",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "DIRECT_SYSTEM",
    "Decision",
    "DecisionModel",
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
    "confidence",
    "load_calibration",
    "softmax",
    "temper_binary",
    "temper_distribution",
]
