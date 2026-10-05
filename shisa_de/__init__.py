"""Shisa DE: typed questions in, typed answers out.

    >>> from shisa_de import DecisionModel
    >>> de = DecisionModel.from_pretrained("shisa-ai/shisa-de-1")
    >>> de.classify({"ticket": "I was charged twice."}, {"intent": ["refund", "cancel"]})
    {'intent': 'refund'}

The readout contract, with full samples, is in `docs/READOUT.md`. The model is
served rather than loaded, so nothing here downloads weights.

DE-1 and DE-2 are read through different contracts (`docs/READOUT-DE2.md` for
what DE-2 changes). The client picks the contract from the served model id, then
the tokenizer source, and otherwise assumes DE-2 with a warning; pass `family=`
to declare it.
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
from .family import DEFAULT_FAMILY, FAMILIES, family_is_explicit, model_family, resolve_family
from .images import ImageError
from .policy import POLICIES, THINK_BUDGET, THINK_GATE, THINK_OPTION_CAP, PolicyRead, read_policy
from .questions import MAX_OPTIONS, Choice, Noul, Question, QuestionError, Score
from .readout import (
    DE2_READOUT_VERSION,
    DIRECT_SYSTEM,
    LETTERS,
    MAX_CODES,
    READOUT_VERSION,
    READOUT_VERSIONS,
    LetterRead,
    Readout,
    ReadoutError,
    Slot,
    codes_for,
    softmax,
)

__version__ = "0.3.0"

__all__ = [
    "Answer",
    "CALIBRATION_FILES",
    "Calibration",
    "Choice",
    "DEFAULT_ENDPOINT",
    "DEFAULT_FAMILY",
    "DEFAULT_MODEL",
    "DE2_READOUT_VERSION",
    "DIRECT_SYSTEM",
    "Decision",
    "DecisionModel",
    "FAMILIES",
    "ImageError",
    "LETTERS",
    "LetterRead",
    "MAX_CODES",
    "MAX_OPTIONS",
    "POLICIES",
    "PolicyRead",
    "Noul",
    "Question",
    "QuestionError",
    "READOUT_VERSION",
    "READOUT_VERSIONS",
    "Readout",
    "ReadoutError",
    "Score",
    "Slot",
    "THINK_BUDGET",
    "THINK_GATE",
    "THINK_OPTION_CAP",
    "__version__",
    "calibration_for",
    "calibration_from_dict",
    "codes_for",
    "confidence",
    "family_is_explicit",
    "load_calibration",
    "load_calibration_file",
    "model_family",
    "read_policy",
    "resolve_family",
    "resolve_calibration",
    "softmax",
    "temper_binary",
    "temper_distribution",
]
