"""Which DE family a served model belongs to.

The family decides more than a calibration record. DE-1 and DE-2 are read
through different contracts (`docs/READOUT.md` and `docs/READOUT-DE2.md`): DE-2
reads the user turn twice, may think before answering, and binds up to 256
options to a codebook. Reading a checkpoint through the other family's contract
returns answers nobody measured, so the family is resolved explicitly and its
source is reported.

Resolution order:

1. a declared family (`DecisionModel(family=...)`, `--family`);
2. a slug in the served model id (`shisa-ai/shisa-de-1`, `de2-v4-...`);
3. a slug in the tokenizer source, for a server that aliases the model id;
4. `DEFAULT_FAMILY`, as an assumption. DE-2 is the forward line, so an id that
   says nothing is read as DE-2, and the client warns when it does so.
"""

from __future__ import annotations

import re

#: An explicit DE-1 marker: `shisa-ai/shisa-de-1`, `de-1`, `de1-cont-...`.
#: The boundaries keep it from matching inside a longer token, so a name like
#: `de-13` or `de1x` is not read as DE-1.
DE1_SLUG = re.compile(r"(?<![a-z0-9])de-?1(?![a-z0-9])", re.IGNORECASE)

#: An explicit DE-2 marker.
DE2_SLUG = re.compile(r"(?<![a-z0-9])de-?2(?![a-z0-9])", re.IGNORECASE)

FAMILIES = ("de1", "de2")

#: The family assumed when nothing names one.
DEFAULT_FAMILY = "de2"


def _slug_family(text: str | None) -> str | None:
    """The one family a string names, or `None` when it names neither or both."""
    text = text or ""
    de1, de2 = bool(DE1_SLUG.search(text)), bool(DE2_SLUG.search(text))
    if de1 == de2:
        return None
    return "de1" if de1 else "de2"


def family_is_explicit(model_id: str | None) -> bool:
    """Whether the id names exactly one family, rather than being assumed."""
    return _slug_family(model_id) is not None


def model_family(model_id: str | None) -> str:
    """The family an id names, otherwise `DEFAULT_FAMILY`.

    The default is an assumption, not a detection: use `family_is_explicit` or
    `resolve_family` when the difference matters. An id naming both families
    (`de2-vs-de1-ablation`) names neither.
    """
    return _slug_family(model_id) or DEFAULT_FAMILY


def normalize_family(spec: str) -> str:
    """`de1`, `DE-1`, `de_2` and the like, as a member of `FAMILIES`."""
    name = re.sub(r"[^a-z0-9]", "", str(spec).lower())
    if name not in FAMILIES:
        raise ValueError(f"unknown family {spec!r}; expected one of {list(FAMILIES)}")
    return name


def resolve_family(model_id: str | None, tokenizer: str | None = None,
                   family: str | None = None) -> tuple[str, str]:
    """`(family, source)`, where source is `declared`, `model id`, `tokenizer` or `assumed`."""
    if family is not None:
        return normalize_family(family), "declared"
    named = _slug_family(model_id)
    if named:
        return named, "model id"
    named = _slug_family(tokenizer)
    if named:
        return named, "tokenizer"
    return DEFAULT_FAMILY, "assumed"


__all__ = ["DE1_SLUG", "DE2_SLUG", "DEFAULT_FAMILY", "FAMILIES", "family_is_explicit",
           "model_family", "normalize_family", "resolve_family"]
