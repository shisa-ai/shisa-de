"""Which DE family a served model id belongs to.

DE-1 and DE-2 share the letter-slot readout below 27 options, so an endpoint
gives no direct signal about which one is behind it. What differs is the
calibration: a temperature fitted on one checkpoint does not transfer to the
other, and applying the wrong one reports a confidence nobody measured.

The rule is deliberately asymmetric. A model id carrying an explicit DE-1 slug
is DE-1; anything else is treated as DE-2, because DE-2 is the forward line and
an id that says nothing about its family is more likely to be a DE-2 checkpoint
or a base than a DE-1 one. `family_is_explicit` says whether the id actually
carried a slug, so callers can report an assumption as an assumption rather than
presenting it as a detection.
"""

from __future__ import annotations

import re

#: An explicit DE-1 marker: `shisa-ai/shisa-de-1`, `de-1`, `de1-cont-...`.
#: The boundaries keep it from matching inside a longer token, so a name like
#: `de-13` or `de1x` is not read as DE-1.
DE1_SLUG = re.compile(r"(?<![a-z0-9])de-?1(?![a-z0-9])", re.IGNORECASE)

#: An explicit DE-2 marker, used only to tell a declared family from an assumed one.
DE2_SLUG = re.compile(r"(?<![a-z0-9])de-?2(?![a-z0-9])", re.IGNORECASE)

FAMILIES = ("de1", "de2")

#: The family assumed when the id carries no slug at all.
DEFAULT_FAMILY = "de2"


def family_is_explicit(model_id: str | None) -> bool:
    """Whether the id names its own family, rather than being assumed."""
    text = model_id or ""
    return bool(DE1_SLUG.search(text) or DE2_SLUG.search(text))


def model_family(model_id: str | None) -> str:
    """`"de1"` when the id carries an explicit DE-1 slug, otherwise `"de2"`.

    DE-2 is the default, not a detection: see the module docstring. Use
    `family_is_explicit` when the difference matters to the caller.
    """
    if DE1_SLUG.search(model_id or ""):
        return "de1"
    return DEFAULT_FAMILY


__all__ = ["DE1_SLUG", "DE2_SLUG", "DEFAULT_FAMILY", "FAMILIES", "family_is_explicit", "model_family"]
