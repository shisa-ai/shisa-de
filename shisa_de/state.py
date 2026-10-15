"""State text canonicalization for the readout.

The scaffold renders the state into the prompt as JSON, so a quote character at
the start or end of a state string arrives escaped, and the served checkpoints
read those delimiters as content rather than as punctuation. Measured through
this client against `shisa-ai/shisa-de-2` on the hosted Shisa API on
2026-10-15, three reads per variant of the same question: the bare text
`What are my balances currently?` and the quoted `"What are my balances
currently?"` were classified differently in every read, and an unmatched
trailing quote moved the answer further than an unmatched leading one.

`strip_quote_delimiters` is the opt-in fix. It removes a run of the same
delimiter at an end of a string when every occurrence of that character sits in
the leading or trailing run, so `"text"`, `"text`, `text"`, `""text` and
`text""` all render as `text` while interior quoting such as `He said "hi"`
is left alone. It is deliberately not the default: the readout contract renders
the state verbatim, removing characters changes answers for quote-bearing
states, and only the caller knows whether a delimiter carries meaning for its
task.
"""

from __future__ import annotations

from typing import Any

#: The characters treated as quote delimiters at a string's edges: the ASCII
#: quotes and backtick, plus the curly single and double quotes.
QUOTE_DELIMITERS = "\"'`\u201c\u201d\u2018\u2019"


def strip_quote_delimiters(value: Any) -> Any:
    """Remove delimiter quote characters from the edges of every string.

    A run of the same quote delimiter at either end is removed when every
    occurrence of that character sits in the leading or trailing run: an
    unmatched stray (`"hi`, `hi"`), a doubled stray (`""hi`, `hi""`), or a
    wrapping pair (`"hi"`, `""hi""`). Interior quoting is kept, so
    `He said "hi"` and `"a" and "b"` are unchanged. Mappings, lists and tuples
    are walked; keys and non-string values are returned untouched.
    """
    if isinstance(value, str):
        return _strip_one(value)
    if isinstance(value, dict):
        return {key: strip_quote_delimiters(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_quote_delimiters(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_quote_delimiters(item) for item in value)
    return value


def _strip_one(text: str) -> str:
    if not text:
        return text
    first, last = text[0], text[-1]
    lead = _edge_run(text, 0, 1) if first in QUOTE_DELIMITERS else 0
    trail = _edge_run(text, len(text) - 1, -1) if last in QUOTE_DELIMITERS else 0
    if first == last and lead + trail >= len(text):
        return ""  # the string is nothing but delimiters
    front = lead if lead and text.count(first) == lead + (trail if last == first else 0) else 0
    back = trail if trail and text.count(last) == trail + (lead if first == last else 0) else 0
    return text[front:len(text) - back]


def _edge_run(text: str, start: int, step: int) -> int:
    """The length of the run of one character from an end of the string."""
    length = 0
    index = start
    while 0 <= index < len(text) and text[index] == text[start]:
        length += 1
        index += step
    return length
