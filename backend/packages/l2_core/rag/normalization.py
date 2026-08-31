from __future__ import annotations

import unicodedata


def normalize_search_text(value: str) -> str:
    """Normalize indexed text and lexical queries with one stable rule set.

    Unicode punctuation becomes a separator so punctuation differences do not
    affect keyword matching. NFKC handles full-width forms, and the final join
    collapses all whitespace runs.
    """

    normalized = unicodedata.normalize("NFKC", value).lower()
    without_punctuation = "".join(" " if unicodedata.category(character).startswith("P") else character for character in normalized)
    return " ".join(without_punctuation.split())
