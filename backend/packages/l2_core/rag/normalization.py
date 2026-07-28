from __future__ import annotations

import unicodedata


def normalize_search_text(value: str) -> str:
    """Normalize indexed text and lexical queries with one stable rule set."""

    return " ".join(unicodedata.normalize("NFKC", value).lower().split())
