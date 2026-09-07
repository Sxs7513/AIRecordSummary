from __future__ import annotations

import unicodedata

_SPOKEN_DIGITS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


def normalize_search_text(value: str) -> str:
    """Normalize indexed text and lexical queries with one stable rule set.

    Unicode punctuation becomes a separator so punctuation differences do not
    affect keyword matching. NFKC handles full-width forms, and the final join
    collapses all whitespace runs.
    """

    normalized = unicodedata.normalize("NFKC", value).lower()
    without_punctuation = "".join(" " if unicodedata.category(character).startswith("P") else character for character in normalized)
    return " ".join(without_punctuation.split())


def expand_spoken_digit_variants(value: str, *, max_variants: int = 16) -> list[str]:
    """Return bounded keyword aliases for digits read individually in Mandarin.

    The original normalized keyword remains first. Arabic digits are then
    rendered as Chinese digits, with both ``一`` and the common spoken identifier
    form ``幺`` considered for every ``1``. This deliberately does not interpret
    cardinal-number grammar such as ``1017`` -> ``一千零一十七``.
    """

    if max_variants <= 0:
        return []
    normalized = normalize_search_text(value)
    if not normalized:
        return []
    variants = [normalized]
    if max_variants == 1 or not any(character.isascii() and character.isdigit() for character in normalized):
        return variants

    one_count = normalized.count("1")
    all_yao_mask = (1 << one_count) - 1
    masks = list(dict.fromkeys([0, all_yao_mask, *range(1, max_variants)]))
    for mask in masks:
        one_index = 0
        rendered: list[str] = []
        for character in normalized:
            if not (character.isascii() and character.isdigit()):
                rendered.append(character)
                continue
            if character == "1":
                rendered.append("幺" if mask & (1 << one_index) else "一")
                one_index += 1
            else:
                rendered.append(_SPOKEN_DIGITS[character])
        candidate = "".join(rendered)
        if candidate not in variants:
            variants.append(candidate)
        if len(variants) >= max_variants:
            break
    return variants
