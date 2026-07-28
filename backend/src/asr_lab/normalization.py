from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

_WHITESPACE = re.compile(r"\s+")
_IGNORED_PUNCTUATION = frozenset("，。！？；：、“”‘’（）()【】[]《》〈〉…—,.!?;:\"'")


def strict_v1(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def zh_asr_v1(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    without_punctuation = "".join(character for character in normalized if character not in _IGNORED_PUNCTUATION)
    return _WHITESPACE.sub("", without_punctuation).strip()


NORMALIZERS: dict[tuple[str, str], Callable[[str], str]] = {
    ("strict", "v1"): strict_v1,
    ("zh_asr", "v1"): zh_asr_v1,
}


def normalize_text(value: str, name: str, version: str) -> str:
    try:
        normalizer = NORMALIZERS[(name, version)]
    except KeyError as error:
        raise ValueError(f"Unsupported normalization: {name}_{version}") from error
    return normalizer(value)

