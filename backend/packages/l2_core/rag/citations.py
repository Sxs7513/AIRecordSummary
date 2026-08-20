from __future__ import annotations

import re
from dataclasses import dataclass

from l2_core.rag.contracts import EvidenceSource

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class NormalizedCitations:
    text: str
    sources: list[EvidenceSource]
    cited_indexes: tuple[int, ...]
    invalid_indexes: tuple[int, ...]


def normalize_answer_citations(text: str, sources: list[EvidenceSource]) -> NormalizedCitations:
    """Keep cited sources and renumber citations by first appearance in the answer."""
    sources_by_index = {
        index: source
        for source in sources
        if (index := source["index"]) >= 1
    }
    remapped_indexes: dict[int, int] = {}
    invalid_indexes: list[int] = []

    def replace(match: re.Match[str]) -> str:
        original_index = int(match.group(1))
        if original_index not in sources_by_index:
            if original_index not in invalid_indexes:
                invalid_indexes.append(original_index)
            return ""
        normalized_index = remapped_indexes.setdefault(original_index, len(remapped_indexes) + 1)
        return f"[{normalized_index}]"

    normalized_text = _CITATION_PATTERN.sub(replace, text)
    normalized_sources: list[EvidenceSource] = []
    for original_index, normalized_index in remapped_indexes.items():
        source = sources_by_index[original_index].copy()
        source["index"] = normalized_index
        normalized_sources.append(source)
    return NormalizedCitations(
        text=normalized_text,
        sources=normalized_sources,
        cited_indexes=tuple(remapped_indexes),
        invalid_indexes=tuple(invalid_indexes),
    )
