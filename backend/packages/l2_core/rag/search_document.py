from __future__ import annotations

from collections.abc import Sequence


def build_retrieval_text(
    text: str,
    topic: str | None = None,
    terms: Sequence[str] = (),
    search_context: str | None = None,
) -> str:
    parts: list[str] = []
    if topic is not None and topic.strip():
        parts.append(f"主题：{topic.strip()}")
    normalized_terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))
    if normalized_terms:
        parts.append(f"标准术语：{'、'.join(normalized_terms)}")
    if search_context is not None and search_context.strip():
        parts.append(f"语义上下文：{search_context.strip()}")
    if not parts:
        return text
    parts.append(f"正文：{text.strip()}")
    return "\n".join(parts)
