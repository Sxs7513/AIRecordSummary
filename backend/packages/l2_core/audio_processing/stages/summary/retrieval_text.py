from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_GLOBAL_HEADINGS = {
    "全局总结",
    "整体总结",
    "总体总结",
    "内容总结",
    "会议总结",
    "录音总结",
    "会议概述",
    "内容概述",
    "录音概述",
    "总结",
    "概述",
}


@dataclass(frozen=True, slots=True)
class SummarySection:
    heading: str
    body: str


def build_summary_retrieval_text(
    recording_title: str,
    summary_text: str,
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int = 512,
) -> str:
    """Build one bounded recording-level search document without another LLM call."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    normalized = summary_text.strip()
    if not normalized:
        return _fit_text(f"录音标题：{recording_title.strip()}", count_tokens, max_tokens)

    preamble, sections = _parse_markdown_sections(normalized)
    overview = _select_overview(preamble, sections)
    headings = [section.heading for section in sections if not _is_global_heading(section.heading)]
    section_leads = [
        f"{section.heading}：{lead}"
        for section in sections
        if not _is_global_heading(section.heading) and (lead := _first_sentence(section.body))
    ]

    parts = [f"录音标题：{recording_title.strip()}"]
    if overview:
        overview_budget = max(1, min(320, max_tokens * 2 // 3))
        parts.append(f"全局概述：\n{_fit_text(overview, count_tokens, overview_budget)}")
    if headings:
        parts.append("内容结构：\n" + "\n".join(f"- {heading}" for heading in headings))
    if section_leads:
        parts.append("章节要点：\n" + "\n".join(f"- {lead}" for lead in section_leads))
    return _fit_parts(parts, count_tokens, max_tokens)


def _parse_markdown_sections(text: str) -> tuple[str, list[SummarySection]]:
    preamble_lines: list[str] = []
    sections: list[SummarySection] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def finish_section() -> None:
        nonlocal current_lines
        if current_heading is not None:
            sections.append(SummarySection(current_heading, _clean_body("\n".join(current_lines))))
        current_lines = []

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is not None:
            finish_section()
            current_heading = _clean_inline_markdown(match.group(1))
            continue
        if current_heading is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)
    finish_section()
    return _clean_body("\n".join(preamble_lines)), sections


def _select_overview(preamble: str, sections: list[SummarySection]) -> str:
    for section in sections:
        if _is_global_heading(section.heading) and section.body:
            return section.body
    if preamble:
        return _first_paragraph(preamble)
    for section in sections:
        if section.body:
            return _first_paragraph(section.body)
    return ""


def _is_global_heading(value: str) -> bool:
    normalized = re.sub(r"[：:\s]", "", value)
    return normalized in _GLOBAL_HEADINGS or normalized.endswith("全局总结") or normalized.endswith("整体总结")


def _first_paragraph(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs[0] if paragraphs else ""


def _first_sentence(text: str) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    compact = re.sub(r"^[*+-]\s+", "", compact)
    if not compact:
        return ""
    match = re.search(r"[。！？!?；;]", compact)
    return compact[: match.end()].strip() if match is not None else compact[:200].strip()


def _clean_body(value: str) -> str:
    lines: list[str] = []
    blank = False
    for raw in value.splitlines():
        line = raw.strip()
        if not line or line == "---":
            if lines and not blank:
                lines.append("")
                blank = True
            continue
        lines.append(_clean_inline_markdown(line))
        blank = False
    return "\n".join(lines).strip()


def _clean_inline_markdown(value: str) -> str:
    return re.sub(r"[*_`]", "", value).strip()


def _fit_parts(parts: list[str], count_tokens: Callable[[str], int], max_tokens: int) -> str:
    accepted: list[str] = []
    for part in parts:
        candidate = "\n\n".join([*accepted, part])
        if count_tokens(candidate) <= max_tokens:
            accepted.append(part)
            continue
        prefix = "\n\n".join(accepted)
        separator = "\n\n" if prefix else ""
        remaining = _fit_text(f"{prefix}{separator}{part}", count_tokens, max_tokens)
        return remaining.rstrip()
    return "\n\n".join(accepted).rstrip()


def _fit_text(text: str, count_tokens: Callable[[str], int], max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    low = 0
    high = len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        candidate = text[:midpoint].rstrip()
        if count_tokens(candidate) <= max_tokens:
            low = midpoint
        else:
            high = midpoint - 1
    return text[:low].rstrip()
