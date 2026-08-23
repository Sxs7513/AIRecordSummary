from __future__ import annotations

from l2_core.audio_processing.stages.summary.retrieval_text import build_summary_retrieval_text


def _characters(value: str) -> int:
    return len(value)


def test_builds_recording_overview_from_global_summary_and_headings() -> None:
    result = build_summary_retrieval_text(
        "周一 15点03分.m4a",
        """
### 全局总结

这段录音主要记录了项目技术方案汇报，以及专家对路演材料和答辩注意事项的修改建议。

---

### 项目背景与技术方案汇报

项目主要解决 AI 算力互联瓶颈。后续还有很多技术细节。

### 答辩注意事项与后续安排

答辩当天必须穿正装；面对专家提问不能当场反驳。
""",
        count_tokens=_characters,
        max_tokens=512,
    )

    assert "录音标题：周一 15点03分.m4a" in result
    assert "项目技术方案汇报" in result
    assert "答辩注意事项" in result
    assert "内容结构" in result
    assert "章节要点" in result


def test_falls_back_to_preamble_without_markdown_headings() -> None:
    result = build_summary_retrieval_text(
        "普通录音",
        "这是一场产品评审会议，参与者讨论了上线风险。\n\n第二段包含详细实施计划。",
        count_tokens=_characters,
        max_tokens=512,
    )

    assert "全局概述：\n这是一场产品评审会议，参与者讨论了上线风险。" in result
    assert "第二段包含详细实施计划" not in result


def test_keeps_output_within_token_budget() -> None:
    result = build_summary_retrieval_text(
        "长录音",
        "## 全局总结\n" + "项目汇报和专家评审。" * 100 + "\n## 答辩安排\n" + "确认答辩时间。" * 100,
        count_tokens=_characters,
        max_tokens=80,
    )

    assert len(result) <= 80
    assert result.startswith("录音标题：长录音")


def test_empty_summary_still_indexes_recording_title() -> None:
    result = build_summary_retrieval_text("项目答辩录音", "  ", count_tokens=_characters, max_tokens=512)

    assert result == "录音标题：项目答辩录音"
