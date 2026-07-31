from __future__ import annotations

import json
from collections.abc import Sequence

from l2_core.audio_processing.stages.recording_models import Utterance


def build_topic_boundary_prompt(utterances: Sequence[Utterance]) -> str:
    payload: list[dict[str, int | str]] = [
        {
            "utterance_index": item.utterance_index,
            "speaker": item.speaker_label,
            "text": item.text,
        }
        for item in utterances
    ]
    system = (
        "你负责识别录音中时间连续的话题区间，不回答录音内容。"
        "必须让输入中的每个 utterance_index 恰好属于一个区间，区间必须连续、升序、无重叠、无缺口。"
        "不要因为两个不连续区间话题相似就合并它们。"
        "每个区间还要输出：topic 为明确、简短的标准主题；terms 为最多 8 个有助于检索且能由原文支持的标准术语；"
        "search_context 用一句话消除口语省略、指代和上下文依赖，说明这段对话实际在讨论或询问什么。"
        "terms 和 search_context 只做语义显式化，不得编造原文没有的实体、数值、结论或事实；无法可靠补充时使用空数组或 null。"
        '只输出 JSON：{"sections":[{"start_utterance_index":0,"end_utterance_index":2,'
        '"topic":"简短标准主题","terms":["标准术语"],"search_context":"一句话语义上下文"}]}。'
    )
    user = f"连续发言：{json.dumps(payload, ensure_ascii=False)}"
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
