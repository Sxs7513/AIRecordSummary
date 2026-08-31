from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    LlmBatchPrompt,
    LlmGenerateBatchResult,
    LlmProvider,
    ResponseFormat,
    ResponseFormatType,
    build_llm_generate_batch_command,
)
from l1_foundation.worker import SyncWorkerClient


class CorrectedItem(BaseModel):
    unit_index: int
    text: str


class CorrectedItemsOutput(BaseModel):
    items: list[CorrectedItem]


LLM_CORRECTION_SYSTEM_PROMPT = (
    "你是半导体会议录音转写文本的专业校对器。",
    "在不改变原意的前提下，修正语音识别造成的同音字、近音字、错别字、断词、英文缩写、专业术语、标点、数字和单位错误。",
    "结合录音上下文、专业语境和词表判断；词表只是纠错候选，不得据此添加原文没有表达的概念。",
    "可以修复上下文能够明确支持的局部漏字、错词和语序问题，但不得总结、扩写、压缩或大范围重写。",
    "必须保留原文中的事实、数字、单位、否定、条件、程度、结论和行动项；只有上下文能够明确确认时才能修改。",
    "只删除能够明确判断为无语义填充的语气词、口吃和机械重复；独立短回复以及具有确认、否定、态度或承接作用的表达必须保留。",
    "可以按中文书写习惯规范数字、符号和单位，但不得改变数值、范围、精度或含义。",
    "输出必须保留原有标点，并在语义和停顿明确的句界补全规范中文标点；不得为了简写而删除标点。",
    "输出是面向阅读的普通文本；变量、下标和希腊字母直接以文本书写，不使用公式排版或公式包裹。",
    "无法确定是否需要修改时，保留原文。",
    "必须保持每个校对单元的编号、边界、顺序和内容归属，不得合并、拆分、删除、增加、重排或跨单元移动内容。",
)
MAX_LLM_HOTWORDS = 500
MAX_LLM_PHRASES = 40
MAX_LLM_PEOPLE = 40


class LlmCorrector:
    """Apply correction by submitting serialized LLM tasks to the L1 Worker API."""

    def __init__(
        self,
        worker_client: SyncWorkerClient,
        provider: LlmProvider,
        context_size: int,
        model_name: str,
        prompt_config: Mapping[str, object],
        batch_max_units: int = 16,
        batch_max_chars: int = 4000,
        context_units: int = 1,
        max_output_tokens: int = 65_536,
    ) -> None:
        self._worker_client = worker_client
        self._provider = provider
        self._context_size = context_size
        self._model_name = model_name
        self._prompt_config = prompt_config
        self._batch_max_units = batch_max_units
        self._batch_max_chars = batch_max_chars
        self._max_output_tokens = max_output_tokens
        self._context_units = context_units

    @property
    def provider(self) -> LlmProvider:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    def correct(
        self,
        texts: Sequence[str],
        report_progress: Callable[[int, int], None] | None = None,
        speaker_labels: Sequence[str] | None = None,
    ) -> list[str]:
        if not texts:
            if report_progress is not None:
                report_progress(0, 0)
            return []
        labels = list(speaker_labels or ("Unknown Speaker" for _ in texts))
        if len(labels) != len(texts):
            raise ValueError("Speaker label count must match correction text count")
        batches = self._batches(texts)
        total = len(batches)
        if report_progress is not None:
            report_progress(0, total)
        output = list(texts)
        prompts = [
            LlmBatchPrompt(
                item_id=str(batch_index),
                messages=self._build_messages(texts, labels, indexes),
                options=self._completion_options(texts, indexes),
            )
            for batch_index, indexes in enumerate(batches)
        ]
        response = self._worker_client.execute(
            build_llm_generate_batch_command(
                self._provider,
                prompts,
                context_size=self._context_size,
            ),
            result_type=LlmGenerateBatchResult,
            on_progress=(
                (lambda progress, _message: report_progress(min(total, round(progress * total)), total))
                if report_progress is not None
                else None
            ),
        )
        by_id = {item.item_id: item.result.text for item in response.items}
        if len(by_id) != total:
            raise RuntimeError(f"LLM correction batch result count mismatch: expected {total}, got {len(by_id)}")
        for completed, indexes in enumerate(batches, start=1):
            candidate = by_id.get(str(completed - 1))
            if candidate is None:
                raise RuntimeError(f"LLM correction batch result is missing item {completed - 1}")
            corrected = self._parse_batch(candidate, texts, indexes)
            for index, text in zip(indexes, corrected, strict=True):
                output[index] = text
        return output

    def _completion_options(self, texts: Sequence[str], indexes: list[int]) -> CompletionOptions:
        estimated_output_tokens = sum(len(texts[index]) for index in indexes) * 4 + len(indexes) * 64 + 256
        provider_limit = min(4096, self._max_output_tokens) if self._provider == LlmProvider.LOCAL else self._max_output_tokens
        return CompletionOptions(
            max_tokens=min(provider_limit, max(256, estimated_output_tokens)),
            temperature=0,
            response_format=ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA,
                json_schema=CorrectedItemsOutput.model_json_schema(),
                strict=False,
            ),
        )

    def _build_messages(self, texts: Sequence[str], speaker_labels: Sequence[str], indexes: list[int]) -> list[ChatMessage]:
        hotwords = "、".join(self._limited_strings("hotwords", MAX_LLM_HOTWORDS))
        phrases = "；".join(self._limited_strings("phrases", MAX_LLM_PHRASES))
        people = "、".join(self._limited_strings("people", MAX_LLM_PEOPLE))
        glossary = f"专业词表：{hotwords}\n固定表达：{phrases}\n人名：{people}\n"
        user = (
            self._build_local_user_prompt(texts, speaker_labels, indexes, glossary)
            if self._provider == LlmProvider.LOCAL
            else self._build_online_user_prompt(texts, speaker_labels, indexes, glossary)
        )
        return [
            ChatMessage(ChatRole.SYSTEM, "\n".join(LLM_CORRECTION_SYSTEM_PROMPT)),
            ChatMessage(ChatRole.USER, user),
        ]

    def _build_local_user_prompt(
        self,
        texts: Sequence[str],
        speaker_labels: Sequence[str],
        indexes: list[int],
        glossary: str,
    ) -> str:
        first = indexes[0]
        last = indexes[-1]
        before = range(max(0, first - self._context_units), first)
        after = range(last + 1, min(len(texts), last + 1 + self._context_units))
        payload = {
            "context_before": [self._prompt_item(index, texts, speaker_labels) for index in before],
            "items": [self._prompt_item(index, texts, speaker_labels) for index in indexes],
            "context_after": [self._prompt_item(index, texts, speaker_labels) for index in after],
        }
        return "".join(
            (
                glossary,
                "context_before、items 和 context_after 均来自同一份录音，并按原始时间顺序排列。",
                "前后 context 是 items 的相邻录音片段，只用于理解指代、术语、语义和句子衔接；只能校正并输出 items。",
                "相邻片段可能跨越不同说话人，也可能因 ASR 滑动窗口存在少量内容重叠；",
                "不要假设它们属于同一说话人，不要把窗口重叠误判为口吃或无意义重复。",
                "必须为 items 中的每个 unit_index 返回且只返回一项，即使无需修改也必须原样返回。\n",
                f"输入 JSON：{json.dumps(payload, ensure_ascii=False)}",
            )
        )

    @staticmethod
    def _build_online_user_prompt(
        texts: Sequence[str],
        speaker_labels: Sequence[str],
        indexes: list[int],
        glossary: str,
    ) -> str:
        payload = {"items": [LlmCorrector._prompt_item(index, texts, speaker_labels) for index in indexes]}
        return "".join(
            (
                glossary,
                "items 中的所有单元均来自同一份录音，并按原始时间顺序排列，是本次需要校正的完整录音文本。",
                "请结合全文上下文校正每个单元。相邻单元可能跨越不同说话人，也可能因 ASR 滑动窗口存在少量内容重叠；",
                "不要假设它们属于同一说话人，不要把窗口重叠误判为口吃或无意义重复。",
                "必须为每个 unit_index 返回且只返回一项，即使无需修改也必须原样返回。\n",
                f"输入 JSON：{json.dumps(payload, ensure_ascii=False)}",
            )
        )

    def _batches(self, texts: Sequence[str]) -> list[list[int]]:
        if self._provider != LlmProvider.LOCAL:
            return [list(range(len(texts)))]
        batches: list[list[int]] = []
        pending: list[int] = []
        pending_chars = 0
        for index, text in enumerate(texts):
            if pending and (len(pending) >= self._batch_max_units or pending_chars + len(text) > self._batch_max_chars):
                batches.append(pending)
                pending = []
                pending_chars = 0
            pending.append(index)
            pending_chars += len(text)
        if pending:
            batches.append(pending)
        return batches

    @staticmethod
    def _prompt_item(index: int, texts: Sequence[str], speaker_labels: Sequence[str]) -> dict[str, object]:
        return {"unit_index": index, "speaker": speaker_labels[index], "text": texts[index]}

    @staticmethod
    def _parse_batch(candidate: str, source: Sequence[str], indexes: list[int]) -> list[str]:
        cleaned = candidate.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced is not None:
            cleaned = fenced.group(1)
        else:
            object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if object_match is not None:
                cleaned = object_match.group(0)
        try:
            parsed = CorrectedItemsOutput.model_validate_json(cleaned)
        except ValueError:
            return [source[index] for index in indexes]
        resolved: dict[int, str] = {}
        for item in parsed.items:
            text = item.text.strip()
            if item.unit_index in resolved or item.unit_index not in indexes or not text:
                return [source[index] for index in indexes]
            resolved[item.unit_index] = text
        if set(resolved) != set(indexes):
            return [source[index] for index in indexes]
        return [resolved[index] for index in indexes]

    def _limited_strings(self, key: str, limit: int) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in self._strings(self._prompt_config.get(key)):
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _strings(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in cast(list[object], value) if str(item).strip()]
