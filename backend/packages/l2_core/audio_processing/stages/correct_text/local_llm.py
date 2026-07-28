from __future__ import annotations

import gc
import json
import re
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel


class CorrectedItem(BaseModel):
    unit_index: int
    text: str


class CorrectedItemsOutput(BaseModel):
    items: list[CorrectedItem]


class LlamaModel(Protocol):
    def __call__(self, prompt: str, **kwargs: object) -> Mapping[str, object]: ...


class LlamaFactory(Protocol):
    def __call__(self, *, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool) -> LlamaModel: ...


class LlamaCppModule(Protocol):
    Llama: LlamaFactory


LLM_CORRECTION_SYSTEM_PROMPT = (
    "你是半导体会议录音转写文本的专业校对器。",
    "原始文本来自语音识别，可能存在同音字、近音字、错别字、英文缩写误识别、标点和断句错误。",
    "专业词表是纠错候选，不是补全素材。可以使用专业词表修正错误，但不能无依据地添加原文中没有的概念。",
    "当原文词语发音接近专业词、上下文属于相关专业讨论、或原文语义明显不通时，可以主动修正为更合理的专业表达。",
    "遇到低频或者语义不通的词时，优先考虑发音相近的常见词。",
    "不要总结，不要扩写，不要补充新结论，不要加入原文没有表达的信息。",
    "允许做轻度口语化清理，但只能删除能够明确判断为无实际语义的句内填充词、口吃和机械重复。",
    "“呃”“额”“嗯啊”等词只有在仅用于停顿、且删除后不改变句意时才可以删除。",
    "“这个”“那个”“就是”“然后”“对”“好”“嗯”等词只有在能够明确判断为无意义填充词时才可以删除；如果承担指代、承接、确认、否定、态度或逻辑关系，必须保留。",
    "独立出现的“嗯”“好”“对”“不是”“可以”等短回复可能代表真实回应或结论，必须保留。",
    "只有明显属于口吃的相邻重复才可以合并；用于强调、确认或列举的重复不得删除。",
    "不要为了简洁而概括、改写或压缩原文，不得删除数字、专业术语、否定词、程度词、条件、结论和行动项。",
    "如果不能确定某个词是否属于无意义语气词，原样保留。",
    "校正后的文本需要与原始录音重新对齐，应尽量保持原有语序，只做少量、确定性的删除和修正，禁止大范围重写。",
    "数字类信息尽量统一写成阿拉伯数字，包括数量、序号、日期、时间、时长、金额、比例、百分比、尺寸、频率、温度、工艺节点、型号、版本号和带单位的数值。",
    (
        "中文口语数字要按语义转换，例如“二零二六年”写作“2026 年”，“十五分钟”写作“15 分钟”，"
        "“百分之三十”写作“30%”，“一百二十纳米”写作“120 纳米”，“两三次”写作“2-3 次”。"
    ),
    "不要把固定词、成语、专有名词或普通词里的中文数字强行改成阿拉伯数字，例如“一方面”“一会儿”“三极管”“二极管”“四舍五入”等应按原意保留。",
    "保留原文中的字母、符号和单位，除非它们明显是识别错误。",
    "不得合并、拆分、删除或重新编号待校正单元。",
    '只输出 JSON 对象，格式为 {"items":[{"unit_index":数字,"text":"校正文本"}]}，不要输出解释。',
)
MAX_LLM_HOTWORDS = 500
MAX_LLM_PHRASES = 40
MAX_LLM_PEOPLE = 40


class LocalLlmCorrector:
    """Lazily load and reuse llama-cpp inside the GPU-normal worker process."""

    def __init__(
        self,
        model_path: Path,
        context_size: int,
        prompt_config: Mapping[str, object],
        batch_max_units: int = 16,
        batch_max_chars: int = 4000,
        context_units: int = 1,
    ) -> None:
        self._model_path = model_path
        self._context_size = context_size
        self._prompt_config = prompt_config
        self._batch_max_units = batch_max_units
        self._batch_max_chars = batch_max_chars
        self._context_units = context_units
        self._model: LlamaModel | None = None

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
        model = self._load_model()
        labels = list(speaker_labels or ("Unknown Speaker" for _ in texts))
        if len(labels) != len(texts):
            raise ValueError("Speaker label count must match correction text count")
        batches = self._batches(texts)
        total = len(batches)
        if report_progress is not None:
            report_progress(0, total)
        output = list(texts)
        for completed, indexes in enumerate(batches, start=1):
            corrected = self._correct_batch(model, texts, labels, indexes)
            for index, text in zip(indexes, corrected, strict=True):
                output[index] = text
            if report_progress is not None:
                report_progress(completed, total)
        return output

    def release(self) -> None:
        """Release the large correction model before another GPU-normal stage loads its model."""
        self._model = None
        gc.collect()

    def _load_model(self) -> LlamaModel:
        if self._model is not None:
            return self._model
        if not self._model_path.is_file():
            raise FileNotFoundError(f"LLM model file not found: {self._model_path}")
        try:
            module = cast(LlamaCppModule, import_module("llama_cpp"))
            factory = module.Llama
        except (ImportError, AttributeError) as error:
            raise RuntimeError("llama-cpp-python is not installed; start the GPU worker with backend/.venv") from error
        try:
            self._model = factory(model_path=str(self._model_path), n_ctx=self._context_size, n_gpu_layers=-1, verbose=False)
        except Exception:
            try:
                self._model = factory(model_path=str(self._model_path), n_ctx=self._context_size, n_gpu_layers=0, verbose=False)
            except Exception as error:
                raise RuntimeError("Unable to initialize the local LLM on either GPU or CPU") from error
        return self._model

    def _correct_batch(
        self,
        model: LlamaModel,
        texts: Sequence[str],
        speaker_labels: Sequence[str],
        indexes: list[int],
    ) -> list[str]:
        response = model(
            self._build_prompt(texts, speaker_labels, indexes),
            max_tokens=max(256, min(4096, sum(len(texts[index]) for index in indexes) * 4)),
            temperature=0,
            stop=["<|im_end|>"],
            echo=False,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return [texts[index] for index in indexes]
        candidate = cast(Mapping[str, object], choices[0]).get("text")
        return self._parse_batch(str(candidate or ""), texts, indexes)

    def _build_prompt(self, texts: Sequence[str], speaker_labels: Sequence[str], indexes: list[int]) -> str:
        hotwords = "、".join(self._limited_strings("hotwords", MAX_LLM_HOTWORDS))
        phrases = "；".join(self._limited_strings("phrases", MAX_LLM_PHRASES))
        people = "、".join(self._limited_strings("people", MAX_LLM_PEOPLE))
        first = indexes[0]
        last = indexes[-1]
        before = range(max(0, first - self._context_units), first)
        after = range(last + 1, min(len(texts), last + 1 + self._context_units))
        payload = {
            "context_before": [self._prompt_item(index, texts, speaker_labels) for index in before],
            "items": [self._prompt_item(index, texts, speaker_labels) for index in indexes],
            "context_after": [self._prompt_item(index, texts, speaker_labels) for index in after],
        }
        user = (
            f"专业词表：{hotwords}\n固定表达：{phrases}\n人名：{people}\n"
            "只校正 items；context_before 和 context_after 仅供理解，禁止输出。\n"
            f"输入 JSON：{json.dumps(payload, ensure_ascii=False)}"
        )
        return f"<|im_start|>system\n{'\n'.join(LLM_CORRECTION_SYSTEM_PROMPT)}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    def _batches(self, texts: Sequence[str]) -> list[list[int]]:
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
