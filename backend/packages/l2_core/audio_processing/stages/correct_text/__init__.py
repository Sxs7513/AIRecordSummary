from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from l1_foundation.llm import LlmProvider
from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.correct_text.llm import LlmCorrector
from l2_core.audio_processing.stages.correct_text.pycorrector import correct_texts_with_pycorrector
from l2_core.audio_processing.stages.recording_models import (
    AsrWindowTranscriptOutput,
    CorrectAsrWindowsInput,
    CorrectedAsrWindowTranscript,
    CorrectedAsrWindowTranscriptOutput,
)

logger = logging.getLogger("audio_processing")


class LocalTextCorrector:
    """Apply pycorrector, deterministic rules, and the configured L1 LLM provider."""

    def __init__(
        self,
        pycorrector_enabled: bool,
        llm_enabled: bool,
        worker_client: SyncWorkerClient | None,
        llm_provider: LlmProvider | None,
        llm_context_size: int,
        llm_model_name: str | None,
        prompt_config_path: Path,
        llm_batch_max_units: int = 16,
        llm_batch_max_chars: int = 4000,
        llm_context_units: int = 1,
        llm_max_output_tokens: int = 65_536,
    ) -> None:
        self._pycorrector_enabled = pycorrector_enabled
        self._llm_enabled = llm_enabled
        self._prompt_config = self._load_prompt_config(prompt_config_path)
        self._llm = (
            LlmCorrector(
                worker_client=worker_client,
                provider=llm_provider,
                context_size=llm_context_size,
                model_name=llm_model_name,
                prompt_config=self._prompt_config,
                batch_max_units=llm_batch_max_units,
                batch_max_chars=llm_batch_max_chars,
                context_units=llm_context_units,
                max_output_tokens=llm_max_output_tokens,
            )
            if llm_enabled and worker_client is not None and llm_provider is not None and llm_model_name is not None
            else None
        )
        if llm_enabled and self._llm is None:
            raise ValueError("worker_client and LLM task configuration are required when llm_enabled is true")

    async def correct(
        self,
        texts: Sequence[str],
        report_progress: Callable[[int, str], None] | None = None,
        speaker_labels: Sequence[str] | None = None,
    ) -> list[str]:
        def ignore_progress(_percent: int, _message: str) -> None:
            return

        report: Callable[[int, str], None] = report_progress or ignore_progress
        report(5, "清理待润色文本")
        current = [self._clean(text) for text in texts]
        if self._pycorrector_enabled:
            report(10, "使用 pycorrector 校正文本")
            current = await self._with_fallback(
                asyncio.to_thread(correct_texts_with_pycorrector, current, self._protected_terms()),
                current,
                "pycorrector",
            )
            report(35, "pycorrector 校正完成")
        else:
            report(35, "跳过 pycorrector 校正")
        current = [self._apply_rules(text) for text in current]
        report(40, "术语和文本规则校正完成")
        if not self._llm_enabled:
            report(95, "文本润色完成")
            return current
        llm = self._llm
        if llm is None:
            raise RuntimeError("LLM corrector is not configured")
        is_local_model = llm.provider == LlmProvider.LOCAL
        logger.info(
            "文本润色：开始 provider=%s model=%s texts=%d",
            llm.provider.value,
            llm.model_name,
            len(current),
        )
        report(45, "加载文本润色模型" if is_local_model else "开始文本润色")
        last_llm_percent = 45

        def report_llm_progress(completed: int, total: int) -> None:
            nonlocal last_llm_percent
            if completed == 0:
                return
            percent = 50 + round(42 * completed / max(1, total))
            if percent <= last_llm_percent:
                return
            last_llm_percent = percent
            report(percent, f"文本润色 {completed}/{total}")

        polished = await self._with_fallback(
            asyncio.to_thread(llm.correct, current, report_llm_progress, speaker_labels),
            current,
            "llm",
        )
        output = [self._apply_rules(text) for text in polished]
        report(95, "文本润色完成")
        return output

    def release(self) -> None:
        return

    async def _with_fallback(self, operation: Awaitable[list[str]], fallback: list[str], provider: str) -> list[str]:
        try:
            return await operation
        except Exception:
            logger.exception("文本校正：%s 执行失败，保留上一阶段结果", provider)
            return fallback

    def _protected_terms(self) -> list[str]:
        values = self._config_strings("hotwords") + self._config_strings("people")
        return list(dict.fromkeys(values))

    def _apply_rules(self, text: str) -> str:
        if self._prompt_config.get("enabled") is False:
            return self._clean(text)
        output = text
        replacements = self._prompt_config.get("replacements")
        if isinstance(replacements, dict):
            for before, after in sorted(cast(Mapping[str, object], replacements).items(), key=lambda pair: len(pair[0]), reverse=True):
                if before:
                    output = output.replace(before, str(after))
        regex_replacements = self._prompt_config.get("regexReplacements")
        if isinstance(regex_replacements, list):
            for item in cast(list[object], regex_replacements):
                if not isinstance(item, dict):
                    continue
                replacement = cast(Mapping[str, object], item)
                pattern = replacement.get("pattern")
                if not isinstance(pattern, str) or not pattern:
                    continue
                flags = re.IGNORECASE if "i" in str(replacement.get("flags") or "") else re.NOFLAG
                output = re.sub(pattern, str(replacement.get("replace") or ""), output, flags=flags)
        return self._clean(output)

    def _config_strings(self, key: str) -> list[str]:
        values = self._prompt_config.get(key)
        if not isinstance(values, list):
            return []
        return [str(value).strip() for value in cast(list[object], values) if str(value).strip()]

    @staticmethod
    def _load_prompt_config(path: Path) -> dict[str, object]:
        if not path.is_file():
            return {}
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Prompt configuration must be a JSON object: {path}")
        return cast(dict[str, object], parsed)

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


class CorrectAsrWindowsStage:
    """Moderately correct each ASR window before it is forced-aligned."""

    name = "correct_asr_windows"
    version = "4"
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = CorrectAsrWindowsInput

    def __init__(
        self,
        artifact_store: ArtifactStore,
        corrector: LocalTextCorrector,
        provider: Literal["pycorrector_llm", "pycorrector", "llm", "rules"],
        model_name: str | None,
        max_edit_ratio: float = 0.35,
    ) -> None:
        self._artifact_store = artifact_store
        self._corrector = corrector
        self._provider: Literal["pycorrector_llm", "pycorrector", "llm", "rules"] = provider
        self._model_name = model_name
        self._max_edit_ratio = max_edit_ratio

    async def try_restore(
        self,
        context: StageContext,
        _input_payload: CorrectAsrWindowsInput,
    ) -> StageResult[CorrectedAsrWindowTranscriptOutput] | None:
        return self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "transcript.corrected_windows",
            CorrectedAsrWindowTranscriptOutput,
        )

    async def run(self, context: StageContext, input_payload: CorrectAsrWindowsInput) -> StageResult[CorrectedAsrWindowTranscriptOutput]:
        raw = AsrWindowTranscriptOutput.model_validate(self._artifact_store.read_json(input_payload.transcript))
        try:
            texts = await self._corrector.correct([item.text for item in raw.windows], context.report_progress) if raw.windows else []
            if len(texts) != len(raw.windows):
                raise ValueError("Window corrector did not preserve window count")
            windows: list[CorrectedAsrWindowTranscript] = []
            for item, corrected in zip(raw.windows, texts, strict=True):
                edit_ratio = self._edit_ratio(item.text, corrected)
                final_text = corrected if edit_ratio <= self._max_edit_ratio else item.text
                if final_text == item.text and corrected != item.text:
                    logger.warning(
                        "窗口文本校正变更过大，回退 ASR 原文：window=%d edit_ratio=%.3f max_edit_ratio=%.3f",
                        item.window_index,
                        edit_ratio,
                        self._max_edit_ratio,
                    )
                windows.append(CorrectedAsrWindowTranscript(**item.model_dump(exclude={"text"}), original_text=item.text, text=final_text))
            output = CorrectedAsrWindowTranscriptOutput(
                asr_provider=raw.provider,
                asr_model_name=raw.model_name,
                correction_provider=self._provider,
                correction_model_name=self._model_name,
                language=raw.language,
                windows=windows,
            )
            return StageResult(
                output=output,
                artifacts=(ArtifactPayload(artifact_type="transcript.corrected_windows", data=output.model_dump(mode="json")),),
            )
        finally:
            self._corrector.release()

    @staticmethod
    def _edit_ratio(original: str, corrected: str) -> float:
        if original == corrected:
            return 0.0
        previous = list(range(len(corrected) + 1))
        for index, left in enumerate(original, start=1):
            current = [index]
            for right_index, right in enumerate(corrected, start=1):
                current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left != right)))
            previous = current
        return previous[-1] / max(1, len(original))


__all__ = ["CorrectAsrWindowsStage", "LocalTextCorrector"]
