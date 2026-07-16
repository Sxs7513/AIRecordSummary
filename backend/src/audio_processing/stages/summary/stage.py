from __future__ import annotations

import gc
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

from audio_processing.stages.recording_models import GenerateSummaryInput, RecordingSummaryOutput, Utterance, UtterancesOutput
from audio_processing.stages.summary.generation import create_pipeline_summary_generation
from generation.emitter import StreamEmitter
from generation.local_llm_runtime import local_llm_inference_lock
from generation.service import GenerationService
from pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult
from pipeline.runtime.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """请总结这段录音，只能根据录音文本写，不要编造。
开头先写一个全局总结，用 1-2 段话说明这段录音整体在讨论什么、最重要的结论或结果是什么。
按照录音里的先后顺序总结，可以自然分成几段，每段围绕一个真实讨论主题或连续发生的事情展开。
每段标题要直接写具体主题，不要使用“阶段一/阶段二/片段一/片段二/第一阶段/第二阶段”这类流程标签。
总结要比逐句复述更高一层，写清楚这段讨论在解决什么问题、形成了什么看法、有哪些结论或待办。
不要机械写成“Speaker A 说……、Speaker B 说……”这种发言记录。只有人物身份本身重要时才提到人。
每段都要保留具体事情、数字、结论和待办，不要只写空泛概括。
用自然的大白话写，不要写成报告腔，也不要只写空泛概括。
用 Markdown 输出。不要使用代码块或缩进代码格式。不要输出思考过程，不要输出 JSON。"""
STOP_TOKENS = ["</s>", "<|im_end|>"]


class LlamaModel(Protocol):
    def __call__(self, prompt: str, **kwargs: object) -> Mapping[str, object] | Iterable[Mapping[str, object]]: ...

    def close(self) -> None: ...


class LlamaFactory(Protocol):
    def __call__(self, *, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool) -> LlamaModel: ...


class LlamaCppModule(Protocol):
    Llama: LlamaFactory


@dataclass(frozen=True, slots=True)
class SummaryChunk:
    index: int
    start_ms: int
    end_ms: int
    utterances: tuple[Utterance, ...]


class SafeTextStream:
    """Hide an optional leading <think> block before it can become a user-visible delta."""

    def __init__(self, on_delta: Callable[[str], None]) -> None:
        self._on_delta = on_delta
        self._pending = ""
        self._state = "initial"

    def feed(self, value: str) -> None:
        if not value:
            return
        self._pending += value
        while self._pending:
            if self._state == "initial":
                stripped = self._pending.lstrip()
                if "<think>".startswith(stripped.lower()):
                    return
                if stripped.lower().startswith("<think>"):
                    self._state = "thinking"
                    self._pending = stripped[len("<think>") :]
                    continue
                self._state = "visible"
            if self._state == "thinking":
                marker = self._pending.lower().find("</think>")
                if marker < 0:
                    self._pending = self._pending[-7:]
                    return
                self._pending = self._pending[marker + len("</think>") :]
                self._state = "visible"
                continue
            self._on_delta(self._pending)
            self._pending = ""

    def finish(self) -> None:
        if self._state == "visible" and self._pending:
            self._on_delta(self._pending)
        self._pending = ""


class GenerateSummaryStage:
    """Summarize with a large context by default, or rolling memory for explicitly enabled long recordings."""

    name = "generate_summary"
    version = "2"
    resource_queue = ResourceQueue.GPU_NORMAL
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = GenerateSummaryInput

    def __init__(
        self,
        artifact_store: ArtifactStore,
        model_path: Path,
        context_size: int,
        prompt_config_path: Path,
        max_output_tokens: int = 4096,
        rolling_enabled: bool = False,
        rolling_threshold_ms: int = 1_800_000,
        rolling_chunk_duration_ms: int = 600_000,
        rolling_chunk_max_chars: int = 8000,
        rolling_chunk_max_tokens: int = 1800,
        rolling_memory_max_chars: int = 6000,
        generation_service: GenerationService | None = None,
        verbose: bool = False,
    ) -> None:
        if context_size <= max_output_tokens:
            raise ValueError("Summary context_size must be greater than max_output_tokens")
        self._artifact_store = artifact_store
        self._model_path = model_path
        self._context_size = context_size
        self._max_output_tokens = max_output_tokens
        self._rolling_enabled = rolling_enabled
        self._rolling_threshold_ms = rolling_threshold_ms
        self._rolling_chunk_duration_ms = rolling_chunk_duration_ms
        self._rolling_chunk_max_chars = rolling_chunk_max_chars
        self._rolling_chunk_max_tokens = rolling_chunk_max_tokens
        self._rolling_memory_max_chars = rolling_memory_max_chars
        self._system_prompt = self._load_system_prompt(prompt_config_path)
        self._generation_service = generation_service
        self._verbose = verbose
        self._model: LlamaModel | None = None
        self._force_cpu = False

    async def run(self, context: StageContext, input_payload: GenerateSummaryInput) -> StageResult[RecordingSummaryOutput]:
        utterances = UtterancesOutput.model_validate(self._artifact_store.read_json(input_payload.utterances)).segments
        emitter = self._create_emitter(context, input_payload)
        if emitter is not None:
            emitter.start()
        try:
            output = self.generate(utterances, context.report_progress, emitter)
            if emitter is not None:
                emitter.succeed(output.model_dump(mode="json"))
        except Exception as error:
            if emitter is not None:
                emitter.fail("summary_generation_failed", str(error), retryable=True)
            raise
        return StageResult(output=output, artifacts=(ArtifactPayload(artifact_type="summary.recording", data=output.model_dump(mode="json")),))

    def generate(
        self, utterances: Sequence[Utterance], report_progress: Callable[[int, str], None], emitter: StreamEmitter | None = None
    ) -> RecordingSummaryOutput:
        """Generate one summary from materialized utterances without pipeline artifact concerns."""
        try:
            if emitter is not None:
                emitter.phase("preparing", "正在准备录音总结", 1)
            summary = self._summarize(utterances, report_progress, emitter) if utterances else "暂无可总结的润色文本。"
            if emitter is not None and not utterances:
                emitter.text(summary)
            return RecordingSummaryOutput(provider="local_llm", model_name=self._model_path.name, summary_text=summary)
        finally:
            self.release()

    def release(self) -> None:
        with local_llm_inference_lock:
            model = self._model
            self._model = None
            self._force_cpu = False
            if model is None:
                gc.collect()
                return
            try:
                model.close()
            except Exception as error:
                logger.warning("summary：模型引用已释放，但 llama.cpp close 失败：%s", error)
            finally:
                del model
                gc.collect()
        logger.info("summary：模型已释放")

    def _create_emitter(self, context: StageContext, input_payload: GenerateSummaryInput) -> StreamEmitter | None:
        if self._generation_service is None:
            return None
        run = create_pipeline_summary_generation(
            self._generation_service,
            cast(UUID, context.stage_run_id),
            cast(UUID, context.subject_id),
            context.attempt_count,
            {"utterances_artifact": input_payload.utterances.model_dump(mode="json"), "strategy": "rolling" if self._rolling_enabled else "large_context"},
        )
        return self._generation_service.emitter(run.id)

    def _summarize(self, utterances: Sequence[Utterance], report_progress: Callable[[int, str], None], emitter: StreamEmitter | None) -> str:
        if self.should_use_rolling_summary(utterances):
            chunks = self.build_rolling_chunks(utterances)
            logger.info("summary：使用滚动总结，片段数=%d", len(chunks))
            report_progress(5, f"滚动总结：准备 {len(chunks)} 个片段")
            if emitter is not None:
                emitter.phase("summarizing", f"正在整理 {len(chunks)} 个录音片段", 5)
            self._load_model()
            return self._run_rolling_summary(chunks, report_progress, emitter)
        summarized_utterances = self._truncate_utterances(utterances, self._input_char_budget())
        logger.info("summary：使用 %d token 大上下文单次总结，发言段数=%d", self._context_size, len(summarized_utterances))
        report_progress(5, f"使用 {self._context_size} token 大上下文生成总结")
        report_progress(25, "加载本地总结模型")
        if emitter is not None:
            emitter.phase("generating", "正在生成录音总结", 25)
        self._load_model()
        prompt = self._build_single_prompt(summarized_utterances)
        summary = self._complete_with_fallback(prompt, self._max_output_tokens, emitter.text if emitter is not None else None)
        report_progress(95, "整理总结结果")
        return summary

    def should_use_rolling_summary(self, utterances: Sequence[Utterance]) -> bool:
        """Return whether this recording needs the configured rolling-memory strategy."""
        if not self._rolling_enabled or not utterances:
            return False
        duration_ms = max(item.end_ms for item in utterances) - min(item.start_ms for item in utterances)
        return duration_ms >= self._rolling_threshold_ms

    def build_rolling_chunks(self, utterances: Sequence[Utterance]) -> list[SummaryChunk]:
        """Split a long recording into bounded chronological inputs for rolling summary."""
        effective_max_chars = min(
            self._rolling_chunk_max_chars,
            max(3000, self._context_size - self._rolling_chunk_max_tokens - self._rolling_memory_max_chars - 1800),
        )
        chunks: list[SummaryChunk] = []
        current: list[Utterance] = []
        current_start_ms: int | None = None
        current_chars = 0
        for utterance in utterances:
            item_size = self._utterance_size(utterance)
            duration = 0 if current_start_ms is None else max(0, utterance.end_ms - current_start_ms)
            if current and (duration > self._rolling_chunk_duration_ms or current_chars + item_size > effective_max_chars):
                chunks.append(SummaryChunk(len(chunks) + 1, current_start_ms or current[0].start_ms, current[-1].end_ms, tuple(current)))
                current = []
                current_start_ms = None
                current_chars = 0
            current_start_ms = utterance.start_ms if current_start_ms is None else current_start_ms
            current.append(utterance)
            current_chars += item_size
        if current:
            chunks.append(SummaryChunk(len(chunks) + 1, current_start_ms or current[0].start_ms, current[-1].end_ms, tuple(current)))
        return chunks

    def _run_rolling_summary(self, chunks: Sequence[SummaryChunk], report_progress: Callable[[int, str], None], emitter: StreamEmitter | None) -> str:
        memory = ""
        chunk_summaries: list[dict[str, object]] = []
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            progress = 10 + round(65 * (index - 1) / max(1, total))
            report_progress(progress, f"滚动总结片段 {index}/{total}")
            if emitter is not None:
                emitter.phase("summarizing", f"正在整理录音片段 {index}/{total}", progress)
            logger.info("summary：滚动总结片段 %d/%d", index, total)
            raw = self._complete_with_fallback(self._build_refine_prompt(chunk, total, memory), self._rolling_chunk_max_tokens)
            parsed = self._parse_json_object(raw)
            if parsed is None:
                chunk_summary = raw
                memory = f"{memory}\n{chunk_summary}".strip()
            else:
                chunk_summary = self._strip_thinking(str(parsed.get("chunkSummary") or ""))
                memory = self._strip_thinking(str(parsed.get("memory") or memory))
            memory = self._truncate_tail(memory, self._rolling_memory_max_chars)
            chunk_summaries.append(
                {
                    "index": chunk.index,
                    "time_range": f"{self._format_ms(chunk.start_ms)}-{self._format_ms(chunk.end_ms)}",
                    "summary": chunk_summary or "该片段无明显可总结内容。",
                }
            )
        report_progress(80, "汇总滚动总结")
        if emitter is not None:
            emitter.phase("generating", "正在生成最终录音总结", 80)
        final_chunks, final_memory = self._fit_final_inputs(chunk_summaries, memory)
        summary = self._complete_with_fallback(
            self._build_final_prompt(final_chunks, final_memory), self._max_output_tokens, emitter.text if emitter is not None else None
        )
        report_progress(95, "整理最终总结")
        return summary

    def _complete_with_fallback(self, prompt: str, max_tokens: int, on_delta: Callable[[str], None] | None = None) -> str:
        try:
            return self._complete_text(prompt, max_tokens, on_delta)
        except RuntimeError as error:
            if self._force_cpu:
                raise
            logger.warning("summary：GPU 初始化或推理失败，回退到 CPU：%s", error)
            self._model = None
            self._force_cpu = True
            return self._complete_text(prompt, max_tokens, on_delta)

    def _complete_text(self, prompt: str, max_tokens: int, on_delta: Callable[[str], None] | None = None) -> str:
        with local_llm_inference_lock:
            response = self._load_model()(prompt, max_tokens=max_tokens, temperature=0.1, stop=STOP_TOKENS, echo=False, stream=on_delta is not None)
            if isinstance(response, Mapping):
                summary = self._completion_text(cast(Mapping[str, object], response))
                if on_delta is not None:
                    logger.warning("summary：llama-cpp 未返回迭代式 stream，最终文本只能一次性写入消息流")
                    on_delta(summary)
                return summary
            safe_stream = SafeTextStream(on_delta) if on_delta is not None else None
            chunks: list[str] = []
            for chunk in response:
                text = self._completion_text(chunk, allow_empty=True)
                chunks.append(text)
                if safe_stream is not None:
                    safe_stream.feed(text)
        if safe_stream is not None:
            safe_stream.finish()
        summary = self._strip_thinking("".join(chunks))
        if not summary:
            raise RuntimeError("Local summary model returned an empty completion")
        return summary

    @staticmethod
    def _completion_text(response: Mapping[str, object], allow_empty: bool = False) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            if allow_empty:
                return ""
            raise RuntimeError("Local summary model returned no completion")
        choice = cast(Mapping[str, object], choices[0])
        return str(choice.get("text") or "")

    def _load_model(self) -> LlamaModel:
        if self._model is not None:
            return self._model
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Local summary model file not found: {self._model_path}")
        try:
            module = cast(LlamaCppModule, import_module("llama_cpp"))
        except ImportError as error:
            raise RuntimeError("llama-cpp-python is not installed; start the GPU worker with backend/.venv") from error
        n_gpu_layers = 0 if self._force_cpu else -1
        try:
            self._model = module.Llama(model_path=str(self._model_path), n_ctx=self._context_size, n_gpu_layers=n_gpu_layers, verbose=self._verbose)
        except Exception as error:
            if self._force_cpu:
                raise RuntimeError("Unable to initialize the local summary model on CPU") from error
            self._force_cpu = True
            return self._load_model()
        return self._model

    def _build_single_prompt(self, utterances: Sequence[Utterance]) -> str:
        return self._chat_prompt(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"录音标题：未命名录音\n\n润色后的录音文本：\n{self._format_utterances(utterances)}"},
            ]
        )

    def _build_refine_prompt(self, chunk: SummaryChunk, total_chunks: int, memory: str) -> str:
        system = (
            f"{self._system_prompt}\n"
            "你正在做滚动记忆式长录音总结。必须基于当前片段和前文滚动记忆，不能编造。\n"
            "当前步骤只输出严格 JSON，不要 Markdown，不要解释。\n"
            'JSON schema: {"chunkSummary":"当前片段的详细总结","memory":"传给后续片段的滚动记忆"}\n'
            "chunkSummary 按原文顺序概括当前内容的讨论重点，不要用“阶段/片段”作为标题，"
            "不要机械写成逐个 speaker 的发言记录。memory 保留后面还会用到的事实、结论和待办。"
        )
        user = (
            f"录音标题：未命名录音\n当前片段：{chunk.index}/{total_chunks}，时间范围：{self._format_ms(chunk.start_ms)}-{self._format_ms(chunk.end_ms)}\n\n"
            f"前文滚动记忆：\n{memory or '无'}\n\n当前片段文本：\n{self._format_utterances(chunk.utterances)}"
        )
        return self._chat_prompt([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def _build_final_prompt(self, chunk_summaries: Sequence[Mapping[str, object]], memory: str) -> str:
        summaries = "\n\n".join(f"片段 {item['index']} [{item['time_range']}]\n{item['summary']}" for item in chunk_summaries)
        system = (
            f"{self._system_prompt}\n"
            "这是长录音的最终综合总结。请基于片段总结和滚动记忆输出给用户看的最终中文总结。\n"
            "片段总结只是内部处理中间结果，最终输出不要出现“片段 1/片段 2/阶段一/阶段二”等内部编号或流程标签。\n"
            "开头先写一个全局总结，用 1-2 段话概括整段录音的核心内容、主要结论或整体结果。\n"
            "按照录音里的先后顺序总结，可以自然分成几段；每段标题要直接写真实主题，而不是写处理阶段。\n"
            "总结要比逐句复述更高一层，不要机械写成逐个 speaker 的发言记录。\n"
            "保留具体事实、数字、结论和待办；只有人物身份本身重要时才提到人。\n"
            "用自然的大白话写，不要写成报告腔，也不要只写空泛概括。\n"
            "不要输出 <think>，不要输出 JSON，不要编造。"
        )
        user = f"录音标题：未命名录音\n\n最终滚动记忆：\n{memory or '无'}\n\n片段总结：\n{summaries}\n\n输出最终总结，不要把长录音压缩成很短一段。"
        return self._chat_prompt([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def _chat_prompt(self, messages: Sequence[dict[str, str]]) -> str:
        model = self._model
        if model is not None:
            template = getattr(model, "metadata", {}).get("tokenizer.chat_template")
            if isinstance(template, str) and template:
                try:
                    formatter_module = import_module("llama_cpp.llama_chat_format")
                    formatter_factory = formatter_module.Jinja2ChatFormatter
                    concrete_model = cast(Any, model)
                    eos_token_id = concrete_model.token_eos()
                    bos_token_id = concrete_model.token_bos()
                    eos_token = concrete_model._model.token_get_text(eos_token_id) if eos_token_id != -1 else "<|im_end|>"
                    bos_token = concrete_model._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""
                    formatter = formatter_factory(
                        template=template,
                        eos_token=eos_token,
                        bos_token=bos_token,
                        stop_token_ids=[eos_token_id] if eos_token_id != -1 else None,
                    )
                    return str(formatter(messages=messages, enable_thinking=False).prompt)
                except Exception:
                    logger.debug("summary：模型 chat template 不可用，回退 ChatML", exc_info=True)
        return "".join(f"<|im_start|>{message['role']}\n{message['content'].strip()}\n<|im_end|>\n" for message in messages) + "<|im_start|>assistant\n"

    def _fit_final_inputs(self, chunk_summaries: Sequence[dict[str, object]], memory: str) -> tuple[list[dict[str, object]], str]:
        budget = max(4000, self._context_size - self._max_output_tokens - 1600)
        memory_budget = min(len(memory), max(1200, budget // 3))
        fitted_memory = self._truncate_tail(memory, memory_budget)
        remaining = max(1200, budget - len(fitted_memory))
        per_chunk = max(400, remaining // max(1, len(chunk_summaries)))
        return [{**item, "summary": self._truncate_head(str(item["summary"]), per_chunk)} for item in chunk_summaries], fitted_memory

    def _input_char_budget(self) -> int:
        return max(1200, self._context_size - self._max_output_tokens - 1200)

    @staticmethod
    def _truncate_utterances(utterances: Sequence[Utterance], max_chars: int) -> list[Utterance]:
        used = 0
        result: list[Utterance] = []
        for utterance in utterances:
            remaining = max_chars - used
            if remaining <= 0:
                break
            text = utterance.text[:remaining]
            item = utterance.model_copy(update={"text": text})
            result.append(item)
            used += GenerateSummaryStage._utterance_size(item)
        return result

    @staticmethod
    def _utterance_size(utterance: Utterance) -> int:
        return len(utterance.text) + len(utterance.speaker_label or "") + 32

    @staticmethod
    def _format_utterances(utterances: Sequence[Utterance]) -> str:
        return "\n".join(
            f"[{GenerateSummaryStage._format_ms(item.start_ms)}-{GenerateSummaryStage._format_ms(item.end_ms)}] "
            f"{item.speaker_label or 'Unknown Speaker'}: {item.text}"
            for item in utterances
        )

    @staticmethod
    def _format_ms(milliseconds: int) -> str:
        total_seconds = max(0, milliseconds // 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    @staticmethod
    def _parse_json_object(value: str) -> dict[str, object] | None:
        start = value.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for index, character in enumerate(value[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif character == "\\":
                    escape = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(value[start : index + 1])
                    except json.JSONDecodeError:
                        return None
                    if not isinstance(payload, dict):
                        return None
                    return {str(key): item for key, item in cast(dict[object, object], payload).items()}
        return None

    @staticmethod
    def _strip_thinking(value: str) -> str:
        value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
        value = re.sub(r"</?think>", "", value, flags=re.IGNORECASE)
        return re.sub(r"^\s*(思考过程|推理过程|分析过程)[:：].*?(?=\n\n|$)", "", value, flags=re.DOTALL).strip()

    @staticmethod
    def _truncate_tail(value: str, max_chars: int) -> str:
        return value if len(value) <= max_chars else value[-max_chars:]

    @staticmethod
    def _truncate_head(value: str, max_chars: int) -> str:
        return value if len(value) <= max_chars else value[:max_chars].rstrip() + "\n[已截断]"

    @staticmethod
    def _load_system_prompt(path: Path) -> str:
        if not path.is_file():
            return DEFAULT_SYSTEM_PROMPT
        raw_payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(raw_payload, dict):
            return DEFAULT_SYSTEM_PROMPT
        payload = cast(dict[str, object], raw_payload)
        config = payload.get("summary", payload)
        if not isinstance(config, dict):
            return DEFAULT_SYSTEM_PROMPT
        summary_config = cast(dict[str, object], config)
        if summary_config.get("enabled") is False:
            return DEFAULT_SYSTEM_PROMPT
        system = summary_config.get("system")
        if not isinstance(system, list):
            return DEFAULT_SYSTEM_PROMPT
        lines = [str(item).strip() for item in cast(list[object], system) if str(item).strip()]
        return "\n".join(lines) or DEFAULT_SYSTEM_PROMPT
