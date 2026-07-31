from __future__ import annotations

from collections.abc import Iterator, Sequence

from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    LanguageModel,
    LlmBatchPrompt,
    LlmBatchWorkerHandler,
    LlmCompletion,
    LlmProvider,
    LlmStreamEvent,
    LlmWorkerHandler,
    ProviderCapabilities,
    build_llm_generate_batch_command,
    build_llm_generate_command,
)
from l1_foundation.worker import WorkerExecutionContext


class FakeModel(LanguageModel):
    def __init__(self) -> None:
        self.released = False

    @property
    def provider(self) -> LlmProvider:
        return LlmProvider.GEMINI

    @property
    def model_name(self) -> str:
        return "gemini-test"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, json_object=True, strict_json_schema=True)

    def complete(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> LlmCompletion:
        return LlmCompletion("完成", self.provider, self.model_name)

    def stream(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> Iterator[LlmStreamEvent]:
        yield LlmStreamEvent("流", self.provider, self.model_name)
        yield LlmStreamEvent("式", self.provider, self.model_name, finish_reason="stop")
        yield LlmStreamEvent("", self.provider, self.model_name, prompt_tokens=12, completion_tokens=2)

    def release(self) -> None:
        self.released = True


class FakeContext(WorkerExecutionContext):
    def __init__(self) -> None:
        self.deltas: list[tuple[str, str | None]] = []
        self.progress: list[float] = []

    @property
    def is_cancel_requested(self) -> bool:
        return False

    def raise_if_cancelled(self) -> None:
        return

    def report_progress(self, progress: float, message: str | None = None) -> None:
        self.progress.append(progress)

    def emit_delta(self, text: str, item_id: str | None = None) -> None:
        self.deltas.append((text, item_id))


def test_llm_worker_command_serializes_provider_and_operation() -> None:
    command = build_llm_generate_command(
        LlmProvider.GEMINI,
        [ChatMessage(ChatRole.USER, "测试")],
        CompletionOptions(max_tokens=20),
        context_size=8192,
        stream=False,
    )

    assert command.operation == "llm.generate.gemini"
    assert command.input.provider == LlmProvider.GEMINI
    assert command.input.messages[0].content == "测试"


def test_llm_worker_handler_supports_streaming() -> None:
    command = build_llm_generate_command(
        LlmProvider.GEMINI,
        [ChatMessage(ChatRole.USER, "测试")],
        CompletionOptions(max_tokens=20),
        context_size=8192,
        stream=True,
    )
    context = FakeContext()
    result = LlmWorkerHandler(LlmProvider.GEMINI, lambda _context_size: FakeModel())(command.input, context)

    assert result.text == "流式"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 2
    assert context.deltas == [("流", None), ("式", None)]


def test_non_streaming_llm_uses_interruptible_stream_without_emitting_deltas() -> None:
    command = build_llm_generate_command(
        LlmProvider.GEMINI,
        [ChatMessage(ChatRole.USER, "测试")],
        CompletionOptions(max_tokens=20),
        context_size=8192,
        stream=False,
    )
    context = FakeContext()
    result = LlmWorkerHandler(LlmProvider.GEMINI, lambda _context_size: FakeModel())(command.input, context)

    assert result.text == "流式"
    assert context.deltas == []


def test_llm_batch_handler_reuses_one_model_and_infers_items_sequentially() -> None:
    created = 0
    created_models: list[FakeModel] = []

    def factory(_context_size: int) -> FakeModel:
        nonlocal created
        created += 1
        model = FakeModel()
        created_models.append(model)
        return model

    prompts = [LlmBatchPrompt(str(index), [ChatMessage(ChatRole.USER, f"测试 {index}")], CompletionOptions(max_tokens=20), stream=True) for index in range(2)]
    command = build_llm_generate_batch_command(LlmProvider.GEMINI, prompts, context_size=8192)
    context = FakeContext()
    item_handler = LlmWorkerHandler(LlmProvider.GEMINI, factory)
    batch_handler = LlmBatchWorkerHandler(LlmProvider.GEMINI, item_handler)
    result = batch_handler(command.input, context)

    assert created == 1
    assert [item.item_id for item in result.items] == ["0", "1"]
    assert context.deltas == [("流", "0"), ("式", "0"), ("流", "1"), ("式", "1")]
    assert context.progress == [0, 0.5, 1]
    assert not created_models[0].released

    batch_handler.release()

    assert created_models[0].released
