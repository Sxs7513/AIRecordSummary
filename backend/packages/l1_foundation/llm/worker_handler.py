from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import signature
from time import monotonic
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from l1_foundation.llm.contracts import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    JsonObject,
    LanguageModel,
    LlmCompletion,
    LlmProvider,
    ResponseFormat,
    ResponseFormatType,
    ToolCall,
    ToolDefinition,
)
from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker.contracts import ComputeCommand, WorkerExecutionContext

logger = logging.getLogger("llm")


def _worker_tool_calls() -> list[WorkerToolCall]:
    return []


def _worker_tool_definitions() -> list[WorkerToolDefinition]:
    return []


class WorkerToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: JsonObject
    thought_signature: str | None = None


class WorkerToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: JsonObject


class WorkerChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str = ""
    tool_calls: list[WorkerToolCall] = Field(default_factory=_worker_tool_calls)
    tool_call_id: str | None = None
    name: str | None = None


class WorkerResponseFormat(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: ResponseFormatType = ResponseFormatType.TEXT
    json_schema: JsonObject | None = None
    strict: bool = True

    @model_validator(mode="after")
    def validate_schema(self) -> WorkerResponseFormat:
        if self.type == ResponseFormatType.JSON_SCHEMA and self.json_schema is None:
            raise ValueError("JSON_SCHEMA response format requires json_schema")
        return self


class WorkerCompletionOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(gt=0)
    model: str | None = Field(default=None, min_length=1)
    min_request_interval_seconds: float | None = Field(default=None, ge=0)
    temperature: float = Field(ge=0, le=2)
    response_format: WorkerResponseFormat
    tools: list[WorkerToolDefinition] = Field(default_factory=_worker_tool_definitions)
    tool_choice: Literal["auto", "required", "none"] = "auto"


class LlmGenerateInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: LlmProvider
    messages: list[WorkerChatMessage] = Field(min_length=1)
    options: WorkerCompletionOptions
    context_size: int = Field(gt=0)
    stream: bool
    model_profile: str = "default"


class LlmGenerateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    provider: LlmProvider
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    request_id: str | None = None
    tool_calls: list[WorkerToolCall] = Field(default_factory=_worker_tool_calls)


class LlmGenerateBatchItemInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str = Field(min_length=1)
    request: LlmGenerateInput


class LlmGenerateBatchInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: LlmProvider
    items: list[LlmGenerateBatchItemInput] = Field(min_length=1)


class LlmGenerateBatchItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    result: LlmGenerateResult


class LlmGenerateBatchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[LlmGenerateBatchItemResult]


@dataclass(frozen=True, slots=True)
class LlmBatchPrompt:
    item_id: str
    messages: Sequence[ChatMessage]
    options: CompletionOptions
    stream: bool = False


def build_llm_generate_command(
    provider: LlmProvider,
    messages: Sequence[ChatMessage],
    options: CompletionOptions,
    *,
    context_size: int,
    stream: bool,
    resource_queue: ResourceQueue | None = None,
    model_profile: str = "default",
) -> ComputeCommand[LlmGenerateInput]:
    schema = options.response_format.json_schema
    return ComputeCommand(
        task_id=uuid4(),
        operation=f"llm.generate.{provider.value}",
        operation_version="1",
        resource_queue=resource_queue or (ResourceQueue.GPU_NORMAL if provider == LlmProvider.LOCAL else ResourceQueue.IO),
        wait_for_subscriber=stream,
        input=LlmGenerateInput(
            provider=provider,
            messages=[
                WorkerChatMessage(
                    role=message.role,
                    content=message.content,
                    tool_calls=[
                        WorkerToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=dict(call.arguments),
                            thought_signature=call.thought_signature,
                        )
                        for call in message.tool_calls
                    ],
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
                for message in messages
            ],
            options=WorkerCompletionOptions(
                max_tokens=options.max_tokens,
                model=options.model,
                min_request_interval_seconds=options.min_request_interval_seconds,
                temperature=options.temperature,
                response_format=WorkerResponseFormat(
                    type=options.response_format.type,
                    json_schema=dict(schema) if schema is not None else None,
                    strict=options.response_format.strict,
                ),
                tools=[
                    WorkerToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=dict(tool.parameters),
                    )
                    for tool in options.tools
                ],
                tool_choice=options.tool_choice,
            ),
            context_size=context_size,
            stream=stream,
            model_profile=model_profile,
        ),
    )


def build_llm_generate_batch_command(
    provider: LlmProvider,
    prompts: Sequence[LlmBatchPrompt],
    *,
    context_size: int,
    wait_for_subscriber: bool = False,
) -> ComputeCommand[LlmGenerateBatchInput]:
    if not prompts:
        raise ValueError("LLM batch must contain at least one prompt")
    items: list[LlmGenerateBatchItemInput] = []
    for prompt in prompts:
        command = build_llm_generate_command(
            provider,
            prompt.messages,
            prompt.options,
            context_size=context_size,
            stream=prompt.stream,
        )
        items.append(LlmGenerateBatchItemInput(item_id=prompt.item_id, request=command.input))
    return ComputeCommand(
        task_id=uuid4(),
        operation=f"llm.generate_batch.{provider.value}",
        operation_version="1",
        resource_queue=ResourceQueue.GPU_NORMAL if provider == LlmProvider.LOCAL else ResourceQueue.IO,
        wait_for_subscriber=wait_for_subscriber,
        input=LlmGenerateBatchInput(provider=provider, items=items),
    )


class LlmWorkerHandler:
    """Worker-side LLM operation; L3 supplies the provider-specific model factory."""

    def __init__(
        self,
        provider: LlmProvider,
        model_factory: Callable[[int], LanguageModel] | Callable[[int, str], LanguageModel],
    ) -> None:
        self._provider = provider
        self._legacy_model_factory: Callable[[int], LanguageModel] | None = None
        self._model_factory: Callable[[int, str], LanguageModel]
        if len(signature(model_factory).parameters) == 1:
            self._legacy_model_factory = cast(Callable[[int], LanguageModel], model_factory)
            self._model_factory = self._create_from_legacy_factory
        else:
            self._model_factory = cast(Callable[[int, str], LanguageModel], model_factory)
        self._model: LanguageModel | None = None
        self._context_size: int | None = None
        self._model_profile: str | None = None

    def __call__(self, value: LlmGenerateInput, context: WorkerExecutionContext) -> LlmGenerateResult:
        if value.provider != self._provider:
            raise ValueError(f"LLM handler provider mismatch: expected={self._provider.value} actual={value.provider.value}")
        started_at = monotonic()
        context.raise_if_cancelled()
        model = self._model_for(value.context_size, value.model_profile)
        request_model = value.options.model or model.model_name
        logger.info(
            "LLM Worker 请求开始 provider=%s model=%s context_size=%d stream=%s",
            model.provider.value,
            request_model,
            value.context_size,
            str(value.stream).lower(),
        )
        try:
            messages = [
                ChatMessage(
                    role=message.role,
                    content=message.content,
                    tool_calls=tuple(
                        ToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                            thought_signature=call.thought_signature,
                        )
                        for call in message.tool_calls
                    ),
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
                for message in value.messages
            ]
            options = self._options(value)
            # Plain text uses the interruptible stream path even when callers
            # do not surface deltas. Structured responses must honor
            # stream=False: Gemini 3.1 rejects streamed JSON Schema requests.
            use_stream = (
                not options.tools
                and (value.stream or options.response_format.type == ResponseFormatType.TEXT)
            )
            result = (
                self._stream(model, messages, options, context, emit_deltas=value.stream)
                if use_stream
                else self._result(model.complete(messages, options))
            )
        except Exception:
            logger.info(
                "LLM Worker 请求失败 provider=%s model=%s stream=%s elapsed_ms=%d",
                model.provider.value,
                request_model,
                str(value.stream).lower(),
                round((monotonic() - started_at) * 1000),
                exc_info=True,
            )
            raise
        logger.info(
            "LLM Worker 请求完成 provider=%s model=%s stream=%s elapsed_ms=%d",
            result.provider.value,
            result.model,
            str(value.stream).lower(),
            round((monotonic() - started_at) * 1000),
        )
        return result

    def release(self) -> None:
        model = self._model
        self._model = None
        self._context_size = None
        self._model_profile = None
        if model is not None:
            model.release()

    def _model_for(self, context_size: int, model_profile: str) -> LanguageModel:
        if self._model is not None and (self._provider != LlmProvider.LOCAL or (self._context_size == context_size and self._model_profile == model_profile)):
            return self._model
        self.release()
        self._model = self._model_factory(context_size, model_profile)
        self._context_size = context_size
        self._model_profile = model_profile
        return self._model

    def _create_from_legacy_factory(self, context_size: int, _model_profile: str) -> LanguageModel:
        if self._legacy_model_factory is None:
            raise RuntimeError("Legacy LLM model factory was not configured")
        return self._legacy_model_factory(context_size)

    @staticmethod
    def _options(value: LlmGenerateInput) -> CompletionOptions:
        response_format = value.options.response_format
        return CompletionOptions(
            max_tokens=value.options.max_tokens,
            model=value.options.model,
            min_request_interval_seconds=value.options.min_request_interval_seconds,
            temperature=value.options.temperature,
            response_format=ResponseFormat(
                type=response_format.type,
                json_schema=response_format.json_schema,
                strict=response_format.strict,
            ),
            tools=tuple(
                ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
                for tool in value.options.tools
            ),
            tool_choice=value.options.tool_choice,
        )

    @staticmethod
    def _stream(
        model: LanguageModel,
        messages: Sequence[ChatMessage],
        options: CompletionOptions,
        context: WorkerExecutionContext,
        *,
        emit_deltas: bool,
    ) -> LlmGenerateResult:
        text_parts: list[str] = []
        completion = LlmGenerateResult(text="", provider=model.provider, model=model.model_name)
        events = model.stream(messages, options)
        try:
            for event in events:
                context.raise_if_cancelled()
                if event.text_delta:
                    text_parts.append(event.text_delta)
                    if emit_deltas:
                        context.emit_delta(event.text_delta)
                completion = LlmGenerateResult(
                    text="",
                    provider=event.provider,
                    model=event.model,
                    finish_reason=event.finish_reason or completion.finish_reason,
                    prompt_tokens=event.prompt_tokens if event.prompt_tokens is not None else completion.prompt_tokens,
                    completion_tokens=event.completion_tokens if event.completion_tokens is not None else completion.completion_tokens,
                    request_id=event.request_id or completion.request_id,
                )
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()
        text = "".join(text_parts)
        if not text.strip():
            raise RuntimeError("LLM returned an empty streaming completion")
        return completion.model_copy(update={"text": text})

    @staticmethod
    def _result(completion: LlmCompletion) -> LlmGenerateResult:
        return LlmGenerateResult(
            text=completion.text,
            provider=completion.provider,
            model=completion.model,
            finish_reason=completion.finish_reason,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            request_id=completion.request_id,
            tool_calls=[
                WorkerToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=dict(call.arguments),
                    thought_signature=call.thought_signature,
                )
                for call in completion.tool_calls
            ],
        )


class _BatchItemContext:
    def __init__(self, context: WorkerExecutionContext, item_id: str) -> None:
        self._context = context
        self._item_id = item_id

    @property
    def is_cancel_requested(self) -> bool:
        return self._context.is_cancel_requested

    def raise_if_cancelled(self) -> None:
        self._context.raise_if_cancelled()

    def report_progress(self, progress: float, message: str | None = None) -> None:
        return

    def emit_delta(self, text: str, item_id: str | None = None) -> None:
        self._context.emit_delta(text, item_id or self._item_id)


class LlmBatchWorkerHandler:
    """Execute a request batch while keeping actual model inference at batch size one."""

    def __init__(self, provider: LlmProvider, item_handler: LlmWorkerHandler) -> None:
        self._provider = provider
        self._item_handler = item_handler

    def __call__(self, value: LlmGenerateBatchInput, context: WorkerExecutionContext) -> LlmGenerateBatchResult:
        if value.provider != self._provider:
            raise ValueError(f"LLM batch provider mismatch: expected={self._provider.value} actual={value.provider.value}")
        results: list[LlmGenerateBatchItemResult] = []
        total = len(value.items)
        for index, item in enumerate(value.items):
            context.raise_if_cancelled()
            context.report_progress(index / total, f"LLM batch {index + 1}/{total}")
            result = self._item_handler(item.request, _BatchItemContext(context, item.item_id))
            results.append(LlmGenerateBatchItemResult(item_id=item.item_id, result=result))
        context.report_progress(1, f"LLM batch {total}/{total}")
        return LlmGenerateBatchResult(items=results)

    def release(self) -> None:
        """Release the model shared by the serialized item handler."""
        self._item_handler.release()
