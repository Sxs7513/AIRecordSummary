from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class LlmProvider(StrEnum):
    LOCAL = "local"
    ZHIPU = "zhipu"
    GEMINI = "gemini"


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


class ResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    type: ResponseFormatType = ResponseFormatType.TEXT
    json_schema: Mapping[str, object] | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        if self.type == ResponseFormatType.JSON_SCHEMA and self.json_schema is None:
            raise ValueError("JSON_SCHEMA response format requires json_schema")
        if self.type != ResponseFormatType.JSON_SCHEMA and self.json_schema is not None:
            raise ValueError("json_schema is only valid for JSON_SCHEMA response format")


@dataclass(frozen=True, slots=True)
class CompletionOptions:
    max_tokens: int
    temperature: float = 0.0
    response_format: ResponseFormat = ResponseFormat()

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")


@dataclass(frozen=True, slots=True)
class LlmCompletion:
    text: str
    provider: LlmProvider
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class LlmStreamEvent:
    text_delta: str
    provider: LlmProvider
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    streaming: bool
    json_object: bool
    strict_json_schema: bool


class LanguageModel(Protocol):
    @property
    def provider(self) -> LlmProvider: ...

    @property
    def model_name(self) -> str: ...

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def complete(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> LlmCompletion: ...

    def stream(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> Iterator[LlmStreamEvent]: ...

    def release(self) -> None: ...
