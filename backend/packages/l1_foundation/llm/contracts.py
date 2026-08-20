from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, cast

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


def as_json_value(value: object) -> JsonValue:
    """Validate and copy an untrusted Python value into the JSON value domain."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): as_json_value(item) for key, item in cast(Mapping[object, object], value).items()}
    if isinstance(value, list | tuple):
        return [as_json_value(item) for item in cast(list[object] | tuple[object, ...], value)]
    raise TypeError(f"Value is not JSON-compatible: {type(value).__name__}")


def as_json_object(value: object) -> JsonObject:
    normalized = as_json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("JSON value is not an object")
    return normalized


class LlmProvider(StrEnum):
    LOCAL = "local"
    ZHIPU = "zhipu"
    GEMINI = "gemini"


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JsonObject


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonObject
    # Gemini 3 attaches this opaque value to a function call.  It must be
    # replayed verbatim with the corresponding call before sending its result.
    thought_signature: str | None = None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    thought_signature: str | None = None


class ResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    type: ResponseFormatType = ResponseFormatType.TEXT
    json_schema: JsonObject | None = None
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
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: Literal["auto", "required", "none"] = "auto"
    model: str | None = None
    min_request_interval_seconds: float | None = None
    enbale_thinking: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.model is not None and not self.model.strip():
            raise ValueError("model override must not be blank")
        if self.min_request_interval_seconds is not None and self.min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds must not be negative")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("tool_choice=required requires at least one tool")


@dataclass(frozen=True, slots=True)
class LlmCompletion:
    text: str
    provider: LlmProvider
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


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
    tool_calling: bool = False


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
