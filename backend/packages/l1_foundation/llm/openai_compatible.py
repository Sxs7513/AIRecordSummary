from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from threading import Lock
from time import monotonic, sleep
from typing import cast

import httpx

from l1_foundation.llm.contracts import (
    ChatMessage,
    CompletionOptions,
    JsonObject,
    JsonValue,
    LanguageModel,
    LlmCompletion,
    LlmProvider,
    LlmStreamEvent,
    ProviderCapabilities,
    ResponseFormatType,
    ToolCall,
    as_json_object,
)
from l1_foundation.llm.errors import LlmConfigurationError, LlmResponseError, UnsupportedResponseFormatError

logger = logging.getLogger("llm")
RATE_LIMIT_STATUS_CODE = 429
REQUEST_LOG_STRING_LIMIT = 50
RESPONSE_LOG_STRING_LIMIT = 200


class SynchronousRequestRateLimiter:
    """Reserve request start slots across model instances in one worker process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._next_request_at: dict[tuple[LlmProvider, str], float] = {}

    def wait(self, provider: LlmProvider, model: str, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            return
        key = (provider, model)
        with self._lock:
            now = monotonic()
            request_at = max(now, self._next_request_at.get(key, now))
            self._next_request_at[key] = request_at + interval_seconds
        delay_seconds = request_at - now
        if delay_seconds <= 0:
            return
        logger.info(
            "Online LLM：等待请求时间槽 provider=%s model=%s delay_seconds=%g interval_seconds=%g",
            provider.value,
            model,
            delay_seconds,
            interval_seconds,
        )
        sleep(delay_seconds)


_SHARED_REQUEST_RATE_LIMITER = SynchronousRequestRateLimiter()


class OpenAiCompatibleLanguageModel(LanguageModel):
    """Typed HTTP adapter shared by OpenAI-compatible chat-completions providers."""

    def __init__(
        self,
        *,
        provider: LlmProvider,
        provider_label: str,
        api_key_name: str,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float,
        strict_json_schema: bool,
        max_temperature: float,
        include_temperature: bool = True,
        reasoning_effort: str | None = None,
        stream_include_usage: bool = False,
        rate_limit_max_attempts: int = 1,
        rate_limit_retry_seconds: float = 0,
        min_request_interval_seconds: float = 0,
        request_rate_limiter: SynchronousRequestRateLimiter | None = None,
    ) -> None:
        if not api_key.strip():
            raise LlmConfigurationError(f"{api_key_name} is required when provider={provider.value}")
        if not model.strip():
            raise LlmConfigurationError(f"{provider_label} model is required when provider={provider.value}")
        if rate_limit_max_attempts < 1:
            raise LlmConfigurationError("rate_limit_max_attempts must be positive")
        if rate_limit_retry_seconds < 0:
            raise LlmConfigurationError("rate_limit_retry_seconds must not be negative")
        if min_request_interval_seconds < 0:
            raise LlmConfigurationError("min_request_interval_seconds must not be negative")
        self._provider = provider
        self._provider_label = provider_label
        self._model = model.strip()
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._strict_json_schema = strict_json_schema
        self._max_temperature = max_temperature
        self._include_temperature = include_temperature
        self._reasoning_effort = reasoning_effort
        self._stream_include_usage = stream_include_usage
        self._rate_limit_max_attempts = rate_limit_max_attempts
        self._rate_limit_retry_seconds = rate_limit_retry_seconds
        self._min_request_interval_seconds = min_request_interval_seconds
        self._request_rate_limiter = request_rate_limiter or _SHARED_REQUEST_RATE_LIMITER
        self._client: httpx.Client | None = None

    @property
    def provider(self) -> LlmProvider:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, json_object=True, strict_json_schema=self._strict_json_schema, tool_calling=True)

    def complete(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> LlmCompletion:
        started_at = monotonic()
        request_model = options.model or self.model_name
        logger.info(
            "Online LLM：开始请求 provider=%s model=%s stream=false",
            self.provider.value,
            request_model,
        )
        try:
            response = self._post_chat_completions(
                self._payload(messages, options, stream=False),
                self._request_interval_seconds(options),
            )
            self._raise_for_status(response)
            data = self._json_mapping(response)
            self._log_response_payload(data, stream=False)
            text, tool_calls, finish_reason = self._message_content(data)
            usage = self._mapping(data.get("usage"))
            completion = LlmCompletion(
                text=text.strip(),
                provider=self.provider,
                model=str(data.get("model") or request_model),
                finish_reason=finish_reason,
                prompt_tokens=self._integer(usage.get("prompt_tokens")),
                completion_tokens=self._integer(usage.get("completion_tokens")),
                request_id=self._request_id(data),
                tool_calls=tool_calls,
            )
        except Exception:
            logger.info(
                "Online LLM：请求失败 provider=%s model=%s stream=false elapsed_ms=%d",
                self.provider.value,
                request_model,
                self._elapsed_ms(started_at),
                exc_info=True,
            )
            raise
        logger.info(
            "Online LLM：请求完成 provider=%s model=%s stream=false elapsed_ms=%d prompt_tokens=%s completion_tokens=%s request_id=%s",
            self.provider.value,
            completion.model,
            self._elapsed_ms(started_at),
            completion.prompt_tokens,
            completion.completion_tokens,
            completion.request_id,
        )
        return completion

    def stream(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> Iterator[LlmStreamEvent]:
        started_at = monotonic()
        event_count = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        request_id: str | None = None
        response_model = options.model or self.model_name
        logger.info(
            "Online LLM：开始请求 provider=%s model=%s stream=true",
            self.provider.value,
            response_model,
        )
        try:
            for line in self._stream_chat_completion_lines(
                self._payload(messages, options, stream=True),
                self._request_interval_seconds(options),
            ):
                if not line.startswith("data:"):
                    continue
                value = line.removeprefix("data:").strip()
                if not value or value == "[DONE]":
                    continue
                try:
                    data = cast(object, json.loads(value))
                except json.JSONDecodeError as error:
                    raise LlmResponseError(f"{self._provider_label} returned an invalid SSE JSON event") from error
                try:
                    event = as_json_object(data)
                except TypeError as error:
                    raise LlmResponseError(f"{self._provider_label} returned an invalid SSE JSON object") from error
                self._log_response_payload(event, stream=True)
                usage = self._mapping(event.get("usage"))
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    delta, finish_reason = self._delta_text(event)
                elif usage:
                    # OpenAI-compatible providers may send usage in a final
                    # chunk with no choices. Preserve it as a metadata-only event.
                    delta, finish_reason = "", None
                else:
                    raise LlmResponseError(f"{self._provider_label} returned a stream event with neither choices nor usage")
                response_model = str(event.get("model") or response_model)
                current_prompt_tokens = self._integer(usage.get("prompt_tokens"))
                current_completion_tokens = self._integer(usage.get("completion_tokens"))
                if current_prompt_tokens is not None:
                    prompt_tokens = current_prompt_tokens
                if current_completion_tokens is not None:
                    completion_tokens = current_completion_tokens
                request_id = self._request_id(event) or request_id
                if delta or finish_reason is not None or current_prompt_tokens is not None or current_completion_tokens is not None:
                    event_count += 1
                    yield LlmStreamEvent(
                        text_delta=delta,
                        provider=self.provider,
                        model=response_model,
                        finish_reason=finish_reason,
                        prompt_tokens=self._integer(usage.get("prompt_tokens")),
                        completion_tokens=self._integer(usage.get("completion_tokens")),
                        request_id=self._request_id(event),
                    )
        except Exception:
            logger.info(
                "Online LLM：请求失败 provider=%s model=%s stream=true elapsed_ms=%d events=%d",
                self.provider.value,
                response_model,
                self._elapsed_ms(started_at),
                event_count,
                exc_info=True,
            )
            raise
        logger.info(
            "Online LLM：请求完成 provider=%s model=%s stream=true elapsed_ms=%d events=%d prompt_tokens=%s completion_tokens=%s request_id=%s",
            self.provider.value,
            response_model,
            self._elapsed_ms(started_at),
            event_count,
            prompt_tokens,
            completion_tokens,
            request_id,
        )

    def release(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            client.close()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=min(self._timeout_seconds, 30.0),
                    read=self._timeout_seconds,
                    write=min(self._timeout_seconds, 60.0),
                    pool=min(self._timeout_seconds, 30.0),
                ),
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            )
        return self._client

    def _post_chat_completions(
        self,
        payload: JsonObject,
        min_request_interval_seconds: float,
    ) -> httpx.Response:
        self._log_request_payload(payload, stream=False)
        request_model = str(payload.get("model") or self.model_name)
        for attempt in range(1, self._rate_limit_max_attempts + 1):
            self._wait_for_request_slot(request_model, min_request_interval_seconds)
            response = self._get_client().post("/chat/completions", json=payload)
            if not self._should_retry_rate_limit(response, attempt):
                return response
            self._wait_before_rate_limit_retry(attempt, request_model)
        raise AssertionError("rate-limit retry loop must return a response")

    def _stream_chat_completion_lines(
        self,
        payload: JsonObject,
        min_request_interval_seconds: float,
    ) -> Iterator[str]:
        self._log_request_payload(payload, stream=True)
        request_model = str(payload.get("model") or self.model_name)
        for attempt in range(1, self._rate_limit_max_attempts + 1):
            self._wait_for_request_slot(request_model, min_request_interval_seconds)
            should_retry = False
            with self._get_client().stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code == RATE_LIMIT_STATUS_CODE:
                    response.read()
                    should_retry = self._should_retry_rate_limit(response, attempt)
                if not should_retry:
                    self._raise_for_status(response)
                    yield from response.iter_lines()
                    return
            self._wait_before_rate_limit_retry(attempt, request_model)
        raise AssertionError("rate-limit retry loop must return a response")

    def _should_retry_rate_limit(self, response: httpx.Response, attempt: int) -> bool:
        return response.status_code == RATE_LIMIT_STATUS_CODE and attempt < self._rate_limit_max_attempts

    def _wait_before_rate_limit_retry(self, attempt: int, request_model: str) -> None:
        logger.info(
            "Online LLM：请求被限流，等待重试 provider=%s model=%s attempt=%d/%d retry_in_seconds=%g",
            self.provider.value,
            request_model,
            attempt,
            self._rate_limit_max_attempts,
            self._rate_limit_retry_seconds,
        )
        sleep(self._rate_limit_retry_seconds)

    def _request_interval_seconds(self, options: CompletionOptions) -> float:
        if options.min_request_interval_seconds is not None:
            return options.min_request_interval_seconds
        return self._min_request_interval_seconds

    def _wait_for_request_slot(self, request_model: str, interval_seconds: float) -> None:
        self._request_rate_limiter.wait(self.provider, request_model, interval_seconds)

    def _payload(self, messages: Sequence[ChatMessage], options: CompletionOptions, *, stream: bool) -> JsonObject:
        if options.temperature > self._max_temperature:
            raise LlmConfigurationError(f"{self._provider_label} temperature must be between 0 and {self._max_temperature:g}")
        payload: JsonObject = {
            "model": options.model or self._model,
            "messages": [self._message_payload(item) for item in messages],
            "max_tokens": options.max_tokens,
            "stream": stream,
        }
        if self._include_temperature:
            payload["temperature"] = options.temperature
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = self._reasoning_effort
        if stream and self._stream_include_usage:
            payload["stream_options"] = {"include_usage": True}
        if options.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in options.tools
            ]
            payload["tool_choice"] = options.tool_choice
        response_format = options.response_format
        if response_format.type == ResponseFormatType.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        elif response_format.type == ResponseFormatType.JSON_SCHEMA:
            if self._strict_json_schema and response_format.strict:
                schema = self._json_schema_for_provider(response_format.json_schema or {})
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": response_format.strict,
                        "schema": schema,
                    },
                }
            else:
                if response_format.strict:
                    raise UnsupportedResponseFormatError(f"{self._provider_label} does not advertise strict JSON Schema constrained generation")
                payload["response_format"] = {"type": "json_object"}
                schema_instruction = (
                    "仅输出符合以下 JSON Schema 的 JSON 对象，不要输出解释或 Markdown：\n"
                    f"{json.dumps(response_format.json_schema, ensure_ascii=False, separators=(',', ':'))}"
                )
                message_list = cast(list[JsonObject], payload["messages"])
                message_list.insert(0, {"role": "system", "content": schema_instruction})
        return payload

    def _json_schema_for_provider(self, schema: JsonObject) -> JsonObject:
        """Translate standard JSON Schema into the provider dialect when needed."""

        return schema

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            try:
                response.read()
                body = response.text[-2000:]
            except httpx.HTTPError as body_error:
                body = f"<response body unavailable: {type(body_error).__name__}>"
            raise LlmResponseError(f"{self._provider_label} API returned HTTP {response.status_code}: {body}") from error

    def _log_request_payload(self, payload: JsonObject, *, stream: bool) -> None:
        """Log request shape for provider debugging without exposing full prompt data."""

        logger.info(
            "Online LLM：请求体摘要 provider=%s model=%s stream=%s payload=%s",
            self.provider.value,
            str(payload.get("model") or self.model_name),
            str(stream).lower(),
            json.dumps(
                self._truncated_for_log(payload, string_limit=REQUEST_LOG_STRING_LIMIT),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def _log_response_payload(self, payload: JsonObject, *, stream: bool) -> None:
        """Log provider response fields while bounding every string value."""

        logger.info(
            "Online LLM：响应体摘要 provider=%s model=%s stream=%s payload=%s",
            self.provider.value,
            str(payload.get("model") or self.model_name),
            str(stream).lower(),
            json.dumps(
                self._truncated_for_log(payload, string_limit=RESPONSE_LOG_STRING_LIMIT),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    @classmethod
    def _truncated_for_log(cls, value: JsonValue, *, string_limit: int) -> JsonValue:
        if isinstance(value, dict):
            return {
                str(key): cls._truncated_for_log(item, string_limit=string_limit)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._truncated_for_log(item, string_limit=string_limit) for item in value]
        if isinstance(value, str) and len(value) > string_limit:
            return f"{value[:string_limit]}…<truncated,len={len(value)}>"
        return value

    def _json_mapping(self, response: httpx.Response) -> JsonObject:
        try:
            data = cast(object, response.json())
        except ValueError as error:
            raise LlmResponseError(f"{self._provider_label} returned invalid JSON") from error
        try:
            return as_json_object(data)
        except TypeError as error:
            raise LlmResponseError(f"{self._provider_label} returned a non-object response") from error

    def _message_content(self, data: JsonObject) -> tuple[str, tuple[ToolCall, ...], str | None]:
        choice = self._first_choice(data)
        message = self._mapping(choice.get("message"))
        text = message.get("content")
        content = text if isinstance(text, str) else ""
        tool_calls = self._parse_tool_calls(
            message.get("tool_calls"),
            fallback_thought_signature=self._thought_signature(message),
        )
        if not content and not tool_calls:
            raise LlmResponseError(f"{self._provider_label} returned an empty completion")
        return content, tool_calls, self._optional_string(choice.get("finish_reason"))

    @classmethod
    def _message_payload(cls, message: ChatMessage) -> JsonObject:
        payload: JsonObject = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                cls._tool_call_payload(call)
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            payload["name"] = message.name
        return payload

    def _parse_tool_calls(
        self,
        value: object,
        *,
        fallback_thought_signature: str | None = None,
    ) -> tuple[ToolCall, ...]:
        if not isinstance(value, list):
            return ()
        result: list[ToolCall] = []
        for item in cast(list[object], value):
            try:
                call = as_json_object(item)
            except TypeError as error:
                raise LlmResponseError(f"{self._provider_label} returned an invalid tool call") from error
            function = self._mapping(call.get("function"))
            call_id = call.get("id")
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(raw_arguments, str):
                raise LlmResponseError(f"{self._provider_label} returned an incomplete tool call")
            try:
                arguments = cast(object, json.loads(raw_arguments))
            except json.JSONDecodeError as error:
                raise LlmResponseError(f"{self._provider_label} returned invalid tool call arguments") from error
            try:
                argument_object = as_json_object(arguments)
            except TypeError as error:
                raise LlmResponseError(f"{self._provider_label} returned non-object tool call arguments") from error
            # Gemini's OpenAI-compatible responses normally put the signature
            # on the first tool call. Accept the message-level form too: it
            # has appeared in compatibility responses and the signature still
            # belongs to the first functionCall when replayed.
            signature = self._thought_signature(call)
            if not result and signature is None:
                signature = fallback_thought_signature
            result.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=argument_object,
                    thought_signature=signature,
                )
            )
        return tuple(result)

    @staticmethod
    def _tool_call_payload(call: ToolCall) -> JsonObject:
        payload: JsonObject = {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }
        if call.thought_signature is not None:
            payload["extra_content"] = {"google": {"thought_signature": call.thought_signature}}
        return payload

    @classmethod
    def _thought_signature(cls, call: JsonObject) -> str | None:
        extra_content = cls._mapping(call.get("extra_content"))
        google = cls._mapping(extra_content.get("google"))
        signature = google.get("thought_signature")
        return signature if isinstance(signature, str) else None

    def _delta_text(self, data: JsonObject) -> tuple[str, str | None]:
        choice = self._first_choice(data)
        delta = self._mapping(choice.get("delta"))
        return str(delta.get("content") or ""), self._optional_string(choice.get("finish_reason"))

    def _first_choice(self, data: JsonObject) -> JsonObject:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmResponseError(f"{self._provider_label} returned no completion choices")
        try:
            return as_json_object(choices[0])
        except TypeError as error:
            raise LlmResponseError(f"{self._provider_label} returned an invalid completion choice") from error

    @staticmethod
    def _mapping(value: JsonValue | object) -> JsonObject:
        try:
            return as_json_object(value)
        except TypeError:
            return {}

    @staticmethod
    def _integer(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return str(value) if value is not None else None

    @staticmethod
    def _request_id(data: JsonObject) -> str | None:
        value = data.get("request_id") or data.get("id")
        return str(value) if value is not None else None

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return round((monotonic() - started_at) * 1000)
