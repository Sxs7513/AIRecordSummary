from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from time import monotonic, sleep
from typing import cast

import httpx

from l1_foundation.llm.contracts import (
    ChatMessage,
    CompletionOptions,
    LanguageModel,
    LlmCompletion,
    LlmProvider,
    LlmStreamEvent,
    ProviderCapabilities,
    ResponseFormatType,
)
from l1_foundation.llm.errors import LlmConfigurationError, LlmResponseError, UnsupportedResponseFormatError

logger = logging.getLogger("llm")
RATE_LIMIT_STATUS_CODE = 429


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
    ) -> None:
        if not api_key.strip():
            raise LlmConfigurationError(f"{api_key_name} is required when provider={provider.value}")
        if not model.strip():
            raise LlmConfigurationError(f"{provider_label} model is required when provider={provider.value}")
        if rate_limit_max_attempts < 1:
            raise LlmConfigurationError("rate_limit_max_attempts must be positive")
        if rate_limit_retry_seconds < 0:
            raise LlmConfigurationError("rate_limit_retry_seconds must not be negative")
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
        self._client: httpx.Client | None = None

    @property
    def provider(self) -> LlmProvider:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, json_object=True, strict_json_schema=self._strict_json_schema)

    def complete(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> LlmCompletion:
        started_at = monotonic()
        logger.info(
            "Online LLM：开始请求 provider=%s model=%s stream=false",
            self.provider.value,
            self.model_name,
        )
        try:
            response = self._post_chat_completions(self._payload(messages, options, stream=False))
            self._raise_for_status(response)
            data = self._json_mapping(response)
            text, finish_reason = self._message_text(data)
            usage = self._mapping(data.get("usage"))
            completion = LlmCompletion(
                text=text.strip(),
                provider=self.provider,
                model=str(data.get("model") or self.model_name),
                finish_reason=finish_reason,
                prompt_tokens=self._integer(usage.get("prompt_tokens")),
                completion_tokens=self._integer(usage.get("completion_tokens")),
                request_id=self._request_id(data),
            )
        except Exception:
            logger.info(
                "Online LLM：请求失败 provider=%s model=%s stream=false elapsed_ms=%d",
                self.provider.value,
                self.model_name,
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
        response_model = self.model_name
        logger.info(
            "Online LLM：开始请求 provider=%s model=%s stream=true",
            self.provider.value,
            self.model_name,
        )
        try:
            for line in self._stream_chat_completion_lines(self._payload(messages, options, stream=True)):
                if not line.startswith("data:"):
                    continue
                value = line.removeprefix("data:").strip()
                if not value or value == "[DONE]":
                    continue
                try:
                    data = cast(object, json.loads(value))
                except json.JSONDecodeError as error:
                    raise LlmResponseError(f"{self._provider_label} returned an invalid SSE JSON event") from error
                if not isinstance(data, Mapping):
                    raise LlmResponseError(f"{self._provider_label} returned a non-object SSE event")
                event = cast(Mapping[str, object], data)
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
                self.model_name,
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

    def _post_chat_completions(self, payload: Mapping[str, object]) -> httpx.Response:
        for attempt in range(1, self._rate_limit_max_attempts + 1):
            response = self._get_client().post("/chat/completions", json=payload)
            if not self._should_retry_rate_limit(response, attempt):
                return response
            self._wait_before_rate_limit_retry(attempt)
        raise AssertionError("rate-limit retry loop must return a response")

    def _stream_chat_completion_lines(self, payload: Mapping[str, object]) -> Iterator[str]:
        for attempt in range(1, self._rate_limit_max_attempts + 1):
            should_retry = False
            with self._get_client().stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code == RATE_LIMIT_STATUS_CODE:
                    response.read()
                    should_retry = self._should_retry_rate_limit(response, attempt)
                if not should_retry:
                    self._raise_for_status(response)
                    yield from response.iter_lines()
                    return
            self._wait_before_rate_limit_retry(attempt)
        raise AssertionError("rate-limit retry loop must return a response")

    def _should_retry_rate_limit(self, response: httpx.Response, attempt: int) -> bool:
        return response.status_code == RATE_LIMIT_STATUS_CODE and attempt < self._rate_limit_max_attempts

    def _wait_before_rate_limit_retry(self, attempt: int) -> None:
        logger.info(
            "Online LLM：请求被限流，等待重试 provider=%s model=%s attempt=%d/%d retry_in_seconds=%g",
            self.provider.value,
            self.model_name,
            attempt,
            self._rate_limit_max_attempts,
            self._rate_limit_retry_seconds,
        )
        sleep(self._rate_limit_retry_seconds)

    def _payload(self, messages: Sequence[ChatMessage], options: CompletionOptions, *, stream: bool) -> dict[str, object]:
        if options.temperature > self._max_temperature:
            raise LlmConfigurationError(f"{self._provider_label} temperature must be between 0 and {self._max_temperature:g}")
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": item.role.value, "content": item.content} for item in messages],
            "max_tokens": options.max_tokens,
            "stream": stream,
        }
        if self._include_temperature:
            payload["temperature"] = options.temperature
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = self._reasoning_effort
        if stream and self._stream_include_usage:
            payload["stream_options"] = {"include_usage": True}
        response_format = options.response_format
        if response_format.type == ResponseFormatType.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        elif response_format.type == ResponseFormatType.JSON_SCHEMA:
            if self._strict_json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": response_format.strict,
                        "schema": response_format.json_schema,
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
                message_list = cast(list[dict[str, str]], payload["messages"])
                message_list.insert(0, {"role": "system", "content": schema_instruction})
        return payload

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

    def _json_mapping(self, response: httpx.Response) -> Mapping[str, object]:
        try:
            data = cast(object, response.json())
        except ValueError as error:
            raise LlmResponseError(f"{self._provider_label} returned invalid JSON") from error
        if not isinstance(data, Mapping):
            raise LlmResponseError(f"{self._provider_label} returned a non-object response")
        return cast(Mapping[str, object], data)

    def _message_text(self, data: Mapping[str, object]) -> tuple[str, str | None]:
        choice = self._first_choice(data)
        message = self._mapping(choice.get("message"))
        text = message.get("content")
        if not isinstance(text, str) or not text:
            raise LlmResponseError(f"{self._provider_label} returned an empty completion")
        return text, self._optional_string(choice.get("finish_reason"))

    def _delta_text(self, data: Mapping[str, object]) -> tuple[str, str | None]:
        choice = self._first_choice(data)
        delta = self._mapping(choice.get("delta"))
        return str(delta.get("content") or ""), self._optional_string(choice.get("finish_reason"))

    def _first_choice(self, data: Mapping[str, object]) -> Mapping[str, object]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise LlmResponseError(f"{self._provider_label} returned no completion choices")
        return cast(Mapping[str, object], choices[0])

    @staticmethod
    def _mapping(value: object) -> Mapping[str, object]:
        return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _integer(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return str(value) if value is not None else None

    @staticmethod
    def _request_id(data: Mapping[str, object]) -> str | None:
        value = data.get("request_id") or data.get("id")
        return str(value) if value is not None else None

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return round((monotonic() - started_at) * 1000)
