from __future__ import annotations

import json
import logging
from typing import cast

import httpx
import pytest

import l1_foundation.llm.openai_compatible as openai_compatible_module
from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    GeminiLanguageModel,
    LlmProvider,
    LlmResponseError,
    ResponseFormat,
    ResponseFormatType,
    create_language_model,
)


def test_gemini_provider_supports_non_streaming_strict_json_schema(caplog: pytest.LogCaptureFixture) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "gemini-request-1",
                "model": "gemini-test",
                "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
        )

    caplog.set_level(logging.INFO, logger="llm")
    model = _model(httpx.MockTransport(handler))
    output = model.complete(
        [ChatMessage(ChatRole.USER, "return json")],
        CompletionOptions(
            max_tokens=100,
            response_format=ResponseFormat(
                ResponseFormatType.JSON_SCHEMA,
                {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                strict=True,
            ),
        ),
    )

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        },
    }
    assert captured["reasoning_effort"] == "minimal"
    assert "temperature" not in captured
    assert output.provider == LlmProvider.GEMINI
    assert output.text == '{"ok":true}'
    assert output.request_id == "gemini-request-1"
    assert "Online LLM：开始请求 provider=gemini model=gemini-test stream=false" in caplog.text
    assert "Online LLM：请求完成 provider=gemini model=gemini-test stream=false" in caplog.text


def test_gemini_provider_supports_sse_streaming(caplog: pytest.LogCaptureFixture) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        body = (
            'data: {"id":"gemini-request-2","model":"gemini-test","choices":[{"delta":{"content":"流"},"finish_reason":null}]}\n\n'
            'data: {"id":"gemini-request-2","model":"gemini-test","choices":[{"delta":{"content":"式"},"finish_reason":"stop"}]}\n\n'
            'data: {"id":"gemini-request-2","model":"gemini-test","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    caplog.set_level(logging.INFO, logger="llm")
    model = _model(httpx.MockTransport(handler))

    events = list(model.stream([ChatMessage(ChatRole.USER, "stream")], CompletionOptions(max_tokens=100)))

    assert captured["stream_options"] == {"include_usage": True}
    assert [event.text_delta for event in events if event.text_delta] == ["流", "式"]
    assert all(event.provider == LlmProvider.GEMINI for event in events)
    assert next(event.finish_reason for event in events if event.finish_reason is not None) == "stop"
    assert events[-1].prompt_tokens == 12
    assert events[-1].completion_tokens == 2
    assert "Online LLM：开始请求 provider=gemini model=gemini-test stream=true" in caplog.text
    assert "Online LLM：请求完成 provider=gemini model=gemini-test stream=true" in caplog.text


def test_gemini_retries_rate_limit_twice_before_non_streaming_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(
            200,
            json={
                "id": "gemini-retry-success",
                "model": "gemini-test",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    def record_wait(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(openai_compatible_module, "sleep", record_wait)
    caplog.set_level(logging.INFO, logger="llm")
    model = _model(httpx.MockTransport(handler))

    output = model.complete([ChatMessage(ChatRole.USER, "retry")], CompletionOptions(max_tokens=10))

    assert output.text == "ok"
    assert attempts == 3
    assert waits == [10, 10]
    assert caplog.text.count("Online LLM：请求被限流，等待重试") == 2
    assert "attempt=1/3 retry_in_seconds=10" in caplog.text
    assert "attempt=2/3 retry_in_seconds=10" in caplog.text


def test_gemini_retries_rate_limit_for_streaming(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(
            200,
            text='data: {"id":"stream-retry","model":"gemini-test","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    def record_wait(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(openai_compatible_module, "sleep", record_wait)
    caplog.set_level(logging.INFO, logger="llm")
    model = _model(httpx.MockTransport(handler))

    events = list(model.stream([ChatMessage(ChatRole.USER, "retry stream")], CompletionOptions(max_tokens=10)))

    assert [event.text_delta for event in events] == ["ok"]
    assert attempts == 2
    assert waits == [10]
    assert "Online LLM：请求被限流，等待重试" in caplog.text


def test_gemini_raises_after_three_rate_limited_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    def record_wait(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(openai_compatible_module, "sleep", record_wait)
    model = _model(httpx.MockTransport(handler))

    with pytest.raises(LlmResponseError, match="HTTP 429"):
        model.complete([ChatMessage(ChatRole.USER, "exhaust")], CompletionOptions(max_tokens=10))

    assert attempts == 3
    assert waits == [10, 10]


def test_gemini_streaming_http_error_reads_response_body() -> None:
    model = _model(
        httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={"error": {"message": "invalid streaming request"}},
            )
        )
    )

    with pytest.raises(LlmResponseError, match="HTTP 400.*invalid streaming request"):
        list(model.stream([ChatMessage(ChatRole.USER, "invalid")], CompletionOptions(max_tokens=10)))


def test_factory_selects_gemini_provider() -> None:
    model = create_language_model(
        LlmProvider.GEMINI,
        gemini_api_key="test-key",
        gemini_model="gemini-test",
    )

    assert isinstance(model, GeminiLanguageModel)
    assert model.provider == LlmProvider.GEMINI
    model.release()


def _model(transport: httpx.MockTransport) -> GeminiLanguageModel:
    model = GeminiLanguageModel("test-key", "gemini-test")
    model._client = httpx.Client(  # pyright: ignore[reportPrivateUsage]
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )
    return model
