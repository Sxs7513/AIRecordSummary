from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
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
    ToolCall,
    ToolDefinition,
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
                {
                    "$defs": {
                        "Result": {
                            "type": "object",
                            "properties": {
                                "ok": {"type": "boolean", "default": False},
                                "label": {"type": "string", "minLength": 1, "maxLength": 20},
                                "reference_index": {
                                    "anyOf": [
                                        {"type": "integer", "minimum": 1},
                                        {"type": "null"},
                                    ],
                                    "default": None,
                                    "description": "Optional reference index",
                                },
                            },
                            "required": ["ok"],
                        }
                    },
                    "$ref": "#/$defs/Result",
                },
                strict=True,
            ),
        ),
    )

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "label": {"type": "string"},
                    "reference_index": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Optional reference index",
                    },
                },
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }
    assert captured["reasoning_effort"] == "minimal"
    assert "temperature" not in captured
    assert output.provider == LlmProvider.GEMINI
    assert output.text == '{"ok":true}'
    assert output.request_id == "gemini-request-1"
    assert "Online LLM：开始请求 provider=gemini model=gemini-test stream=false" in caplog.text
    assert "Online LLM：请求完成 provider=gemini model=gemini-test stream=false" in caplog.text


@pytest.mark.parametrize(
    ("requested_model", "expected_reasoning_effort"),
    [
        ("gemini-3.6-flash", "minimal"),
        ("gemini-3.7-flash", "low"),
        ("gemini-3.7-flash-001", "low"),
    ],
)
def test_gemini_provider_supports_per_request_model_override(
    monkeypatch: pytest.MonkeyPatch,
    requested_model: str,
    expected_reasoning_effort: str,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(openai_compatible_module, "sleep", no_sleep)
    output = _model(httpx.MockTransport(handler)).complete(
        [ChatMessage(ChatRole.USER, "audit")],
        CompletionOptions(max_tokens=20, model=requested_model),
    )

    assert captured["model"] == requested_model
    assert captured["reasoning_effort"] == expected_reasoning_effort
    assert output.model == requested_model


def test_gemini_37_default_model_uses_low_reasoning_effort() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    model = _model(httpx.MockTransport(handler), model_name="gemini-3.7-flash")
    model.complete([ChatMessage(ChatRole.USER, "audit")], CompletionOptions(max_tokens=20))

    assert captured["reasoning_effort"] == "low"


def test_gemini_request_interval_uses_provider_default_and_per_request_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    waits: list[float] = []

    def fake_monotonic() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        waits.append(seconds)
        now += seconds

    def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    monkeypatch.setattr(openai_compatible_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(openai_compatible_module, "sleep", fake_sleep)
    model = _model(httpx.MockTransport(handler), min_request_interval_seconds=5)
    messages = [ChatMessage(ChatRole.USER, "audit")]

    model.complete(messages, CompletionOptions(max_tokens=20))
    model.complete(messages, CompletionOptions(max_tokens=20))
    audit_options = CompletionOptions(
        max_tokens=20,
        model="gemini-3.6-flash",
        min_request_interval_seconds=15,
    )
    model.complete(messages, audit_options)
    model.complete(messages, audit_options)

    assert waits == [5, 15]


def test_gemini_concurrent_requests_reserve_distinct_start_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    waits_lock = Lock()

    monkeypatch.setattr(openai_compatible_module, "monotonic", lambda: 100.0)

    def record_wait(seconds: float) -> None:
        with waits_lock:
            waits.append(seconds)

    monkeypatch.setattr(openai_compatible_module, "sleep", record_wait)
    model = _model(httpx.MockTransport(lambda _request: httpx.Response(500)))

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(model._wait_for_request_slot, "gemini-test", 5) for _ in range(3)]  # pyright: ignore[reportPrivateUsage]
        for future in futures:
            future.result()

    assert sorted(waits) == [5, 10]


def test_gemini_request_interval_survives_model_release(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    waits: list[float] = []

    def fake_monotonic() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        waits.append(seconds)
        now += seconds

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gemini-test",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    monkeypatch.setattr(openai_compatible_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(openai_compatible_module, "sleep", fake_sleep)
    limiter = openai_compatible_module.SynchronousRequestRateLimiter()
    transport = httpx.MockTransport(handler)
    options = CompletionOptions(max_tokens=20)
    messages = [ChatMessage(ChatRole.USER, "request")]
    first = _model(transport, min_request_interval_seconds=5, request_rate_limiter=limiter)

    first.complete(messages, options)
    first.release()
    second = _model(transport, min_request_interval_seconds=5, request_rate_limiter=limiter)
    second.complete(messages, options)

    assert waits == [5]


def test_gemini_nullable_enum_includes_null_in_translated_schema() -> None:
    model = object.__new__(GeminiLanguageModel)

    schema = model._json_schema_for_provider(  # pyright: ignore[reportPrivateUsage]
        {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "anyOf": [
                        {"type": "string", "enum": ["fact_lookup", "metadata_lookup"]},
                        {"type": "null"},
                    ]
                }
            },
        }
    )

    assert schema["properties"]["strategy_id"] == {
        "type": ["string", "null"],
        "enum": ["fact_lookup", "metadata_lookup", None],
    }


def test_gemini_provider_uses_json_object_for_non_strict_json_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "gemini-non-strict-schema",
                "model": "gemini-test",
                "choices": [{"message": {"content": '{"items":[]}'}, "finish_reason": "stop"}],
            },
        )

    model = _model(httpx.MockTransport(handler))
    output = model.complete(
        [ChatMessage(ChatRole.USER, "返回审计结果")],
        CompletionOptions(
            max_tokens=100,
            response_format=ResponseFormat(
                ResponseFormatType.JSON_SCHEMA,
                {
                    "$defs": {"Item": {"type": "string"}},
                    "type": "object",
                    "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
                },
                strict=False,
            ),
        ),
    )

    assert output.text == '{"items":[]}'
    assert captured["response_format"] == {"type": "json_object"}
    messages = cast(list[dict[str, object]], captured["messages"])
    assert messages[0]["role"] == "system"
    assert "仅输出符合以下 JSON Schema" in cast(str, messages[0]["content"])
    assert messages[1] == {"role": "user", "content": "返回审计结果"}


def test_gemini_provider_sends_tools_and_parses_native_function_calls() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "gemini-tool-request",
                "model": "gemini-test",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "extra_content": {"google": {"thought_signature": "signature-from-gemini"}},
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"I2C specification"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    model = _model(httpx.MockTransport(handler))
    output = model.complete(
        [
            ChatMessage(ChatRole.USER, "核验候选"),
            ChatMessage(
                ChatRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="previous-call",
                        name="web_search",
                        arguments={"query": "I2C"},
                        thought_signature="signature-to-replay",
                    ),
                ),
            ),
            ChatMessage(
                ChatRole.TOOL,
                '{"summary":"official result"}',
                tool_call_id="previous-call",
                name="web_search",
            ),
        ],
        CompletionOptions(
            max_tokens=100,
            tools=(
                ToolDefinition(
                    name="web_search",
                    description="搜索公开资料",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
            tool_choice="required",
        ),
    )

    assert captured["tool_choice"] == "required"
    captured_messages = cast(list[dict[str, object]], captured["messages"])
    assert captured_messages[1]["tool_calls"] == [
        {
            "id": "previous-call",
            "type": "function",
            "extra_content": {"google": {"thought_signature": "signature-to-replay"}},
            "function": {"name": "web_search", "arguments": '{"query":"I2C"}'},
        }
    ]
    assert captured_messages[2] == {
        "role": "tool",
        "content": '{"summary":"official result"}',
        "tool_call_id": "previous-call",
        "name": "web_search",
    }
    assert cast(list[dict[str, object]], captured["tools"])[0]["function"] == {
        "name": "web_search",
        "description": "搜索公开资料",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    assert output.text == ""
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].id == "call-1"
    assert output.tool_calls[0].name == "web_search"
    assert output.tool_calls[0].arguments == {"query": "I2C specification"}
    assert output.tool_calls[0].thought_signature == "signature-from-gemini"


def test_gemini_provider_accepts_message_level_thought_signature_for_a_tool_call() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gemini-tool-request",
                "model": "gemini-test",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "extra_content": {"google": {"thought_signature": "message-signature"}},
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "web_search", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    output = _model(httpx.MockTransport(handler)).complete(
        [ChatMessage(ChatRole.USER, "搜索")],
        CompletionOptions(max_tokens=100, tools=(ToolDefinition("web_search", "搜索", {"type": "object"}),)),
    )

    assert output.tool_calls[0].thought_signature == "message-signature"


def test_gemini_request_log_truncates_nested_prompt_values(caplog: pytest.LogCaptureFixture) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "gemini-log-request",
                "model": "gemini-test",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    long_text = "证据" * 40
    caplog.set_level(logging.INFO, logger="llm")
    output = _model(httpx.MockTransport(handler)).complete(
        [ChatMessage(ChatRole.USER, long_text)],
        CompletionOptions(max_tokens=10),
    )

    assert output.text == "ok"
    assert captured["messages"] == [{"role": "user", "content": long_text}]
    assert "Online LLM：请求体摘要" in caplog.text
    assert long_text not in caplog.text
    assert "<truncated,len=80>" in caplog.text


def test_gemini_response_log_truncates_every_string_field_at_200(caplog: pytest.LogCaptureFixture) -> None:
    response_text = "R" * 200 + "response-tail"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gemini-response-log",
                "model": "gemini-test",
                "choices": [{"message": {"content": response_text}, "finish_reason": "stop"}],
            },
        )

    caplog.set_level(logging.INFO, logger="llm")
    output = _model(httpx.MockTransport(handler)).complete(
        [ChatMessage(ChatRole.USER, "log response")],
        CompletionOptions(max_tokens=10),
    )

    assert output.text == response_text
    response_log = next(message for message in caplog.messages if "Online LLM：响应体摘要" in message)
    assert "R" * 200 in response_log
    assert "response-tail" not in response_log
    assert "<truncated,len=213>" in response_log


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


def test_gemini_complete_and_stream_share_request_interval_state(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    waits: list[float] = []

    def fake_monotonic() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        waits.append(seconds)
        now += seconds

    def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], json.loads(request.content))
        if payload["stream"] is True:
            body = (
                'data: {"model":"gemini-test","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "model": "gemini-test",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    monkeypatch.setattr(openai_compatible_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(openai_compatible_module, "sleep", fake_sleep)
    model = _model(httpx.MockTransport(handler), min_request_interval_seconds=5)
    messages = [ChatMessage(ChatRole.USER, "request")]

    model.complete(messages, CompletionOptions(max_tokens=20))
    events = list(model.stream(messages, CompletionOptions(max_tokens=20)))

    assert waits == [5]
    assert [event.text_delta for event in events] == ["ok"]


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


def _model(
    transport: httpx.MockTransport,
    *,
    model_name: str = "gemini-test",
    min_request_interval_seconds: float = 0,
    request_rate_limiter: openai_compatible_module.SynchronousRequestRateLimiter | None = None,
) -> GeminiLanguageModel:
    model = GeminiLanguageModel(
        "test-key",
        model_name,
        min_request_interval_seconds=min_request_interval_seconds,
        request_rate_limiter=request_rate_limiter or openai_compatible_module.SynchronousRequestRateLimiter(),
    )
    model._client = httpx.Client(  # pyright: ignore[reportPrivateUsage]
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )
    return model
