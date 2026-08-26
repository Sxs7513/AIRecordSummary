from __future__ import annotations

import json
from typing import cast

import httpx

from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    LlmProvider,
    QwenLanguageModel,
    ResponseFormat,
    ResponseFormatType,
    create_language_model,
)


def test_qwen_provider_uses_dashscope_chat_completions_contract() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/compatible-mode/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "qwen-request-1",
                "model": "qwen3.8-flash",
                "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
        )

    model = _model(httpx.MockTransport(handler))
    output = model.complete(
        [ChatMessage(ChatRole.USER, "Return JSON")],
        CompletionOptions(
            max_tokens=100,
            response_format=ResponseFormat(
                ResponseFormatType.JSON_SCHEMA,
                {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                strict=True,
            ),
        ),
    )

    assert captured["model"] == "qwen3.8-flash"
    assert captured["max_completion_tokens"] == 100
    assert "max_tokens" not in captured
    assert captured["enable_thinking"] is False
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        },
    }
    assert output.provider == LlmProvider.QWEN
    assert output.text == '{"ok":true}'
    assert output.request_id == "qwen-request-1"


def test_qwen_provider_supports_streaming_usage_and_ignores_reasoning_content() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        body = (
            'data: {"id":"qwen-request-2","model":"qwen3.8-flash","choices":[{"delta":{"reasoning_content":"hidden"},"finish_reason":null}]}\n\n'
            'data: {"id":"qwen-request-2","model":"qwen3.8-flash","choices":[{"delta":{"content":"你"},"finish_reason":null}]}\n\n'
            'data: {"id":"qwen-request-2","model":"qwen3.8-flash","choices":[{"delta":{"content":"好"},"finish_reason":"stop"}]}\n\n'
            'data: {"id":"qwen-request-2","model":"qwen3.8-flash","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    events = list(_model(httpx.MockTransport(handler)).stream([ChatMessage(ChatRole.USER, "你好")], CompletionOptions(max_tokens=20)))

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["max_completion_tokens"] == 20
    assert captured["enable_thinking"] is False
    assert [event.text_delta for event in events] == ["你", "好", ""]
    assert events[-1].prompt_tokens == 5
    assert events[-1].completion_tokens == 2


def test_factory_creates_qwen_provider() -> None:
    model = create_language_model(
        LlmProvider.QWEN,
        qwen_ai_platform_api_key="test-key",
        qwen_llm_model="qwen3.8-flash",
    )

    assert isinstance(model, QwenLanguageModel)
    assert model.provider == LlmProvider.QWEN
    assert model.model_name == "qwen3.8-flash"


def _model(transport: httpx.MockTransport) -> QwenLanguageModel:
    model = QwenLanguageModel("test-key", "qwen3.8-flash")
    model._client = httpx.Client(  # pyright: ignore[reportPrivateUsage]
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )
    return model
