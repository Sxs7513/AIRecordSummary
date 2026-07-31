from __future__ import annotations

import json
from typing import cast

import httpx
import pytest

from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    ResponseFormat,
    ResponseFormatType,
    UnsupportedResponseFormatError,
    ZhipuLanguageModel,
)


def test_zhipu_provider_supports_non_streaming_json_object() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "model": "glm-test",
                "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    model = _model(httpx.MockTransport(handler))
    output = model.complete(
        [ChatMessage(ChatRole.USER, "json")],
        CompletionOptions(max_tokens=100, response_format=ResponseFormat(ResponseFormatType.JSON_OBJECT)),
    )

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["stream"] is False
    assert output.text == '{"ok":true}'
    assert output.request_id == "request-1"
    assert output.prompt_tokens == 10


def test_zhipu_provider_supports_sse_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"id":"request-2","model":"glm-test","choices":[{"delta":{"content":"你"},"finish_reason":null}]}\n\n'
            'data: {"id":"request-2","model":"glm-test","choices":[{"delta":{"content":"好"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    model = _model(httpx.MockTransport(handler))

    events = list(model.stream([ChatMessage(ChatRole.USER, "你好")], CompletionOptions(max_tokens=100)))

    assert [event.text_delta for event in events] == ["你", "好"]
    assert events[-1].finish_reason == "stop"


def test_zhipu_rejects_strict_json_schema() -> None:
    model = _model(httpx.MockTransport(lambda _request: httpx.Response(500)))

    with pytest.raises(UnsupportedResponseFormatError):
        model.complete(
            [ChatMessage(ChatRole.USER, "json")],
            CompletionOptions(
                max_tokens=100,
                response_format=ResponseFormat(
                    ResponseFormatType.JSON_SCHEMA,
                    {"type": "object"},
                    strict=True,
                ),
            ),
        )


def _model(transport: httpx.MockTransport) -> ZhipuLanguageModel:
    model = ZhipuLanguageModel("test-key", "glm-test")
    model._client = httpx.Client(  # pyright: ignore[reportPrivateUsage]
        base_url="https://open.bigmodel.cn/api/paas/v4",
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )
    return model

