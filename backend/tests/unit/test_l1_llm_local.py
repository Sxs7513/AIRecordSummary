from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import pytest

import l1_foundation.llm.local as local_module
from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    LocalLlamaLanguageModel,
    ResponseFormat,
    ResponseFormatType,
)
from l1_foundation.llm.local import LlamaCppModule


class FakeLlama:
    metadata: Mapping[str, object] = {}

    def __init__(self) -> None:
        self.options: dict[str, object] = {}
        self.closed = False

    def __call__(self, prompt: str, **kwargs: object) -> Mapping[str, object] | Iterable[Mapping[str, object]]:
        assert "<|im_start|>user" in prompt
        self.options = kwargs
        if kwargs.get("stream") is True:
            return iter(({"choices": [{"text": "hel"}]}, {"choices": [{"text": "lo", "finish_reason": "stop"}]}))
        return {"choices": [{"text": '{"ok":true}', "finish_reason": "stop"}]}

    def close(self) -> None:
        self.closed = True

    def token_eos(self) -> int:
        return -1

    def token_bos(self) -> int:
        return -1

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        count = len(text.decode("utf-8"))
        return list(range(count + (1 if add_bos else 0)))


def test_local_provider_supports_typed_non_stream_and_json_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grammar = object()
    llama = FakeLlama()

    class FakeGrammar:
        @classmethod
        def from_json_schema(cls, json_schema: str, verbose: bool = False) -> object:
            assert json.loads(json_schema)["properties"]["ok"]["type"] == "boolean"
            assert not verbose
            return grammar

    class FakeModule:
        LlamaGrammar = FakeGrammar

        @staticmethod
        def Llama(*, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool) -> FakeLlama:
            assert model_path.endswith("model.gguf")
            assert n_ctx == 8192
            assert n_gpu_layers == -1
            assert not verbose
            return llama

    model_path = tmp_path / "model.gguf"
    model_path.touch()

    def import_fake_module(_name: str) -> LlamaCppModule:
        return cast(LlamaCppModule, FakeModule())

    monkeypatch.setattr(local_module, "import_module", import_fake_module)
    model = LocalLlamaLanguageModel(model_path, 8192)
    output = model.complete(
        [ChatMessage(ChatRole.USER, "return json")],
        CompletionOptions(
            max_tokens=100,
            response_format=ResponseFormat(
                ResponseFormatType.JSON_SCHEMA,
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            ),
        ),
    )

    assert output.text == '{"ok":true}'
    assert output.finish_reason == "stop"
    assert output.prompt_tokens is not None
    assert output.completion_tokens == len(output.text)
    assert llama.options["grammar"] is grammar


def test_local_provider_supports_streaming_and_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llama = FakeLlama()

    class FakeModule:
        class LlamaGrammar:
            @classmethod
            def from_json_schema(cls, json_schema: str, verbose: bool = False) -> object:
                return object()

        @staticmethod
        def Llama(*, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool) -> FakeLlama:
            return llama

    model_path = tmp_path / "model.gguf"
    model_path.touch()

    def import_fake_module(_name: str) -> LlamaCppModule:
        return cast(LlamaCppModule, FakeModule())

    monkeypatch.setattr(local_module, "import_module", import_fake_module)
    caplog.set_level(logging.INFO, logger="llm")
    model = LocalLlamaLanguageModel(model_path, 8192)

    events = list(model.stream([ChatMessage(ChatRole.USER, "hello")], CompletionOptions(max_tokens=10)))
    model.release()

    assert [event.text_delta for event in events] == ["hel", "lo"]
    assert events[-1].finish_reason == "stop"
    assert events[-1].prompt_tokens is not None
    assert events[-1].completion_tokens == len("hello")
    assert llama.closed
    assert "Local LLM：开始加载模型" in caplog.text
    assert "Local LLM：模型加载完成" in caplog.text
    assert "Local LLM：开始释放模型" in caplog.text
    assert "Local LLM：模型已释放" in caplog.text
