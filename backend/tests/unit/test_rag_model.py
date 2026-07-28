from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

import pytest
from langchain_core.messages import HumanMessage

import l2_core.rag.model as rag_model
from l1_foundation.settings import Settings
from l2_core.rag.model import LlamaModel, LocalLlamaChatModel


class FakeLlama:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def __call__(self, prompt: str, **kwargs: object) -> object:
        self.options = kwargs
        return {"choices": [{"text": '{"status":"unresolved"}'}]}

    def close(self) -> None:
        return


def test_complete_constrains_generation_with_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    schema: Mapping[str, object] = {"type": "object", "properties": {"status": {"type": "string"}}}
    grammar = object()

    class FakeGrammar:
        @classmethod
        def from_json_schema(cls, json_schema: str, verbose: bool = False) -> object:
            assert '"status"' in json_schema
            assert not verbose
            return grammar

    class FakeLlamaCpp:
        LlamaGrammar = FakeGrammar

    llama = FakeLlama()
    model = LocalLlamaChatModel(cast(Settings, object()))
    monkeypatch.setattr(rag_model, "import_module", lambda _name: FakeLlamaCpp())
    monkeypatch.setattr(model, "_load_model", lambda: cast(LlamaModel, llama))

    output = asyncio.run(model.complete([HumanMessage("route")], max_tokens=100, json_schema=schema))

    assert output == '{"status":"unresolved"}'
    assert llama.options["grammar"] is grammar
    assert llama.options["stream"] is False
