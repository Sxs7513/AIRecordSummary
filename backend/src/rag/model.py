from __future__ import annotations

import asyncio
import gc
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from importlib import import_module
from threading import Thread
from typing import Protocol, cast

from langchain_core.messages import BaseMessage

from generation.local_llm_runtime import local_llm_inference_lock
from settings import Settings

logger = logging.getLogger("rag")


class RagLanguageModel(Protocol):
    async def complete(
        self,
        messages: Sequence[BaseMessage],
        max_tokens: int,
        temperature: float = 0.0,
        json_schema: Mapping[str, object] | None = None,
    ) -> str: ...

    def stream(self, messages: Sequence[BaseMessage], max_tokens: int, temperature: float = 0.1) -> AsyncIterator[str]: ...


class LlamaModel(Protocol):
    def __call__(self, prompt: str, **kwargs: object) -> object: ...

    def close(self) -> None: ...


class LlamaGrammarFactory(Protocol):
    @classmethod
    def from_json_schema(cls, json_schema: str, verbose: bool = False) -> object: ...


class LlamaCppModule(Protocol):
    LlamaGrammar: LlamaGrammarFactory


class LocalLlamaChatModel:
    """Small LangChain-message adapter around the project's local llama.cpp runtime."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: LlamaModel | None = None

    def release(self) -> None:
        with local_llm_inference_lock:
            model = self._model
            self._model = None
            if model is None:
                gc.collect()
                return
            try:
                model.close()
            except Exception as error:
                logger.warning("rag local LLM reference released, but llama.cpp close failed: %s", error)
            finally:
                del model
                gc.collect()
        logger.info("rag local LLM released")

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        max_tokens: int,
        temperature: float = 0.0,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        return await asyncio.to_thread(self._complete_sync, messages, max_tokens, temperature, json_schema)

    async def stream(self, messages: Sequence[BaseMessage], max_tokens: int, temperature: float = 0.1) -> AsyncIterator[str]:
        queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def publish(value: str | BaseException | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, value)

        def generate() -> None:
            try:
                with local_llm_inference_lock:
                    response = self._load_model()(
                        self._chatml(messages),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["<|im_end|>", "</s>"],
                        stream=True,
                    )
                    if isinstance(response, Mapping):
                        publish(self._text(cast(Mapping[str, object], response)))
                    else:
                        for item in cast(Sequence[Mapping[str, object]], response):
                            publish(self._text(item))
            except BaseException as error:
                publish(error)
            finally:
                publish(None)

        Thread(target=generate, name="rag-llama-stream", daemon=True).start()
        while True:
            value = await queue.get()
            if value is None:
                return
            if isinstance(value, BaseException):
                raise value
            if value:
                yield value

    def _complete_sync(
        self,
        messages: Sequence[BaseMessage],
        max_tokens: int,
        temperature: float,
        json_schema: Mapping[str, object] | None,
    ) -> str:
        options: dict[str, object] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<|im_end|>", "</s>"],
            "stream": False,
        }
        if json_schema is not None:
            try:
                module = cast(LlamaCppModule, import_module("llama_cpp"))
                options["grammar"] = module.LlamaGrammar.from_json_schema(json.dumps(json_schema, ensure_ascii=False), verbose=False)
            except (ImportError, AttributeError) as error:
                raise RuntimeError("llama-cpp-python does not support JSON Schema constrained generation") from error
        with local_llm_inference_lock:
            response = self._load_model()(self._chatml(messages), **options)
        if not isinstance(response, Mapping):
            raise RuntimeError("Local RAG model returned an invalid completion")
        return self._text(cast(Mapping[str, object], response)).strip()

    def _load_model(self) -> LlamaModel:
        if self._model is not None:
            return self._model
        model_path = self._settings.resolved_local_llm_model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Local RAG model file not found: {model_path}")
        try:
            module = import_module("llama_cpp")
            factory = module.Llama
            self._model = cast(
                LlamaModel,
                factory(
                    model_path=str(model_path),
                    n_ctx=self._settings.rag_context_size,
                    n_gpu_layers=-1,
                    verbose=self._settings.local_llm_verbose,
                ),
            )
        except ImportError as error:
            raise RuntimeError("llama-cpp-python is required for local RAG; use backend/.venv") from error
        return self._model

    @staticmethod
    def _chatml(messages: Sequence[BaseMessage]) -> str:
        parts: list[str] = []
        for message in messages:
            role = "assistant" if message.type == "ai" else "system" if message.type == "system" else "user"
            content = message.content if isinstance(message.content, str) else str(message.content)
            parts.append(f"<|im_start|>{role}\n{content.strip()}\n<|im_end|>\n")
        return "".join(parts) + "<|im_start|>assistant\n"

    @staticmethod
    def _text(response: Mapping[str, object]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            return ""
        return str(cast(Mapping[str, object], choices[0]).get("text") or "")
