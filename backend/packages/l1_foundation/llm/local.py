from __future__ import annotations

import gc
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

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
from l1_foundation.llm.errors import LlmConfigurationError, LlmResponseError

logger = logging.getLogger("llm")
STOP_TOKENS = ["</s>", "<|im_end|>"]


class LlamaModel(Protocol):
    metadata: Mapping[str, object]

    def __call__(self, prompt: str, **kwargs: object) -> Mapping[str, object] | Iterable[Mapping[str, object]]: ...

    def close(self) -> None: ...

    def token_eos(self) -> int: ...

    def token_bos(self) -> int: ...

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]: ...


class LlamaGrammarFactory(Protocol):
    @classmethod
    def from_json_schema(cls, json_schema: str, verbose: bool = False) -> object: ...


class LlamaFactory(Protocol):
    def __call__(self, *, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool) -> LlamaModel: ...


class LlamaCppModule(Protocol):
    Llama: LlamaFactory
    LlamaGrammar: LlamaGrammarFactory


class LocalLlamaLanguageModel(LanguageModel):
    """Lock-free llama.cpp provider; callers own resource admission and serialization."""

    def __init__(self, model_path: Path, context_size: int, verbose: bool = False) -> None:
        if context_size < 1:
            raise ValueError("context_size must be positive")
        self._model_path = model_path
        self._context_size = context_size
        self._verbose = verbose
        self._model: LlamaModel | None = None
        self._force_cpu = False

    @property
    def provider(self) -> LlmProvider:
        return LlmProvider.LOCAL

    @property
    def model_name(self) -> str:
        return self._model_path.name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, json_object=True, strict_json_schema=True)

    def complete(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> LlmCompletion:
        invoke_options = self._invoke_options(options, stream=False)
        prompt = self._chat_prompt(messages, options)
        try:
            response = self._load_model()(prompt, **invoke_options)
        except Exception:
            if self._force_cpu:
                raise
            logger.info("Local LLM GPU inference failed; retrying on CPU", exc_info=True)
            self._switch_to_cpu()
            response = self._load_model()(prompt, **invoke_options)
        if not isinstance(response, Mapping):
            raise LlmResponseError("Local LLM returned an invalid non-streaming response")
        text, finish_reason = self._completion_text(cast(Mapping[str, object], response))
        if not text.strip():
            raise LlmResponseError("Local LLM returned an empty completion")
        clean_text = text.strip()
        return LlmCompletion(
            text=clean_text,
            provider=self.provider,
            model=self.model_name,
            finish_reason=finish_reason,
            prompt_tokens=self._token_count(prompt, add_bos=True),
            completion_tokens=self._token_count(clean_text, add_bos=False),
        )

    def stream(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> Iterator[LlmStreamEvent]:
        invoke_options = self._invoke_options(options, stream=True)
        prompt = self._chat_prompt(messages, options)
        prompt_tokens = self._token_count(prompt, add_bos=True)
        emitted = False
        try:
            for event in self._stream_response(self._load_model()(prompt, **invoke_options), prompt_tokens):
                emitted = True
                yield event
            return
        except Exception:
            if self._force_cpu or emitted:
                raise
            logger.info("Local LLM GPU streaming failed before the first delta; retrying on CPU", exc_info=True)
            self._switch_to_cpu()
        yield from self._stream_response(self._load_model()(prompt, **invoke_options), prompt_tokens)

    def _stream_response(
        self,
        response: Mapping[str, object] | Iterable[Mapping[str, object]],
        prompt_tokens: int,
    ) -> Iterator[LlmStreamEvent]:
        if isinstance(response, Mapping):
            text, finish_reason = self._completion_text(cast(Mapping[str, object], response))
            if text:
                yield LlmStreamEvent(
                    text,
                    self.provider,
                    self.model_name,
                    finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=self._token_count(text, add_bos=False),
                )
            return
        completion_parts: list[str] = []
        for chunk in response:
            text, finish_reason = self._completion_text(chunk, allow_empty=True)
            completion_parts.append(text)
            if text or finish_reason is not None:
                terminal = finish_reason is not None
                yield LlmStreamEvent(
                    text,
                    self.provider,
                    self.model_name,
                    finish_reason,
                    prompt_tokens=prompt_tokens if terminal else None,
                    completion_tokens=self._token_count("".join(completion_parts), add_bos=False) if terminal else None,
                )

    def _token_count(self, text: str, *, add_bos: bool) -> int:
        return len(self._load_model().tokenize(text.encode("utf-8"), add_bos=add_bos, special=True))

    def _switch_to_cpu(self) -> None:
        model = self._model
        self._model = None
        self._force_cpu = True
        if model is not None:
            logger.info("Local LLM：开始释放 GPU 模型 model=%s", self.model_name)
            try:
                model.close()
            except Exception:
                logger.info("Local LLM GPU model close failed during CPU fallback", exc_info=True)
            finally:
                del model
                gc.collect()
                logger.info("Local LLM：GPU 模型已释放 model=%s", self.model_name)

    def release(self) -> None:
        model = self._model
        self._model = None
        self._force_cpu = False
        if model is None:
            gc.collect()
            return
        logger.info("Local LLM：开始释放模型 model=%s", self.model_name)
        try:
            model.close()
        except Exception as error:
            logger.info("Local LLM reference released, but llama.cpp close failed: %s", error)
        finally:
            del model
            gc.collect()
            logger.info("Local LLM：模型已释放 model=%s", self.model_name)

    def _invoke_options(self, options: CompletionOptions, *, stream: bool) -> dict[str, object]:
        if options.model is not None and options.model != self.model_name:
            raise LlmConfigurationError("local LLM does not support per-request model overrides")
        values: dict[str, object] = {
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "stop": STOP_TOKENS,
            "echo": False,
            "stream": stream,
        }
        response_format = options.response_format
        if response_format.type == ResponseFormatType.JSON_SCHEMA:
            module = self._llama_cpp()
            values["grammar"] = module.LlamaGrammar.from_json_schema(
                json.dumps(response_format.json_schema, ensure_ascii=False),
                verbose=False,
            )
        elif response_format.type == ResponseFormatType.JSON_OBJECT:
            values["grammar"] = self._llama_cpp().LlamaGrammar.from_json_schema(
                '{"type":"object"}',
                verbose=False,
            )
        return values

    def _load_model(self) -> LlamaModel:
        if self._model is not None:
            return self._model
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Local LLM model file not found: {self._model_path}")
        module = self._llama_cpp()
        n_gpu_layers = 0 if self._force_cpu else -1
        device = "cpu" if self._force_cpu else "gpu"
        logger.info(
            "Local LLM：开始加载模型 model=%s device=%s context_size=%d",
            self.model_name,
            device,
            self._context_size,
        )
        try:
            self._model = module.Llama(
                model_path=str(self._model_path),
                n_ctx=self._context_size,
                n_gpu_layers=n_gpu_layers,
                verbose=self._verbose,
            )
        except Exception:
            if self._force_cpu:
                raise
            logger.info("Local LLM：GPU 模型加载失败，改用 CPU model=%s", self.model_name, exc_info=True)
            self._force_cpu = True
            device = "cpu"
            logger.info(
                "Local LLM：开始加载模型 model=%s device=%s context_size=%d",
                self.model_name,
                device,
                self._context_size,
            )
            self._model = module.Llama(
                model_path=str(self._model_path),
                n_ctx=self._context_size,
                n_gpu_layers=0,
                verbose=self._verbose,
            )
        logger.info("Local LLM：模型加载完成 model=%s device=%s", self.model_name, device)
        return self._model

    @staticmethod
    def _llama_cpp() -> LlamaCppModule:
        try:
            return cast(LlamaCppModule, import_module("llama_cpp"))
        except (ImportError, AttributeError) as error:
            raise RuntimeError("llama-cpp-python is required for the local LLM provider") from error

    def _chat_prompt(self, messages: Sequence[ChatMessage], options: CompletionOptions) -> str:
        model = self._load_model()
        metadata_value = cast(object, getattr(model, "metadata", None))
        metadata: Mapping[str, object] = cast(Mapping[str, object], metadata_value) if isinstance(metadata_value, Mapping) else dict[str, object]()
        template: object | None = metadata.get("tokenizer.chat_template")
        enable_thinking = options.enbale_thinking
        if isinstance(template, str) and template:
            try:
                formatter_module = import_module("llama_cpp.llama_chat_format")
                formatter_factory = formatter_module.Jinja2ChatFormatter
                eos_token_id = model.token_eos()
                bos_token_id = model.token_bos()
                concrete_model = cast(Any, model)
                eos_token = concrete_model._model.token_get_text(eos_token_id) if eos_token_id != -1 else "<|im_end|>"
                bos_token = concrete_model._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""
                formatter = formatter_factory(
                    template=template,
                    eos_token=eos_token,
                    bos_token=bos_token,
                    stop_token_ids=[eos_token_id] if eos_token_id != -1 else None,
                )
                payload = [{"role": item.role.value, "content": item.content} for item in messages]
                return str(formatter(messages=payload, enable_thinking=enable_thinking).prompt)
            except Exception:
                logger.info("Local LLM chat template unavailable; falling back to ChatML", exc_info=True)
        return "".join(f"<|im_start|>{message.role.value}\n{message.content.strip()}\n<|im_end|>\n" for message in messages) + "<|im_start|>assistant\n"

    @staticmethod
    def _completion_text(response: Mapping[str, object], allow_empty: bool = False) -> tuple[str, str | None]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            if allow_empty:
                return "", None
            raise LlmResponseError("Local LLM returned no completion choices")
        choice = cast(Mapping[str, object], choices[0])
        finish_reason = choice.get("finish_reason")
        return str(choice.get("text") or ""), str(finish_reason) if finish_reason is not None else None
