from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO, cast

from typing_utils import dynamic_attribute

logger = logging.getLogger("evaluation")


class QwenHfRuntime:
    def __init__(self, model_path: Path, adapter_path: Path | None, max_new_tokens: int) -> None:
        try:
            import peft
            import torch
            import transformers
        except ImportError as error:
            raise RuntimeError("Qwen HF evaluation requires torch, peft, and transformers>=5.13") from error
        use_cuda = bool(torch.cuda.is_available())
        use_mps = bool(not use_cuda and torch.backends.mps.is_available())
        dtype = torch.bfloat16 if use_cuda else torch.float16 if use_mps else torch.float32
        runtime_device = "cuda:0" if use_cuda else "mps" if use_mps else "cpu"
        logger.info(
            "Qwen ASR LoRA：准备评测运行时 device=%s dtype=%s load_strategy=%s",
            runtime_device,
            dtype,
            "cpu_then_mps" if use_mps else "direct",
        )
        self._torch: Any = torch
        processor_class = dynamic_attribute(transformers, "AutoProcessor")
        model_class = dynamic_attribute(transformers, "Qwen3ASRForConditionalGeneration")
        self._processor: Any = processor_class.from_pretrained(str(model_path), local_files_only=True)
        load_options: dict[str, object] = {
            "dtype": dtype,
            "local_files_only": True,
        }
        # Loading directly with device_map="mps" lets the HF loader copy many
        # tensors to Metal concurrently. PyTorch 2.13 can corrupt its MPS
        # shader registry in that path and abort the interpreter (SIGABRT).
        # Load on CPU first, then move the complete model to MPS sequentially.
        if use_cuda:
            load_options["device_map"] = "cuda:0"
        self._model = model_class.from_pretrained(str(model_path), **load_options)
        if adapter_path is not None:
            peft_model_class = dynamic_attribute(peft, "PeftModel")
            load_adapter = dynamic_attribute(peft_model_class, "from_pretrained")
            self._model = load_adapter(self._model, str(adapter_path), is_trainable=False)
        if use_mps:
            logger.info("Qwen ASR LoRA：基础模型 CPU 加载完成，开始单线程迁移到 MPS")
            self._model = self._model.to("mps")
        self._model = self._model.eval()
        logger.info("Qwen ASR LoRA：评测运行时加载完成 device=%s", self._model.device)
        self._max_new_tokens = max_new_tokens

    def transcribe(self, audio: str, language: str | None, prompt: str = "") -> str:
        inputs: Any = self._processor.apply_transcription_request(
            audio=audio,
            language=self._language(language),
            prompt=prompt or None,
        ).to(self._model.device, self._model.dtype)
        with self._torch.inference_mode():
            output_ids: Any = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        generated_ids: Any = output_ids[:, inputs["input_ids"].shape[1] :]
        values: Any = self._processor.decode(generated_ids, return_format="transcription_only")
        return str(values[0]).strip() if values else ""

    @staticmethod
    def _language(value: str | None) -> str | None:
        normalized = (value or "").strip().lower()
        if not normalized or normalized in {"auto", "none"}:
            return None
        return {"zh": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese", "en": "English", "en-us": "English"}.get(normalized, value)


def serve(
    *,
    model_path: Path,
    adapter_path: Path | None,
    max_new_tokens: int,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    runtime = QwenHfRuntime(model_path, adapter_path, max_new_tokens)
    _write(output_stream, {"event": "ready"})
    for line in input_stream:
        try:
            request_value: object = json.loads(line)
            if not isinstance(request_value, dict):
                raise ValueError("Inference request must be an object")
            request = cast(dict[str, object], request_value)
            if request.get("command") == "shutdown":
                _write(output_stream, {"event": "stopped"})
                return
            text = runtime.transcribe(
                str(request["audio"]),
                str(request["language"]) if request.get("language") is not None else None,
                str(request.get("prompt", "")),
            )
            _write(output_stream, {"id": request.get("id"), "text": text})
        except Exception as error:
            _write(output_stream, {"error": str(error)})


def _write(stream: TextIO, value: object) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()
