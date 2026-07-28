from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO


class QwenHfRuntime:
    def __init__(self, model_path: Path, adapter_path: Path | None, max_new_tokens: int) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Qwen HF evaluation requires torch, peft, and transformers>=5.13") from error
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16 if torch.backends.mps.is_available() else torch.float32
        device_map = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self._model = Qwen3ASRForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        ).eval()
        if adapter_path is not None:
            self._model = PeftModel.from_pretrained(self._model, str(adapter_path), is_trainable=False).eval()
        self._max_new_tokens = max_new_tokens

    def transcribe(self, audio: str, language: str | None, prompt: str = "") -> str:
        inputs = self._processor.apply_transcription_request(
            audio=audio,
            language=self._language(language),
            prompt=prompt or None,
        ).to(self._model.device, self._model.dtype)
        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        values = self._processor.decode(generated_ids, return_format="transcription_only")
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
            request: Any = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Inference request must be an object")
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
