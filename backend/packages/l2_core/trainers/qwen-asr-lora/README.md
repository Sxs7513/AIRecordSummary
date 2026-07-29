# Qwen ASR LoRA Trainer

This is an isolated L2 runtime for `Qwen/Qwen3-ASR-1.7B-hf`.

It intentionally owns a separate `.venv` because the production `qwen-asr`
runtime pins Transformers 4.x while the native Hugging Face checkpoint requires
Transformers 5.13 or newer.

The process boundary is file based:

- `train --manifest ...` reads immutable JSONL/audio inputs and writes an adapter plus `training-result.json`;
- `serve ...` provides a JSON-lines stdin/stdout protocol for evaluating the base model or an adapter;
- it never connects to the application database.
