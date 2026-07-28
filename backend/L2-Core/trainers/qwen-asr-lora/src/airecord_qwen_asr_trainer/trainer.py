from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from airecord_qwen_asr_trainer.contracts import TrainingManifest
from airecord_qwen_asr_trainer.data import QwenAsrDataCollator


def train(manifest_path: Path) -> dict[str, object]:
    manifest = TrainingManifest.load(manifest_path)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Trainer, TrainingArguments
    except ImportError as error:
        raise RuntimeError("Qwen HF training requires torch, datasets, peft, and transformers>=5.13") from error

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    processor = AutoProcessor.from_pretrained(str(manifest.base_model), local_files_only=True)
    base_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(manifest.base_model),
        dtype=dtype,
        local_files_only=True,
    )
    lora_config = LoraConfig(
        r=manifest.preset.rank,
        lora_alpha=manifest.preset.alpha,
        lora_dropout=manifest.preset.dropout,
        target_modules=list(manifest.preset.target_modules),
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    data_files = {"train": str(manifest.train_file)}
    has_validation = bool(manifest.validation_file is not None and manifest.validation_file.stat().st_size > 0)
    if has_validation and manifest.validation_file is not None:
        data_files["validation"] = str(manifest.validation_file)
    dataset = load_dataset("json", data_files=data_files)

    manifest.output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(manifest.output_dir),
        per_device_train_batch_size=manifest.preset.batch_size,
        per_device_eval_batch_size=manifest.preset.batch_size,
        gradient_accumulation_steps=manifest.preset.gradient_accumulation_steps,
        learning_rate=manifest.preset.learning_rate,
        num_train_epochs=manifest.preset.epochs,
        warmup_ratio=manifest.preset.warmup_ratio,
        logging_steps=5,
        save_strategy="steps",
        save_steps=manifest.preset.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=manifest.preset.save_steps if has_validation else None,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=manifest.preset.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=QwenAsrDataCollator(processor),
        processing_class=processor,
    )
    result = trainer.train()
    model.save_pretrained(str(manifest.output_dir), safe_serialization=True)
    processor.save_pretrained(str(manifest.output_dir))
    output: dict[str, object] = {
        "run_id": manifest.run_id,
        "status": "succeeded",
        "base_model": str(manifest.base_model),
        "adapter_dir": str(manifest.output_dir),
        "preset": asdict(manifest.preset),
        "train_metrics": dict(result.metrics),
    }
    result_path = manifest.output_dir / "training-result.json"
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output
