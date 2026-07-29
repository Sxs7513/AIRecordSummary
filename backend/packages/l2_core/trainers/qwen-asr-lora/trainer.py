from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from contracts import TrainingManifest
from data import QwenAsrDataCollator
from typing_utils import dynamic_attribute

logger = logging.getLogger("train")
_TRAIN_PROGRESS_PREFIX = "TRAIN_PROGRESS "


class TrainingProgressCallback(TrainerCallback):
    """Emit one machine-readable INFO record for every Trainer logging step."""

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, object] | None = None,
        **_kwargs: Any,
    ) -> Any:
        del args
        step = int(state.global_step)
        max_steps = int(state.max_steps)
        if step <= 0 or max_steps <= 0:
            return control
        payload: dict[str, object] = {
            "step": step,
            "max_steps": max_steps,
            "percent": round(step / max_steps * 100, 2),
            "epoch": float(state.epoch or 0),
        }
        for key in ("loss", "learning_rate", "grad_norm"):
            value = (logs or {}).get(key)
            if isinstance(value, int | float):
                payload[key] = value
        logger.info("%s%s", _TRAIN_PROGRESS_PREFIX, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return control


def train(manifest_path: Path) -> dict[str, object]:
    manifest = TrainingManifest.load(manifest_path)
    logger.info(
        "Qwen ASR LoRA：manifest 加载完成 run_id=%s base_model=%s train_file=%s validation_file=%s output_dir=%s",
        manifest.run_id,
        manifest.base_model,
        manifest.train_file,
        manifest.validation_file,
        manifest.output_dir,
    )
    try:
        import datasets
        import peft
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("Qwen HF training requires torch, datasets, peft, and transformers>=5.13") from error

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dataloader_num_workers = 0 if device_name == "mps" else manifest.preset.num_workers
    logger.info(
        "Qwen ASR LoRA：训练运行时就绪 device=%s dtype=%s dataloader_workers=%d torch=%s transformers=%s peft=%s",
        device_name,
        dtype,
        dataloader_num_workers,
        torch.__version__,
        transformers.__version__,
        peft.__version__,
    )

    processor_class = dynamic_attribute(transformers, "AutoProcessor")
    model_class = dynamic_attribute(transformers, "Qwen3ASRForConditionalGeneration")
    logger.info("Qwen ASR LoRA：开始加载 processor")
    processor: Any = processor_class.from_pretrained(str(manifest.base_model), local_files_only=True)
    logger.info("Qwen ASR LoRA：开始加载基础模型")
    base_model: Any = model_class.from_pretrained(
        str(manifest.base_model),
        dtype=dtype,
        local_files_only=True,
    )
    logger.info("Qwen ASR LoRA：基础模型加载完成")
    lora_config_class = dynamic_attribute(peft, "LoraConfig")
    lora_config: Any = lora_config_class(
        r=manifest.preset.rank,
        lora_alpha=manifest.preset.alpha,
        lora_dropout=manifest.preset.dropout,
        target_modules=list(manifest.preset.target_modules),
        bias="none",
    )
    get_peft_model_runtime = dynamic_attribute(peft, "get_peft_model")
    model: Any = get_peft_model_runtime(base_model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    logger.info(
        "Qwen ASR LoRA：LoRA adapter 已挂载 rank=%d alpha=%d dropout=%s targets=%s",
        manifest.preset.rank,
        manifest.preset.alpha,
        manifest.preset.dropout,
        ",".join(manifest.preset.target_modules),
    )

    data_files = {"train": str(manifest.train_file)}
    has_validation = bool(manifest.validation_file is not None and manifest.validation_file.stat().st_size > 0)
    if has_validation and manifest.validation_file is not None:
        data_files["validation"] = str(manifest.validation_file)
    load_dataset_runtime = dynamic_attribute(datasets, "load_dataset")
    dataset: Any = load_dataset_runtime("json", data_files=data_files)
    logger.info(
        "Qwen ASR LoRA：数据集加载完成 train_samples=%d validation_samples=%d",
        len(dataset["train"]),
        len(dataset["validation"]) if has_validation else 0,
    )

    manifest.output_dir.mkdir(parents=True, exist_ok=True)
    training_arguments_class = dynamic_attribute(transformers, "TrainingArguments")
    arguments: Any = training_arguments_class(
        output_dir=str(manifest.output_dir),
        per_device_train_batch_size=manifest.preset.batch_size,
        per_device_eval_batch_size=manifest.preset.batch_size,
        gradient_accumulation_steps=manifest.preset.gradient_accumulation_steps,
        learning_rate=manifest.preset.learning_rate,
        num_train_epochs=manifest.preset.epochs,
        warmup_ratio=manifest.preset.warmup_ratio,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=manifest.preset.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=manifest.preset.save_steps if has_validation else None,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )
    trainer_class = dynamic_attribute(transformers, "Trainer")
    trainer: Any = trainer_class(
        model=model,
        args=arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=QwenAsrDataCollator(processor),
        processing_class=processor,
        callbacks=[TrainingProgressCallback()],
    )
    logger.info(
        "Qwen ASR LoRA：开始训练 epochs=%s batch_size=%d gradient_accumulation_steps=%d learning_rate=%s validation=%s",
        manifest.preset.epochs,
        manifest.preset.batch_size,
        manifest.preset.gradient_accumulation_steps,
        manifest.preset.learning_rate,
        has_validation,
    )
    result: Any = trainer.train()
    logger.info("Qwen ASR LoRA：训练循环完成 metrics=%s", dict(result.metrics))
    logger.info("Qwen ASR LoRA：开始保存 adapter 和 processor output_dir=%s", manifest.output_dir)
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
    logger.info("Qwen ASR LoRA：训练产物保存完成 result=%s", result_path)
    return output
