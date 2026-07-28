from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for an original Qwen3-ASR model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated Linear module suffixes",
    )
    return parser.parse_args()


def patch_outer_forward(model: Any) -> None:
    model_class = model.__class__
    if getattr(model_class, "_asr_lab_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("Qwen3-ASR model does not expose thinker.forward")

    def forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        input_features: Any = None,
        feature_attention_mask: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    model_class.forward = forward
    model_class._asr_lab_forward_patched = True


def prefix_messages(prompt: str, audio: object) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]


def preprocess_factory(processor: Any) -> Any:
    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        prompt = str(example.get("prompt", ""))
        prefix = processor.apply_chat_template([prefix_messages(prompt, None)], add_generation_prompt=True, tokenize=False)[0]
        return {"prompt": prompt, "audio": example["audio"], "target": example["text"], "prefix_text": prefix}

    return preprocess


@dataclass
class QwenAsrDataCollator:
    processor: Any
    sampling_rate: int = 16_000

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import librosa

        audio_paths = [str(item["audio"]) for item in features]
        audios = [librosa.load(path, sr=self.sampling_rate, mono=True)[0] for path in audio_paths]
        prefixes = [str(item["prefix_text"]) for item in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [prefix + str(item["target"]) + eos for prefix, item in zip(prefixes, features, strict=True)]
        full_inputs = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
        prefix_inputs = self.processor(text=prefixes, audio=audios, return_tensors="pt", padding=True, truncation=False)
        labels = full_inputs["input_ids"].clone()
        for index, prefix_length in enumerate(prefix_inputs["attention_mask"].sum(dim=1).tolist()):
            labels[index, :prefix_length] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


class FloatInputTrainerMixin:
    def _prepare_inputs(self, inputs: Any) -> Any:
        import torch

        prepared = super()._prepare_inputs(inputs)  # type: ignore[misc]
        model_dtype = getattr(self.model, "dtype", None)  # type: ignore[attr-defined]
        if model_dtype is not None:
            for key, value in prepared.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    prepared[key] = value.to(dtype=model_dtype)
        return prepared


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from qwen_asr import Qwen3ASRModel
        from transformers import GenerationConfig, Trainer, TrainingArguments
    except ImportError as error:
        raise RuntimeError("Training requires torch, qwen-asr, datasets, transformers, peft, and librosa") from error

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    wrapper = Qwen3ASRModel.from_pretrained(args.model_path, dtype=dtype, device_map=None, local_files_only=True)
    base_model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(base_model)
    base_model.generation_config = GenerationConfig.from_model_config(base_model.config)
    target_modules = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    if not target_modules:
        raise ValueError("At least one LoRA target module is required")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    data_files = {"train": args.train_file}
    if args.eval_file and Path(args.eval_file).stat().st_size > 0:
        data_files["validation"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(preprocess_factory(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in dataset:
        removable = [column for column in dataset[split].column_names if column not in keep]
        if removable:
            dataset[split] = dataset[split].remove_columns(removable)

    class FloatInputTrainer(FloatInputTrainerMixin, Trainer):
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=args.save_steps if "validation" in dataset else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = FloatInputTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=QwenAsrDataCollator(processor),
        processing_class=processor,
    )
    trainer.train()
    model.save_pretrained(str(output_dir), safe_serialization=True)
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
