from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True, slots=True)
class TrainingPreset:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    epochs: float = 3.0
    warmup_ratio: float = 0.03
    save_steps: int = 100
    num_workers: int = 2
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    @classmethod
    def from_dict(cls, value: object) -> TrainingPreset:
        data = value if isinstance(value, dict) else {}
        target_modules = data.get("target_modules", DEFAULT_TARGET_MODULES)
        if isinstance(target_modules, str):
            target_modules = tuple(item.strip() for item in target_modules.split(",") if item.strip())
        elif isinstance(target_modules, list):
            target_modules = tuple(str(item).strip() for item in target_modules if str(item).strip())
        else:
            target_modules = DEFAULT_TARGET_MODULES
        preset = cls(
            rank=int(data.get("rank", 16)),
            alpha=int(data.get("alpha", 32)),
            dropout=float(data.get("dropout", 0.05)),
            batch_size=int(data.get("batch_size", 1)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 16)),
            learning_rate=float(data.get("learning_rate", 2e-4)),
            epochs=float(data.get("epochs", 3)),
            warmup_ratio=float(data.get("warmup_ratio", 0.03)),
            save_steps=int(data.get("save_steps", 100)),
            num_workers=int(data.get("num_workers", 2)),
            target_modules=target_modules,
        )
        preset.validate()
        return preset

    def validate(self) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("Batch size and gradient accumulation must be positive")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("Learning rate and epochs must be positive")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required")


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    run_id: str
    base_model: Path
    train_file: Path
    validation_file: Path | None
    output_dir: Path
    preset: TrainingPreset

    @classmethod
    def load(cls, path: Path) -> TrainingManifest:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Training manifest must contain a JSON object")
        validation_value = raw.get("validation_file")
        manifest = cls(
            run_id=str(raw["run_id"]),
            base_model=Path(str(raw["base_model"])).resolve(),
            train_file=Path(str(raw["train_file"])).resolve(),
            validation_file=Path(str(validation_value)).resolve() if validation_value else None,
            output_dir=Path(str(raw["output_dir"])).resolve(),
            preset=TrainingPreset.from_dict(raw.get("preset")),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("Training manifest run_id is required")
        if not self.base_model.is_dir():
            raise FileNotFoundError(f"Base model snapshot does not exist: {self.base_model}")
        if not self.train_file.is_file() or self.train_file.stat().st_size == 0:
            raise FileNotFoundError(f"Training JSONL is missing or empty: {self.train_file}")
        if self.validation_file is not None and not self.validation_file.is_file():
            raise FileNotFoundError(f"Validation JSONL does not exist: {self.validation_file}")
