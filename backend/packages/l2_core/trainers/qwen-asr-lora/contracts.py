from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _as_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], value)


def _as_int(value: object, default: int) -> int:
    candidate = default if value is None else value
    if not isinstance(candidate, str | int | float):
        raise ValueError(f"Expected an integer-compatible value, got {type(candidate).__name__}")
    return int(candidate)


def _as_float(value: object, default: float) -> float:
    candidate = default if value is None else value
    if not isinstance(candidate, str | int | float):
        raise ValueError(f"Expected a numeric value, got {type(candidate).__name__}")
    return float(candidate)


def _required_text(data: Mapping[str, object], key: str) -> str:
    if key not in data:
        raise ValueError(f"Training manifest field is required: {key}")
    return str(data[key])


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
    num_workers: int = 0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    @classmethod
    def from_dict(cls, value: object) -> TrainingPreset:
        data: Mapping[str, object] = {} if value is None else _as_mapping(value, name="Training preset")
        target_modules_value = data.get("target_modules", DEFAULT_TARGET_MODULES)
        if isinstance(target_modules_value, str):
            target_modules = tuple(item.strip() for item in target_modules_value.split(",") if item.strip())
        elif isinstance(target_modules_value, list | tuple):
            items = cast(list[object] | tuple[object, ...], target_modules_value)
            target_modules = tuple(str(item).strip() for item in items if str(item).strip())
        else:
            target_modules = DEFAULT_TARGET_MODULES
        preset = cls(
            rank=_as_int(data.get("rank"), 16),
            alpha=_as_int(data.get("alpha"), 32),
            dropout=_as_float(data.get("dropout"), 0.05),
            batch_size=_as_int(data.get("batch_size"), 1),
            gradient_accumulation_steps=_as_int(data.get("gradient_accumulation_steps"), 16),
            learning_rate=_as_float(data.get("learning_rate"), 2e-4),
            epochs=_as_float(data.get("epochs"), 3),
            warmup_ratio=_as_float(data.get("warmup_ratio"), 0.03),
            save_steps=_as_int(data.get("save_steps"), 100),
            num_workers=_as_int(data.get("num_workers"), 0),
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
        if self.num_workers < 0:
            raise ValueError("DataLoader worker count cannot be negative")
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
        raw_value: object = json.loads(path.read_text(encoding="utf-8"))
        raw = _as_mapping(raw_value, name="Training manifest")
        validation_value = raw.get("validation_file")
        manifest = cls(
            run_id=_required_text(raw, "run_id"),
            base_model=Path(_required_text(raw, "base_model")).resolve(),
            train_file=Path(_required_text(raw, "train_file")).resolve(),
            validation_file=Path(str(validation_value)).resolve() if validation_value else None,
            output_dir=Path(_required_text(raw, "output_dir")).resolve(),
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
