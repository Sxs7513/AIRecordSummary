from __future__ import annotations

import json
from pathlib import Path

import pytest

from airecord_qwen_asr_trainer.contracts import TrainingManifest, TrainingPreset


def test_training_preset_rejects_empty_targets() -> None:
    with pytest.raises(ValueError, match="target"):
        TrainingPreset.from_dict({"target_modules": []})


def test_manifest_loads_paths_and_preset(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    train = tmp_path / "train.jsonl"
    train.write_text('{"audio":"a.wav","text":"language Chinese<asr_text>测试"}\n', encoding="utf-8")
    output = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "base_model": str(model),
                "train_file": str(train),
                "output_dir": str(output),
                "preset": {"rank": 8, "target_modules": ["q_proj"]},
            }
        ),
        encoding="utf-8",
    )

    manifest = TrainingManifest.load(manifest_path)

    assert manifest.run_id == "run-1"
    assert manifest.preset.rank == 8
    assert manifest.validation_file is None
