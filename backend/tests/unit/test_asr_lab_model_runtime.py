from pathlib import Path
from unittest.mock import patch

import pytest

from l2_core.asr_lab.model_runtime import build_asr_runtime


@pytest.mark.parametrize(
    ("provider", "runtime_class_name", "python_bin"),
    (
        ("qwen_asr", "QwenAsrRuntime", None),
        ("qwen_hf", "QwenHfSubprocessRuntime", Path("/runtime/python")),
    ),
)
def test_build_asr_runtime_passes_the_same_context_to_all_qwen_providers(
    tmp_path: Path,
    provider: str,
    runtime_class_name: str,
    python_bin: Path | None,
) -> None:
    model = {
        "base_model_name": "Qwen/Qwen3-ASR-1.7B",
        "runtime_config": {"provider": provider},
    }
    with patch(f"l2_core.asr_lab.model_runtime.{runtime_class_name}") as runtime_class:
        runtime = build_asr_runtime(
            model,
            storage_root=tmp_path,
            model_cache_root=tmp_path,
            hf_runtime_python=python_bin,
            context="半导体热词上下文",
        )

    assert runtime is runtime_class.return_value
    assert runtime_class.call_args.kwargs["context"] == "半导体热词上下文"
