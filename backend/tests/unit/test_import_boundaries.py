from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_generation_contract_can_be_imported_in_a_fresh_interpreter() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": "packages"}

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from l2_core.generation.contracts import CreateGenerationCommand; "
            "print(CreateGenerationCommand.__name__)",
        ],
        cwd=backend_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "CreateGenerationCommand"
