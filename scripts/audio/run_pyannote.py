#!/usr/bin/env python3
import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "lib" / "audio-transcoding-analysis" / "scripts" / "run_pyannote.py"),
    run_name="__main__",
)
