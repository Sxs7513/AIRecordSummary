#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import sys
import subprocess
import tempfile
import warnings
import wave
from pathlib import Path


def progress(stage: str, message: str, percent: int):
    print(
        "PROGRESS_JSON:" + json.dumps({"stage": stage, "message": message, "percent": percent}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )


def label_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return f"Speaker {alphabet[index]}"
    return f"Speaker {index + 1}"


def annotation_from_output(diarization_output):
    if hasattr(diarization_output, "exclusive_speaker_diarization"):
        return diarization_output.exclusive_speaker_diarization
    if hasattr(diarization_output, "speaker_diarization"):
        return diarization_output.speaker_diarization
    if hasattr(diarization_output, "itertracks"):
        return diarization_output

    output_type = type(diarization_output).__name__
    fields = ", ".join(sorted(name for name in dir(diarization_output) if not name.startswith("_"))[:20])
    raise RuntimeError(f"Unsupported pyannote diarization output {output_type}. Available fields: {fields}")


class PyannoteProgressHook:
    step_ranges = {
        "segmentation": (45, 68),
        "speaker_counting": (68, 74),
        "embeddings": (74, 90),
        "discrete_diarization": (90, 94),
    }

    step_messages = {
        "segmentation": "pyannote 正在做语音分割",
        "speaker_counting": "pyannote 正在估计说话人数",
        "embeddings": "pyannote 正在提取说话人特征",
        "discrete_diarization": "pyannote 正在生成说话人时间线",
    }

    def __init__(self):
        self.last_percent = None

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        start, end = self.step_ranges.get(step_name, (55, 92))
        if total and completed is not None:
            ratio = max(0.0, min(1.0, completed / total))
            percent = int(start + ratio * (end - start))
            suffix = f" {completed}/{total}"
        else:
            percent = end
            suffix = ""

        if percent == self.last_percent:
            return
        self.last_percent = percent
        message = self.step_messages.get(step_name, f"pyannote 正在处理 {step_name}")
        progress(step_name, f"{message}{suffix}", percent)


def resolve_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_audio_to_device(audio, device):
    if device.type == "cpu":
        return audio
    return {**audio, "waveform": audio["waveform"].to(device)}


def load_audio_for_pyannote(audio_path: str):
    import numpy as np
    import torch

    progress("convert_audio", "转换音频为 pyannote waveform", 30)
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_file.close()
    wav_path = wav_file.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        with wave.open(wav_path, "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

        if sample_width != 2:
            raise RuntimeError(f"Expected 16-bit PCM WAV from ffmpeg, got sample width {sample_width}")

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        waveform = torch.from_numpy(audio).unsqueeze(0)
        progress("audio_ready", "音频 waveform 准备完成", 45)
        return {"waveform": waveform, "sample_rate": sample_rate}
    finally:
        Path(wav_path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--auth-token", default=os.environ.get("PYANNOTE_AUTH_TOKEN"))
    parser.add_argument("--cache-dir", default=os.environ.get("HUGGINGFACE_HUB_CACHE"))
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r"\s*torchcodec is not installed correctly.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"std\(\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )

    if not args.auth_token:
        raise RuntimeError(
            "PYANNOTE_AUTH_TOKEN is required for pyannote/speaker-diarization-3.1. "
            "Set PYANNOTE_AUTH_TOKEN in .env with a HuggingFace token that has access to the pyannote model."
        )

    with contextlib.redirect_stdout(sys.stderr):
        progress("import", "加载 pyannote 依赖", 5)
        try:
            import torch
            from pyannote.audio import Pipeline
        except Exception as exc:
            raise RuntimeError("pyannote.audio is not installed. Run scripts/install_audio_dependencies.sh first.") from exc

        if args.cache_dir:
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        progress("load_pipeline", "加载 pyannote diarization pipeline", 15)
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.auth_token, cache_dir=args.cache_dir)
        except TypeError:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=args.auth_token, cache_dir=args.cache_dir)

        device = resolve_device(torch)
        pipeline.to(device)
        print(f"pyannote device: {device}", file=sys.stderr, flush=True)
        audio = load_audio_for_pyannote(args.audio_path)
        audio = move_audio_to_device(audio, device)
        progress("diarize", "pyannote 开始分离说话人", 45)
        diarization_output = pipeline(audio, hook=PyannoteProgressHook())
        progress("parse", "整理说话人分离结果", 92)
        diarization = annotation_from_output(diarization_output)

    speaker_order = {}
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_order:
            speaker_order[speaker] = len(speaker_order)
        label = label_for_index(speaker_order[speaker])
        segments.append(
            {
                "speakerClusterId": str(speaker),
                "speakerLabel": label,
                "startMs": int(float(turn.start) * 1000),
                "endMs": int(float(turn.end) * 1000),
                "confidence": None,
            }
        )

    progress("done", "Speaker diarization 完成", 100)
    print(json.dumps({"modelName": "pyannote/speaker-diarization-3.1", "segments": segments}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
