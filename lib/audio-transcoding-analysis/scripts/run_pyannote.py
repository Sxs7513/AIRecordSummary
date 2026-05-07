#!/usr/bin/env python3
import argparse
import contextlib
import gc
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


def local_pyannote_pipeline_config(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    snapshots_dir = Path(cache_dir) / "models--pyannote--speaker-diarization-3.1" / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots:
        config_path = snapshot / "config.yaml"
        if config_path.exists():
            return str(config_path)
    return None


def local_hf_snapshot_dir(cache_dir: str | None, model_id: str, required_file: str) -> str | None:
    if not cache_dir:
        return None
    snapshots_dir = Path(cache_dir) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots:
        if (snapshot / required_file).exists():
            return str(snapshot.resolve())
    return None


def temporary_local_pyannote_config(cache_dir: str | None, config_path: str) -> str | None:
    segmentation_dir = local_hf_snapshot_dir(cache_dir, "pyannote/segmentation-3.0", "pytorch_model.bin")
    embedding_dir = local_hf_snapshot_dir(cache_dir, "pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin")
    community_xvec_dir = local_hf_snapshot_dir(cache_dir, "pyannote/speaker-diarization-community-1", "plda/xvec_transform.npz")
    community_plda_dir = local_hf_snapshot_dir(cache_dir, "pyannote/speaker-diarization-community-1", "plda/plda.npz")
    if not segmentation_dir or not embedding_dir or not community_xvec_dir or not community_plda_dir:
        return None

    config_text = Path(config_path).read_text(encoding="utf-8")
    config_text = config_text.replace("segmentation: pyannote/segmentation-3.0", f"segmentation: {json.dumps(segmentation_dir)}")
    config_text = config_text.replace("embedding: pyannote/wespeaker-voxceleb-resnet34-LM", f"embedding: {json.dumps(embedding_dir)}")
    temp_config = tempfile.NamedTemporaryFile("w", suffix=".pyannote.config.yaml", delete=False, encoding="utf-8")
    try:
        temp_config.write(config_text)
        return temp_config.name
    finally:
        temp_config.close()


def load_pyannote_pipeline(Pipeline, cache_dir: str | None, auth_token: str | None, use_local_config: bool):
    if use_local_config:
        local_config = local_pyannote_pipeline_config(cache_dir)
        if local_config:
            patched_config = temporary_local_pyannote_config(cache_dir, local_config)
            if not patched_config:
                print("pyannote local pipeline config found, but local dependency snapshots are incomplete; falling back to online loading.", file=sys.stderr, flush=True)
            else:
                print(f"pyannote loading patched local pipeline config: {patched_config}", file=sys.stderr, flush=True)
                return Pipeline.from_pretrained(patched_config), patched_config
    else:
        print("pyannote local pipeline config disabled; using standard pipeline loading.", file=sys.stderr, flush=True)

    if not auth_token:
        raise RuntimeError(
            "PYANNOTE_AUTH_TOKEN is required because local pyannote config is disabled or no complete local pyannote pipeline/dependency snapshots were found. "
            "Download the model into model-cache/huggingface/hub first, or set PYANNOTE_AUTH_TOKEN for online loading."
        )

    try:
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=auth_token, cache_dir=cache_dir), None
    except TypeError:
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token, cache_dir=cache_dir), None


def use_pyannote_offline_if_complete(cache_dir: str | None, use_local_config: bool) -> bool:
    if not use_local_config:
        return False
    pipeline_config = local_pyannote_pipeline_config(cache_dir)
    if not pipeline_config:
        return False
    temp_config = temporary_local_pyannote_config(cache_dir, pipeline_config)
    if not temp_config:
        return False
    Path(temp_config).unlink(missing_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return True


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


def release_torch_memory(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


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
    parser.add_argument("--no-local-config", action="store_true")
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
    use_local_config = not args.no_local_config
    if use_pyannote_offline_if_complete(args.cache_dir, use_local_config):
        print("pyannote offline mode enabled before import: complete local snapshots found.", file=sys.stderr, flush=True)

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
        pipeline, temp_pipeline_config = load_pyannote_pipeline(Pipeline, args.cache_dir, args.auth_token, use_local_config)

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

    with contextlib.suppress(Exception):
        del diarization
        del diarization_output
        del audio
        del pipeline
        release_torch_memory(torch)
        if temp_pipeline_config:
            Path(temp_pipeline_config).unlink(missing_ok=True)

    progress("done", "Speaker diarization 完成", 100)
    print(json.dumps({"modelName": "pyannote/speaker-diarization-3.1", "segments": segments}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
