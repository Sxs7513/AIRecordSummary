#!/usr/bin/env python3
import argparse
import contextlib
import gc
import json
import os
import signal
import sys
import tempfile
import time
import warnings
import wave
from pathlib import Path


def progress(stage: str, message: str, percent: int):
    print(
        "PROGRESS_JSON:" + json.dumps({"stage": stage, "message": message, "percent": percent}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )


def log_event(event: str, **payload):
    print(
        "[qwen-asr] " + json.dumps({"event": event, **payload}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )


def current_rss_mb():
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:
        return None


@contextlib.contextmanager
def segment_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    def raise_timeout(signum, frame):
        raise TimeoutError(f"segment transcription timed out after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(normalize_text(item) for item in value).strip()
    return str(value).strip()


def strip_trailing_punctuation(text: str) -> str:
    return text.rstrip().rstrip("。！？!?；;，,、.").strip()


def label_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return f"Speaker {alphabet[index]}"
    return f"Speaker {index + 1}"


def get_attr(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def result_items(result):
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return list(result)
    if result is None:
        return []
    return [result]


def seconds_to_ms(value) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number > 10000:
        return int(number)
    return int(number * 1000)


def language_arg(value: str):
    normalized = (value or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    language_map = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "chinese": "Chinese",
        "cn": "Chinese",
        "en": "English",
        "en-us": "English",
        "english": "English",
    }
    return language_map.get(normalized, value)


def extract_text(result) -> str:
    for item in result_items(result):
        text = get_attr(item, "text")
        if text:
            return normalize_text(text)
    return ""


def extract_language(result):
    for item in result_items(result):
        language = get_attr(item, "language")
        if language:
            return normalize_text(language)
    return None


def normalize_timestamp_items(time_stamps):
    if not time_stamps:
        return []
    if isinstance(time_stamps, list) and len(time_stamps) == 1 and isinstance(time_stamps[0], list):
        time_stamps = time_stamps[0]
    return time_stamps if isinstance(time_stamps, list) else []


def segment_from_timestamp_item(item, strip_punctuation: bool):
    text = normalize_text(get_attr(item, "text"))
    if strip_punctuation:
        text = strip_trailing_punctuation(text)
    start = get_attr(item, "start_time", get_attr(item, "start", None))
    end = get_attr(item, "end_time", get_attr(item, "end", None))
    if not text or start is None or end is None:
        return None
    start_ms = seconds_to_ms(start)
    end_ms = seconds_to_ms(end)
    if end_ms <= start_ms:
        return None
    return {"startMs": start_ms, "endMs": end_ms, "text": text}


def merge_timestamp_segments(raw_segments, max_duration_ms: int = 12000, break_on_sentence_end: bool = False):
    if not raw_segments:
        return []
    merged = []
    sentence_endings = ("。", "！", "？", ".", "!", "?")
    for segment in raw_segments:
        if not merged:
            merged.append(segment.copy())
            continue
        current = merged[-1]
        current_duration = segment["endMs"] - current["startMs"]
        should_break = (break_on_sentence_end and current["text"].endswith(sentence_endings)) or current_duration > max_duration_ms
        if should_break:
            merged.append(segment.copy())
        else:
            current["endMs"] = segment["endMs"]
            current["text"] = f'{current["text"]}{segment["text"]}'.strip()
    return merged


def extract_timestamp_segments(result, strip_punctuation: bool, break_on_sentence_end: bool):
    raw_segments = []
    for item in result_items(result):
        time_stamps = normalize_timestamp_items(get_attr(item, "time_stamps", get_attr(item, "timestamps", None)))
        for stamp in time_stamps:
            segment = segment_from_timestamp_item(stamp, strip_punctuation)
            if segment:
                raw_segments.append(segment)
    return merge_timestamp_segments(sorted(raw_segments, key=lambda segment: segment["startMs"]), break_on_sentence_end=break_on_sentence_end)


def extract_vad_segments(result) -> list[list[int]]:
    segments = []
    for item in result_items(result):
        value = get_attr(item, "value", get_attr(item, "timestamp", item))
        if not isinstance(value, list):
            continue
        for segment in value:
            if isinstance(segment, list) and len(segment) >= 2:
                start_ms = int(segment[0])
                end_ms = int(segment[1])
                if start_ms >= 0 and end_ms > start_ms:
                    segments.append([start_ms, end_ms])
    return sorted(segments, key=lambda segment: segment[0])


def merge_vad_segments(segments: list[list[int]], max_merged_ms: int, max_gap_ms: int, min_segment_ms: int) -> list[list[int]]:
    if not segments or max_merged_ms <= 0:
        return segments
    merged = [segments[0]]
    for start_ms, end_ms in segments[1:]:
        current = merged[-1]
        gap_ms = start_ms - current[1]
        current_duration = current[1] - current[0]
        next_duration = end_ms - start_ms
        should_merge_gap = max_gap_ms < 0 or gap_ms <= max_gap_ms
        should_merge_short = min_segment_ms > 0 and (current_duration < min_segment_ms or next_duration < min_segment_ms)
        if end_ms - current[0] <= max_merged_ms and (should_merge_gap or should_merge_short):
            current[1] = end_ms
        else:
            merged.append([start_ms, end_ms])
    return merged


def split_long_segments(segments: list[list[int]], max_segment_ms: int) -> list[list[int]]:
    if max_segment_ms <= 0:
        return segments
    output = []
    for start_ms, end_ms in segments:
        cursor = start_ms
        while cursor < end_ms:
            next_end_ms = min(end_ms, cursor + max_segment_ms)
            output.append([cursor, next_end_ms])
            cursor = next_end_ms
    return output


def annotation_from_pyannote_output(diarization_output):
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
                log_event("pyannote_local_incomplete", configPath=local_config)
            else:
                log_event("pyannote_local_pipeline", configPath=patched_config)
                return Pipeline.from_pretrained(patched_config), patched_config
    else:
        log_event("pyannote_local_config_disabled")

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


def resolve_torch_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_audio_to_device(audio, device):
    if device.type == "cpu":
        return audio
    return {**audio, "waveform": audio["waveform"].to(device)}


def enhance_audio_for_pyannote(audio_path: str) -> str:
    import subprocess

    progress("pyannote_audio_enhance", "增强小音量语音供 pyannote 使用", 23)
    output = tempfile.NamedTemporaryFile(suffix=".pyannote.enhanced.wav", delete=False)
    output.close()
    output_path = output.name
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
                "-af",
                "highpass=f=80,lowpass=f=7600,dynaudnorm=f=150:g=15:p=0.95:m=10,acompressor=threshold=-28dB:ratio=3:attack=8:release=120,loudnorm=I=-18:TP=-1.5:LRA=11",
                "-f",
                "wav",
                output_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        log_event("pyannote_audio_enhanced", path=output_path)
        return output_path
    except Exception:
        Path(output_path).unlink(missing_ok=True)
        raise


def prepare_audio_for_pyannote(audio_path: str) -> tuple[str, bool]:
    try:
        return enhance_audio_for_pyannote(audio_path), True
    except Exception as exc:
        log_event("pyannote_audio_enhance_failed", error=str(exc))
        progress("pyannote_audio_enhance_fallback", "pyannote 音频增强失败，改用原始音频", 23)
        return audio_path, False


def load_audio_for_pyannote(audio_path: str):
    import numpy as np
    import subprocess
    import torch

    progress("pyannote_audio", "转换音频为 pyannote waveform", 25)
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
        progress("pyannote_audio_ready", "pyannote waveform 准备完成", 30)
        return {"waveform": waveform, "sample_rate": sample_rate}
    finally:
        Path(wav_path).unlink(missing_ok=True)


class PyannoteProgressHook:
    step_ranges = {
        "segmentation": (33, 45),
        "speaker_counting": (45, 49),
        "embeddings": (49, 58),
        "discrete_diarization": (58, 62),
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
        start, end = self.step_ranges.get(step_name, (35, 62))
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


def diarize_with_pyannote(args, torch, Pipeline):
    progress("load_pyannote", "加载 pyannote diarization pipeline", 20)
    pipeline, temp_pipeline_config = load_pyannote_pipeline(Pipeline, args.pyannote_cache_dir, args.pyannote_auth_token, args.pyannote_use_local_config)

    device = resolve_torch_device(torch)
    pipeline.to(device)
    log_event("pyannote_start", device=str(device), cacheDir=args.pyannote_cache_dir, rssMb=current_rss_mb())
    try:
        pyannote_audio_path, should_remove_pyannote_audio = prepare_audio_for_pyannote(args.audio_path)
        audio = load_audio_for_pyannote(pyannote_audio_path)
        audio = move_audio_to_device(audio, device)
        progress("diarize", "pyannote 开始分离说话人", 32)
        diarization_output = pipeline(audio, hook=PyannoteProgressHook())
        progress("parse_pyannote", "整理 pyannote 说话人结果", 63)
        diarization = annotation_from_pyannote_output(diarization_output)
    finally:
        if should_remove_pyannote_audio:
            Path(pyannote_audio_path).unlink(missing_ok=True)
            log_event("pyannote_enhanced_audio_removed", path=pyannote_audio_path)

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
        release_inference_memory(torch)
        if temp_pipeline_config:
            Path(temp_pipeline_config).unlink(missing_ok=True)
    log_event("pyannote_done", segmentCount=len(segments), rssMb=current_rss_mb())
    return {"modelName": "pyannote/speaker-diarization-3.1", "segments": segments}


def speaker_segment_key(segment: dict) -> str:
    return segment.get("speakerClusterId") or segment.get("speakerLabel") or "unknown"


def speaker_segment_duration_ms(segment: dict) -> int:
    return int(segment["endMs"]) - int(segment["startMs"])


def absorb_short_speaker_segments(segments: list[dict], min_duration_ms: int, max_gap_ms: int, max_duration_ms: int) -> list[dict]:
    if min_duration_ms <= 0 or len(segments) <= 1:
        return [segment.copy() for segment in segments]

    output = [segment.copy() for segment in sorted(segments, key=lambda item: item["startMs"])]
    index = 0
    while index < len(output):
        segment = output[index]
        if speaker_segment_duration_ms(segment) >= min_duration_ms:
            index += 1
            continue

        previous = output[index - 1] if index > 0 else None
        next_segment = output[index + 1] if index + 1 < len(output) else None
        previous_gap_ms = segment["startMs"] - previous["endMs"] if previous else float("inf")
        next_gap_ms = next_segment["startMs"] - segment["endMs"] if next_segment else float("inf")
        can_use_previous = previous is not None and (max_gap_ms < 0 or previous_gap_ms <= max_gap_ms)
        can_use_next = next_segment is not None and (max_gap_ms < 0 or next_gap_ms <= max_gap_ms)

        if (
            previous
            and next_segment
            and can_use_previous
            and can_use_next
            and speaker_segment_key(previous) == speaker_segment_key(next_segment)
            and speaker_segment_key(previous) != speaker_segment_key(segment)
            and (max_duration_ms <= 0 or next_segment["endMs"] - previous["startMs"] <= max_duration_ms)
        ):
            previous["endMs"] = next_segment["endMs"]
            output.pop(index + 1)
            output.pop(index)
            index = max(0, index - 1)
            continue

        previous_duration_ms = segment["endMs"] - previous["startMs"] if previous else float("inf")
        next_duration_ms = next_segment["endMs"] - segment["startMs"] if next_segment else float("inf")
        can_absorb_previous = previous and can_use_previous and (max_duration_ms <= 0 or previous_duration_ms <= max_duration_ms)
        can_absorb_next = next_segment and can_use_next and (max_duration_ms <= 0 or next_duration_ms <= max_duration_ms)

        if not can_absorb_previous and not can_absorb_next:
            index += 1
            continue

        if can_absorb_previous and (not can_absorb_next or previous_gap_ms <= next_gap_ms):
            previous["endMs"] = max(previous["endMs"], segment["endMs"])
            output.pop(index)
            index = max(0, index - 1)
            continue

        next_segment["startMs"] = min(next_segment["startMs"], segment["startMs"])
        output.pop(index)

    return output


def merge_speaker_segments(segments: list[dict], max_gap_ms: int, max_duration_ms: int) -> list[dict]:
    merged = []
    for segment in sorted(segments, key=lambda item: item["startMs"]):
        if not merged:
            merged.append(segment.copy())
            continue
        current = merged[-1]
        gap_ms = segment["startMs"] - current["endMs"]
        next_duration_ms = segment["endMs"] - current["startMs"]
        can_merge = (
            speaker_segment_key(current) == speaker_segment_key(segment)
            and gap_ms >= 0
            and (max_gap_ms < 0 or gap_ms <= max_gap_ms)
            and (max_duration_ms <= 0 or next_duration_ms <= max_duration_ms)
        )
        if can_merge:
            current["endMs"] = max(current["endMs"], segment["endMs"])
        else:
            merged.append(segment.copy())
    return merged


def prepare_speaker_segments_for_asr(args, diarization: dict) -> list[dict]:
    raw_segments = diarization.get("segments") or []
    smoothed = absorb_short_speaker_segments(
        raw_segments,
        args.speaker_segment_min_duration_ms,
        args.speaker_segment_merge_max_gap_ms,
        args.speaker_segment_merge_max_duration_ms,
    )
    merged = merge_speaker_segments(
        smoothed,
        args.speaker_segment_merge_max_gap_ms,
        args.speaker_segment_merge_max_duration_ms,
    )
    log_event(
        "speaker_segments_prepared",
        rawCount=len(raw_segments),
        smoothedCount=len(smoothed),
        mergedCount=len(merged),
        minDurationMs=args.speaker_segment_min_duration_ms,
        mergeMaxGapMs=args.speaker_segment_merge_max_gap_ms,
        mergeMaxDurationMs=args.speaker_segment_merge_max_duration_ms,
    )
    return merged


def load_external_segments(path: str) -> list[dict]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        return []
    segments = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        start_ms = int(item.get("startMs") or 0)
        end_ms = int(item.get("endMs") or 0)
        if start_ms < 0 or end_ms <= start_ms:
            continue
        segments.append(
            {
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerLabel": normalize_text(item.get("speakerLabel")) or None,
                "speakerClusterId": normalize_text(item.get("speakerClusterId")) or None,
                "speakerConfidence": item.get("speakerConfidence"),
            }
        )
    return sorted(segments, key=lambda segment: segment["startMs"])


def crop_wav_with_ffmpeg(source: str, start_ms: int, end_ms: int) -> str:
    import subprocess

    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-ss",
            str(start_ms / 1000),
            "-to",
            str(end_ms / 1000),
            "-ar",
            "16000",
            "-ac",
            "1",
            output.name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return output.name


def clean_wav_with_ffmpeg(source: str) -> str:
    import subprocess

    output = tempfile.NamedTemporaryFile(suffix=".clean.wav", delete=False)
    output.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            "highpass=f=80,lowpass=f=7600,loudnorm",
            output.name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return output.name


def enhance_asr_clip_with_ffmpeg(source: str) -> str:
    import subprocess

    output = tempfile.NamedTemporaryFile(suffix=".asr.enhanced.wav", delete=False)
    output.close()
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                source,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-af",
                "highpass=f=80,lowpass=f=7600,dynaudnorm=f=120:g=12:p=0.9:m=8,acompressor=threshold=-30dB:ratio=2.5:attack=6:release=100,loudnorm=I=-19:TP=-1.5:LRA=11",
                output.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return output.name
    except Exception:
        Path(output.name).unlink(missing_ok=True)
        raise


def wav_volume_stats(path: str) -> dict:
    try:
        import numpy as np

        with wave.open(path, "rb") as wav:
            sample_width = wav.getsampwidth()
            channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
        if sample_width != 2 or not frames:
            return {"rms": 0.0, "peak": 0.0}
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if audio.size == 0:
            return {"rms": 0.0, "peak": 0.0}
        return {
            "rms": float(np.sqrt(np.mean(np.square(audio)))),
            "peak": float(np.max(np.abs(audio))),
        }
    except Exception as exc:
        log_event("segment_volume_stats_failed", path=path, error=str(exc))
        return {"rms": 1.0, "peak": 1.0}


def maybe_enhance_low_volume_clip(args, clip_path: str, metadata: dict) -> tuple[str, bool, dict]:
    if not args.enhance_low_volume_segments:
        return clip_path, False, {}
    stats = wav_volume_stats(clip_path)
    low_volume = stats["rms"] < args.low_volume_rms_threshold and stats["peak"] < args.low_volume_peak_threshold
    log_event(
        "segment_volume_checked",
        **metadata,
        rms=round(stats["rms"], 6),
        peak=round(stats["peak"], 6),
        lowVolume=low_volume,
        rmsThreshold=args.low_volume_rms_threshold,
        peakThreshold=args.low_volume_peak_threshold,
    )
    if not low_volume:
        return clip_path, False, stats
    try:
        enhanced_path = enhance_asr_clip_with_ffmpeg(clip_path)
        log_event("segment_low_volume_enhanced", **metadata, enhancedPath=enhanced_path)
        return enhanced_path, True, stats
    except Exception as exc:
        log_event("segment_low_volume_enhance_failed", **metadata, error=str(exc))
        return clip_path, False, stats


def detect_device(torch):
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_qwen_model(args, torch, Qwen3ASRModel):
    device_map, dtype = detect_device(torch)
    try:
        import transformers.modeling_utils as modeling_utils

        modeling_utils.caching_allocator_warmup = lambda *unused_args, **unused_kwargs: None
    except Exception:
        pass

    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.use_own_segments and args.forced_aligner_model:
        kwargs["forced_aligner"] = args.forced_aligner_model
        kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": device_map}
    return Qwen3ASRModel.from_pretrained(args.model, **kwargs)


def transcribe_qwen(model, audio, language, context, return_time_stamps=False):
    return model.transcribe(
        audio=audio,
        context=context or "",
        language=language,
        return_time_stamps=return_time_stamps,
    )


def release_inference_memory(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def transcribe_clip_once(args, model, language, torch, clip_path: str):
    with segment_timeout(args.segment_timeout_s):
        with torch.inference_mode():
            with contextlib.redirect_stdout(sys.stderr):
                return transcribe_qwen(model, clip_path, language, args.context, False)


def transcribe_clip_with_clean_retry(args, model, language, torch, clip_path: str, metadata: dict):
    asr_clip_path = clip_path
    remove_asr_clip = False
    volume_stats = {}
    try:
        asr_clip_path, remove_asr_clip, volume_stats = maybe_enhance_low_volume_clip(args, clip_path, metadata)
    except Exception as exc:
        log_event("segment_low_volume_prepare_failed", **metadata, error=str(exc))

    started_at = time.monotonic()
    attempt_name = "enhanced_low_volume" if remove_asr_clip else "original"
    log_event("segment_start", **metadata, attempt=attempt_name, volume=volume_stats, rssMb=current_rss_mb())
    try:
        result = transcribe_clip_once(args, model, language, torch, asr_clip_path)
        log_event("segment_done", **metadata, attempt=attempt_name, elapsedMs=int((time.monotonic() - started_at) * 1000), rssMb=current_rss_mb())
        return result
    except TimeoutError as exc:
        log_event("segment_timeout", **metadata, attempt=attempt_name, error=str(exc), elapsedMs=int((time.monotonic() - started_at) * 1000), rssMb=current_rss_mb())
        release_inference_memory(torch)
    finally:
        if remove_asr_clip:
            Path(asr_clip_path).unlink(missing_ok=True)
            log_event("segment_low_volume_enhanced_removed", **metadata, enhancedPath=asr_clip_path, rssMb=current_rss_mb())

    clean_path = None
    try:
        clean_path = clean_wav_with_ffmpeg(clip_path)
        retry_started_at = time.monotonic()
        log_event("segment_clean_retry_start", **metadata, cleanPath=clean_path, rssMb=current_rss_mb())
        result = transcribe_clip_once(args, model, language, torch, clean_path)
        log_event("segment_clean_retry_done", **metadata, elapsedMs=int((time.monotonic() - retry_started_at) * 1000), rssMb=current_rss_mb())
        return result
    finally:
        if clean_path:
            Path(clean_path).unlink(missing_ok=True)
            log_event("segment_clean_removed", **metadata, cleanPath=clean_path, rssMb=current_rss_mb())


def transcribe_with_vad(args, model, AutoModel, language, torch):
    progress("load_vad", f"加载 VAD 模型 {args.vad_model}", 35)
    vad_model = AutoModel(model=args.vad_model, hub="hf")
    progress("vad", "检测语音片段", 42)
    vad_result = vad_model.generate(input=args.audio_path)
    vad_segments = extract_vad_segments(vad_result)
    del vad_result
    del vad_model
    release_inference_memory(torch)
    if args.merge_vad:
        vad_segments = merge_vad_segments(vad_segments, args.merge_length_s * 1000, args.vad_merge_max_gap_ms, args.vad_min_segment_ms)
    vad_segments = split_long_segments(vad_segments, args.vad_max_segment_ms)

    segments = []
    total = max(1, len(vad_segments))
    for index, (start_ms, end_ms) in enumerate(vad_segments):
        percent = int(65 + (index / total) * 27)
        progress("transcribe", f"Qwen3-ASR 转写语音片段 {index + 1}/{total}", percent)
        clip_path = None
        result = None
        try:
            clip_path = crop_wav_with_ffmpeg(args.audio_path, start_ms, end_ms)
            result = transcribe_clip_with_clean_retry(
                args,
                model,
                language,
                torch,
                clip_path,
                {
                    "index": index + 1,
                    "total": total,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "durationMs": end_ms - start_ms,
                },
            )
            text = extract_text(result)
            if args.strip_trailing_punctuation:
                text = strip_trailing_punctuation(text)
            if text:
                segments.append({"startMs": start_ms, "endMs": end_ms, "text": text})
        except Exception as exc:
            print(f"Skipping Qwen3-ASR segment {start_ms}-{end_ms}: {exc}", file=sys.stderr)
        finally:
            del result
            if clip_path:
                Path(clip_path).unlink(missing_ok=True)
            release_inference_memory(torch)
            log_event("segment_cleanup", index=index + 1, total=total, rssMb=current_rss_mb())
    return segments


def transcribe_with_external_segments(args, model, language, torch):
    source_segments = load_external_segments(args.segments_json_path)
    return transcribe_with_speaker_segments(args, model, language, torch, source_segments)


def transcribe_with_speaker_segments(args, model, language, torch, source_segments: list[dict]):
    if not source_segments:
        return []

    segments = []
    total = max(1, len(source_segments))
    for index, source_segment in enumerate(source_segments):
        start_ms = source_segment["startMs"]
        end_ms = source_segment["endMs"]
        percent = int(65 + (index / total) * 27)
        speaker = source_segment.get("speakerLabel") or "unknown speaker"
        time_range = f"{start_ms / 1000:.2f}s-{end_ms / 1000:.2f}s"
        progress("transcribe", f"Qwen3-ASR 转写说话人片段 {index + 1}/{total} ({speaker}, {time_range})", percent)
        clip_path = None
        result = None
        try:
            clip_path = crop_wav_with_ffmpeg(args.audio_path, start_ms, end_ms)
            result = transcribe_clip_with_clean_retry(
                args,
                model,
                language,
                torch,
                clip_path,
                {
                    "index": index + 1,
                    "total": total,
                    "speaker": speaker,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "durationMs": end_ms - start_ms,
                },
            )
            text = extract_text(result)
            if args.strip_trailing_punctuation:
                text = strip_trailing_punctuation(text)
            if text:
                segments.append(
                    {
                        "startMs": start_ms,
                        "endMs": end_ms,
                        "text": text,
                        "speakerLabel": source_segment.get("speakerLabel"),
                        "speakerClusterId": source_segment.get("speakerClusterId"),
                        "speakerConfidence": source_segment.get("speakerConfidence"),
                    }
                )
        except Exception as exc:
            print(f"Skipping Qwen3-ASR speaker segment {start_ms}-{end_ms}: {exc}", file=sys.stderr)
        finally:
            del result
            if clip_path:
                Path(clip_path).unlink(missing_ok=True)
            release_inference_memory(torch)
            log_event("segment_cleanup", index=index + 1, total=total, rssMb=current_rss_mb())
    return segments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--pyannote-cache-dir", default=os.environ.get("HUGGINGFACE_HUB_CACHE"))
    parser.add_argument("--pyannote-auth-token", default=os.environ.get("PYANNOTE_AUTH_TOKEN"))
    parser.add_argument("--no-pyannote-local-config", action="store_true")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--context", default="")
    parser.add_argument("--forced-aligner-model", default="")
    parser.add_argument("--use-own-segments", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--segment-timeout-s", type=int, default=180)
    parser.add_argument("--enhance-low-volume-segments", action="store_true")
    parser.add_argument("--low-volume-rms-threshold", type=float, default=0.008)
    parser.add_argument("--low-volume-peak-threshold", type=float, default=0.04)
    parser.add_argument("--speaker-segment-merge-max-gap-ms", type=int, default=2000)
    parser.add_argument("--speaker-segment-merge-max-duration-ms", type=int, default=60000)
    parser.add_argument("--speaker-segment-min-duration-ms", type=int, default=1200)
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--vad-max-segment-ms", type=int, default=30000)
    parser.add_argument("--vad-merge-max-gap-ms", type=int, default=1200)
    parser.add_argument("--vad-min-segment-ms", type=int, default=1200)
    parser.add_argument("--merge-length-s", type=int, default=15)
    parser.add_argument("--merge-vad", action="store_true")
    parser.add_argument("--strip-trailing-punctuation", action="store_true")
    parser.add_argument("--break-on-sentence-end", action="store_true")
    parser.add_argument("--segments-json-path", default="")
    args = parser.parse_args()
    args.pyannote_use_local_config = not args.no_pyannote_local_config

    if use_pyannote_offline_if_complete(args.pyannote_cache_dir, args.pyannote_use_local_config):
        log_event("pyannote_offline_enabled_before_import", cacheDir=args.pyannote_cache_dir)

    with contextlib.redirect_stdout(sys.stderr):
        progress("import", "加载 Qwen3-ASR 依赖", 5)
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
        external_segments = load_external_segments(args.segments_json_path)
        try:
            import torch
            from funasr import AutoModel
            from qwen_asr import Qwen3ASRModel
            if not args.segments_json_path:
                from pyannote.audio import Pipeline
        except Exception as exc:
            raise RuntimeError("qwen-asr/funasr/torch is not installed. Install the Qwen ASR Python dependencies first.") from exc

        if args.cache_dir:
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(Path(args.cache_dir).parent / "huggingface")
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.cache_dir).parent / "huggingface" / "hub")
            os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

        if args.pyannote_cache_dir and not args.segments_json_path:
            Path(args.pyannote_cache_dir).mkdir(parents=True, exist_ok=True)

        if args.segments_json_path:
            diarization = {"provider": "external", "modelName": "external", "segments": external_segments}
            speaker_segments = []
        else:
            diarization = diarize_with_pyannote(args, torch, Pipeline)
            speaker_segments = prepare_speaker_segments_for_asr(args, diarization)

        progress("load_model", f"加载 Qwen3-ASR 模型 {args.model}", 64)
        model = load_qwen_model(args, torch, Qwen3ASRModel)

    language = language_arg(args.language)
    language_result = None
    segments = []

    if args.segments_json_path:
        segments = transcribe_with_external_segments(args, model, language, torch)
        if segments:
            progress("segment", "Qwen3-ASR 说话人片段转写完成", 92)
        else:
            print("Qwen3-ASR did not return usable speaker segment transcription; falling back to fsmn-vad.", file=sys.stderr)

    if not segments and speaker_segments:
        segments = transcribe_with_speaker_segments(args, model, language, torch, speaker_segments)
        if segments:
            progress("segment", "Qwen3-ASR pyannote 说话人片段转写完成", 92)
        else:
            print("Qwen3-ASR did not return usable pyannote speaker segment transcription; falling back to fsmn-vad.", file=sys.stderr)

    if not segments and args.use_own_segments and args.forced_aligner_model:
        try:
            progress("transcribe", "Qwen3-ASR 转写并生成时间戳", 35)
            with contextlib.redirect_stdout(sys.stderr):
                result = transcribe_qwen(model, args.audio_path, language, args.context, True)
            language_result = extract_language(result)
            segments = extract_timestamp_segments(result, args.strip_trailing_punctuation, args.break_on_sentence_end)
            if segments:
                progress("segment", "Qwen3-ASR 时间戳分段完成", 92)
            else:
                print("Qwen3-ASR did not return usable timestamp segments; falling back to fsmn-vad.", file=sys.stderr)
        except Exception as exc:
            print(f"Qwen3-ASR timestamp segmentation failed; falling back to fsmn-vad: {exc}", file=sys.stderr)

    if not segments:
        with contextlib.redirect_stdout(sys.stderr):
            segments = transcribe_with_vad(args, model, AutoModel, language, torch)

    payload = {
        "language": language_result or language,
        "modelName": args.model,
        "diarization": diarization,
        "fullText": "".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
    }
    progress("done", "Qwen3-ASR 转写完成", 100)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
