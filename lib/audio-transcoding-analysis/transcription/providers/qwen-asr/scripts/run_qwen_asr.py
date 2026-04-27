#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path


def progress(stage: str, message: str, percent: int):
    print(
        "PROGRESS_JSON:" + json.dumps({"stage": stage, "message": message, "percent": percent}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(normalize_text(item) for item in value).strip()
    return str(value).strip()


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


def segment_from_timestamp_item(item):
    text = normalize_text(get_attr(item, "text"))
    start = get_attr(item, "start_time", get_attr(item, "start", None))
    end = get_attr(item, "end_time", get_attr(item, "end", None))
    if not text or start is None or end is None:
        return None
    start_ms = seconds_to_ms(start)
    end_ms = seconds_to_ms(end)
    if end_ms <= start_ms:
        return None
    return {"startMs": start_ms, "endMs": end_ms, "text": text}


def merge_timestamp_segments(raw_segments, max_duration_ms: int = 12000):
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
        should_break = current["text"].endswith(sentence_endings) or current_duration > max_duration_ms
        if should_break:
            merged.append(segment.copy())
        else:
            current["endMs"] = segment["endMs"]
            current["text"] = f'{current["text"]}{segment["text"]}'.strip()
    return merged


def extract_timestamp_segments(result):
    raw_segments = []
    for item in result_items(result):
        time_stamps = normalize_timestamp_items(get_attr(item, "time_stamps", get_attr(item, "timestamps", None)))
        for stamp in time_stamps:
            segment = segment_from_timestamp_item(stamp)
            if segment:
                raw_segments.append(segment)
    return merge_timestamp_segments(sorted(raw_segments, key=lambda segment: segment["startMs"]))


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


def merge_vad_segments(segments: list[list[int]], max_merged_ms: int) -> list[list[int]]:
    if not segments or max_merged_ms <= 0:
        return segments
    merged = [segments[0]]
    for start_ms, end_ms in segments[1:]:
        current = merged[-1]
        if end_ms - current[0] <= max_merged_ms:
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


def transcribe_with_vad(args, model, AutoModel, language):
    progress("load_vad", f"加载 VAD 模型 {args.vad_model}", 35)
    vad_model = AutoModel(model=args.vad_model, hub="hf")
    progress("vad", "检测语音片段", 42)
    vad_result = vad_model.generate(input=args.audio_path)
    vad_segments = extract_vad_segments(vad_result)
    if args.merge_vad:
        vad_segments = merge_vad_segments(vad_segments, args.merge_length_s * 1000)
    vad_segments = split_long_segments(vad_segments, args.vad_max_segment_ms)

    segments = []
    temp_paths = []
    try:
        total = max(1, len(vad_segments))
        for index, (start_ms, end_ms) in enumerate(vad_segments):
            percent = int(45 + (index / total) * 50)
            progress("transcribe", f"Qwen3-ASR 转写语音片段 {index + 1}/{total}", percent)
            try:
                clip_path = crop_wav_with_ffmpeg(args.audio_path, start_ms, end_ms)
                temp_paths.append(clip_path)
                with contextlib.redirect_stdout(sys.stderr):
                    result = transcribe_qwen(model, clip_path, language, args.context, False)
                text = extract_text(result)
                if text:
                    segments.append({"startMs": start_ms, "endMs": end_ms, "text": text})
            except Exception as exc:
                print(f"Skipping Qwen3-ASR segment {start_ms}-{end_ms}: {exc}", file=sys.stderr)
    finally:
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)
    return segments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--context", default="")
    parser.add_argument("--forced-aligner-model", default="")
    parser.add_argument("--use-own-segments", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--vad-max-segment-ms", type=int, default=30000)
    parser.add_argument("--merge-length-s", type=int, default=15)
    parser.add_argument("--merge-vad", action="store_true")
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        progress("import", "加载 Qwen3-ASR 依赖", 5)
        try:
            import torch
            from funasr import AutoModel
            from qwen_asr import Qwen3ASRModel
        except Exception as exc:
            raise RuntimeError("qwen-asr/funasr/torch is not installed. Run scripts/install_audio_dependencies.sh first.") from exc

        if args.cache_dir:
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(Path(args.cache_dir).parent / "huggingface")
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.cache_dir).parent / "huggingface" / "hub")
            os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

        progress("load_model", f"加载 Qwen3-ASR 模型 {args.model}", 20)
        model = load_qwen_model(args, torch, Qwen3ASRModel)

    language = language_arg(args.language)
    language_result = None
    segments = []

    if args.use_own_segments and args.forced_aligner_model:
        try:
            progress("transcribe", "Qwen3-ASR 转写并生成时间戳", 35)
            with contextlib.redirect_stdout(sys.stderr):
                result = transcribe_qwen(model, args.audio_path, language, args.context, True)
            language_result = extract_language(result)
            segments = extract_timestamp_segments(result)
            if segments:
                progress("segment", "Qwen3-ASR 时间戳分段完成", 92)
            else:
                print("Qwen3-ASR did not return usable timestamp segments; falling back to fsmn-vad.", file=sys.stderr)
        except Exception as exc:
            print(f"Qwen3-ASR timestamp segmentation failed; falling back to fsmn-vad: {exc}", file=sys.stderr)

    if not segments:
        with contextlib.redirect_stdout(sys.stderr):
            segments = transcribe_with_vad(args, model, AutoModel, language)

    payload = {
        "language": language_result or language,
        "modelName": args.model,
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
