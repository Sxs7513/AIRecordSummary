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
    return str(value).strip()


def result_items(result):
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return [result]
    return []


def extract_vad_segments(result) -> list[list[int]]:
    segments = []
    for item in result_items(result):
        if isinstance(item, dict):
            value = item.get("value") or item.get("timestamp") or []
        else:
            value = item
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
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    return output.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--model", default="BELLE-2/Belle-whisper-large-v3-turbo-zh")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--chunk-length-s", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--vad-max-segment-ms", type=int, default=30000)
    parser.add_argument("--merge-length-s", type=int, default=15)
    parser.add_argument("--merge-vad", action="store_true")
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        progress("import", "加载 Hugging Face Whisper 依赖", 5)
        try:
            import torch
            from funasr import AutoModel
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError("transformers/funasr/torch is not installed. Run scripts/install_audio_dependencies.sh first.") from exc

        if args.cache_dir:
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
            os.environ["HF_HOME"] = str(Path(args.cache_dir).parent / "huggingface")

        progress("load_vad", f"加载 VAD 模型 {args.vad_model}", 10)
        vad_model = AutoModel(model=args.vad_model, hub="hf")
        progress("vad", "检测语音片段", 16)
        vad_result = vad_model.generate(input=args.audio_path)
        vad_segments = extract_vad_segments(vad_result)
        if args.merge_vad:
            vad_segments = merge_vad_segments(vad_segments, args.merge_length_s * 1000)
        vad_segments = split_long_segments(vad_segments, args.vad_max_segment_ms)

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        progress("load_model", f"加载 Hugging Face Whisper 模型 {args.model}", 25)
        pipeline_kwargs = {
            "model": args.model,
            "torch_dtype": torch_dtype,
            "device": device,
        }
        if args.cache_dir:
            pipeline_kwargs["model_kwargs"] = {"cache_dir": args.cache_dir}
        transcriber = pipeline("automatic-speech-recognition", **pipeline_kwargs)
        if args.language:
            transcriber.model.config.forced_decoder_ids = transcriber.tokenizer.get_decoder_prompt_ids(
                language=args.language,
                task="transcribe",
            )

    segments = []
    temp_paths = []
    try:
        total = max(1, len(vad_segments))
        for index, (start_ms, end_ms) in enumerate(vad_segments):
            percent = int(30 + (index / total) * 60)
            progress("transcribe", f"Hugging Face Whisper 转写语音片段 {index + 1}/{total}", percent)
            try:
                clip_path = crop_wav_with_ffmpeg(args.audio_path, start_ms, end_ms)
                temp_paths.append(clip_path)
                with contextlib.redirect_stdout(sys.stderr):
                    result = transcriber(
                        clip_path,
                        chunk_length_s=args.chunk_length_s,
                        batch_size=args.batch_size,
                        return_timestamps=False,
                    )
                text = normalize_text(result.get("text") if isinstance(result, dict) else result)
                if text:
                    segments.append({"startMs": start_ms, "endMs": end_ms, "text": text})
            except Exception as exc:
                print(f"Skipping Hugging Face Whisper segment {start_ms}-{end_ms}: {exc}", file=sys.stderr)
    finally:
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)

    payload = {
        "language": args.language or None,
        "modelName": "hf-whisper",
        "fullText": "".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
    }
    progress("done", "Hugging Face Whisper 转写完成", 100)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
