#!/usr/bin/env python3
import argparse
import contextlib
import importlib
import json
import ssl
import sys
from pathlib import Path


def progress(stage: str, message: str, percent: int):
    print(
        "PROGRESS_JSON:" + json.dumps({"stage": stage, "message": message, "percent": percent}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )


class WhisperProgressBar:
    def __init__(self, *args, total=None, disable=False, **kwargs):
        self.total = total or 0
        self.disable = disable
        self.completed = 0
        self.last_percent = None
        if self.total and not self.disable:
            self.emit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.total and not self.disable:
            self.completed = self.total
            self.emit(force=True)

    def update(self, amount):
        if self.disable or not self.total:
            return
        self.completed = min(self.total, self.completed + amount)
        self.emit()

    def emit(self, force=False):
        ratio = max(0.0, min(1.0, self.completed / self.total))
        percent = int(20 + ratio * 70)
        if not force and percent == self.last_percent:
            return
        self.last_percent = percent
        progress(
            "transcribe",
            f"Whisper 正在转写音频 {self.completed}/{self.total} frames",
            percent,
        )


def is_low_quality_segment(segment: dict) -> bool:
    text = segment.get("text", "").strip()
    if not text:
        return True
    if float(segment.get("no_speech_prob") or 0) > 0.85:
        return True
    if float(segment.get("avg_logprob") or 0) < -1.2:
        return True
    return False


def is_repeated_template(text: str, seen: dict[str, int]) -> bool:
    normalized = "".join(text.split())
    if len(normalized) < 12:
        return False
    seen[normalized] = seen.get(normalized, 0) + 1
    return seen[normalized] > 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--initial-prompt", default=None)
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        progress("import", "加载 Whisper 依赖", 5)
        try:
            import certifi

            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
        try:
            import whisper
        except Exception as exc:
            raise RuntimeError("openai-whisper is not installed. Run scripts/install_audio_dependencies.sh first.") from exc

        if args.cache_dir:
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        progress("load_model", f"加载 Whisper 模型 {args.model}", 15)
        model = whisper.load_model(args.model, download_root=args.cache_dir)
        whisper_transcribe = importlib.import_module("whisper.transcribe")
        whisper_transcribe.tqdm.tqdm = WhisperProgressBar
        progress("transcribe", "Whisper 开始转写音频", 20)
        result = model.transcribe(
            args.audio_path,
            verbose=False,
            language=args.language,
            initial_prompt=args.initial_prompt,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        progress("parse", "整理 Whisper 转写结果", 92)
    seen_templates = {}
    segments = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if is_low_quality_segment(segment) or is_repeated_template(text, seen_templates):
            print(
                "[whisper] filtered segment "
                + json.dumps(
                    {
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "text": text[:80],
                        "avg_logprob": segment.get("avg_logprob"),
                        "no_speech_prob": segment.get("no_speech_prob"),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            continue
        segments.append(
            {
                "startMs": int(float(segment["start"]) * 1000),
                "endMs": int(float(segment["end"]) * 1000),
                "text": text,
            }
        )
    full_text = "".join(segment["text"] for segment in segments).strip()
    payload = {
        "language": result.get("language"),
        "modelName": f"whisper-{args.model}",
        "fullText": full_text,
        "segments": segments,
    }
    progress("done", "Whisper 转写完成", 100)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
