from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from inference import serve
from trainer import train

logger = logging.getLogger("train")
evaluation_logger = logging.getLogger("evaluation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated Qwen3-ASR Hugging Face LoRA runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Run LoRA training from an immutable manifest")
    train_parser.add_argument("--manifest", type=Path, required=True)
    serve_parser = subparsers.add_parser("serve", help="Serve JSON-lines transcription over stdin/stdout")
    serve_parser.add_argument("--model-path", type=Path, required=True)
    serve_parser.add_argument("--adapter-path", type=Path)
    serve_parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.command == "train":
        logger.info("Qwen ASR LoRA：收到训练命令 manifest=%s", args.manifest)
        result = train(args.manifest)
        logger.info("Qwen ASR LoRA：训练命令执行成功 run_id=%s", result.get("run_id"))
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str), flush=True)
        return
    evaluation_logger.info(
        "Qwen ASR LoRA：启动推理服务 model_path=%s adapter_path=%s",
        args.model_path,
        args.adapter_path,
    )
    serve(
        model_path=args.model_path.resolve(),
        adapter_path=args.adapter_path.resolve() if args.adapter_path is not None else None,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
