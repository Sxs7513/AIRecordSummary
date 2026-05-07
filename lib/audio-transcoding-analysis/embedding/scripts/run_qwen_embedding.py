#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


def local_model_snapshot(cache_dir: str, model_name: str) -> str | None:
    repo_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    required_files = ["config.json", "modules.json", "tokenizer.json"]
    model_files = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
    for snapshot in snapshots:
        has_required_files = all((snapshot / file_name).exists() for file_name in required_files)
        has_model_file = any((snapshot / file_name).exists() for file_name in model_files)
        has_sharded_safetensors = any(snapshot.glob("model-*.safetensors"))
        if has_required_files and (has_model_file or has_sharded_safetensors):
            return str(snapshot.resolve())
    return None


def resolve_model_source(cache_dir: str, model_name: str) -> str:
    os.environ.setdefault("HF_HOME", str(Path(cache_dir).resolve()))
    snapshot = local_model_snapshot(cache_dir, model_name)
    if not snapshot:
        return model_name
    print(f"[embedding] loading local snapshot: {snapshot}", file=sys.stderr)
    return snapshot


def detect_device(requested):
    if requested and requested != "auto":
        return requested

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=["query", "document"], default="document")
    parser.add_argument("--texts-json", required=True)
    args = parser.parse_args()

    texts = json.loads(args.texts_json)
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("texts-json must be a JSON array of strings")
    if any(not text.strip() for text in texts):
        raise ValueError("embedding input contains empty text")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required for local embedding") from exc

    device = detect_device(args.device)
    print(f"[embedding] using device: {device}", file=sys.stderr)
    model_source = resolve_model_source(args.cache_dir, args.model)
    use_local_snapshot = Path(model_source).exists()
    if use_local_snapshot:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    model = SentenceTransformer(model_source, cache_folder=args.cache_dir, trust_remote_code=True, device=device, local_files_only=use_local_snapshot)
    encode_kwargs = {
        "batch_size": max(1, args.batch_size),
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if args.mode == "query":
        encode_kwargs["prompt_name"] = "query"

    embeddings = model.encode(texts, **encode_kwargs)
    sys.stdout.write(json.dumps({"embeddings": embeddings.tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
