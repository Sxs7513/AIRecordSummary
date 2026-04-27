#!/usr/bin/env python3
import argparse
import json
import sys


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
    model = SentenceTransformer(args.model, cache_folder=args.cache_dir, trust_remote_code=True, device=device)
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
