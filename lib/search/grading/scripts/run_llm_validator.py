#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lib.local_llm.scripts.chat_template import STOP_TOKENS, chat_prompt, messages_from_chatml


def stop_tokens(payload):
    if payload.get("stop"):
        return payload["stop"]
    return STOP_TOKENS


def main():
    payload = json.load(sys.stdin)
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is required for answer validation") from exc

    llm = Llama(model_path=payload["modelPath"], n_ctx=int(payload.get("contextSize") or 4096), verbose=False)
    prompt = chat_prompt(llm, messages_from_chatml(payload["prompt"]))
    output = llm(
        prompt,
        max_tokens=int(payload.get("maxTokens") or 700),
        temperature=float(payload.get("temperature") or 0.0),
        stop=stop_tokens(payload),
    )
    print(json.dumps({"text": output["choices"][0]["text"].strip()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
