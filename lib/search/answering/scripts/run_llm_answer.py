#!/usr/bin/env python3
import json
import sys


def first_json_object(text):
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return None


def build_prompt(query, evidence):
    items = []
    for item in evidence:
        items.append(
            f"[{item['index']}] 录音：{item['recording']['title']} "
            f"时间：{item['chunk']['startMs']}-{item['chunk']['endMs']}ms\n"
            f"{item['chunk']['text']}"
        )
    return (
        "<|im_start|>system\n"
        "你是一个谨慎的录音证据总结助手。只能基于证据回答，不能编造。\n"
        "每个关键结论后必须带 [编号] 引用。只输出 JSON，不要输出 Markdown，不要解释。\n"
        "JSON schema: {\"text\":\"回答文本，关键结论带 [编号]\",\"citations\":[{\"index\":1}],\"notEnoughEvidence\":false}\n"
        "如果证据不足，输出 {\"text\":\"没有在录音中找到足够依据。\",\"citations\":[],\"notEnoughEvidence\":true}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"用户问题：{query}\n\n证据：\n" + "\n\n".join(items) +
        "\n<|im_end|>\n<|im_start|>assistant\n"
    )


def citation_payload(evidence):
    citations = []
    for item in evidence:
        citations.append({
            "index": item["index"],
            "chunkId": item["chunk"]["id"],
            "recordingId": item["recording"]["id"],
            "startMs": item["chunk"]["startMs"],
            "endMs": item["chunk"]["endMs"],
        })
    return citations


def normalize_answer(parsed, evidence):
    citations_by_index = {item["index"]: citation for item, citation in zip(evidence, citation_payload(evidence))}
    citations = []
    for raw in parsed.get("citations") or []:
        try:
            index = int(raw.get("index"))
        except Exception:
            continue
        if index in citations_by_index:
            citations.append(citations_by_index[index])

    text = str(parsed.get("text") or "").strip()
    not_enough = bool(parsed.get("notEnoughEvidence"))
    if not text:
        text = "没有在录音中找到足够依据。" if not_enough else build_extractive_text(evidence[:3])
    if not not_enough and not citations and evidence:
        citations = citation_payload(evidence[: min(3, len(evidence))])
    return {
        "text": text,
        "citations": citations,
        "notEnoughEvidence": not_enough,
    }


def build_extractive_text(evidence):
    parts = []
    for item in evidence:
        parts.append(f"根据录音《{item['recording']['title']}》片段 [{item['index']}]：{item['chunk']['text']}")
    return "\n\n".join(parts) if parts else "没有在录音中找到足够依据。"


def main():
    payload = json.load(sys.stdin)
    evidence = payload.get("evidence") or []
    if not evidence:
        print(json.dumps({"text": "没有在录音中找到足够依据。", "citations": [], "notEnoughEvidence": True}, ensure_ascii=False))
        return

    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is required for local RAG answers") from exc

    prompt = build_prompt(payload["query"], evidence)
    llm = Llama(model_path=payload["modelPath"], n_ctx=int(payload.get("contextSize") or 8192), verbose=False)
    output = llm(prompt, max_tokens=1200, temperature=0.1, stop=["</s>", "<|im_end|>"])
    text = output["choices"][0]["text"].strip()
    json_text = first_json_object(text)
    if json_text:
        try:
            print(json.dumps(normalize_answer(json.loads(json_text), evidence), ensure_ascii=False))
            return
        except Exception as exc:
            print(f"[rag-answer] JSON parse failed: {exc}", file=sys.stderr)

    fallback = {
        "text": text or build_extractive_text(evidence[:3]),
        "citations": citation_payload(evidence[: min(3, len(evidence))]),
        "notEnoughEvidence": False,
    }
    print(json.dumps(fallback, ensure_ascii=False))


if __name__ == "__main__":
    main()
