#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lib.local_llm.scripts.chat_template import STOP_TOKENS, chat_prompt


def citation_payload(evidence, indices):
    by_index = {item["index"]: item for item in evidence}
    citations = []
    for index in indices:
        item = by_index.get(index)
        if not item:
            continue
        citations.append({
            "index": item["index"],
            "chunkId": item["chunk"]["id"],
            "recordingId": item["recording"]["id"],
            "startMs": item["chunk"]["startMs"],
            "endMs": item["chunk"]["endMs"],
        })
    return citations


def build_messages(query, evidence):
    items = []
    for item in evidence:
        location = item["recording"].get("location") or "未配置"
        items.append(
            f"[{item['index']}] 录音标题：{item['recording']['title']}\n"
            f"地点：{location}\n"
            f"时间：{item['chunk']['startMs']}-{item['chunk']['endMs']}ms\n"
            f"{item['chunk']['text']}"
        )
    system = (
        "你是一个谨慎的录音证据总结助手。只能基于证据回答，不能编造。\n"
        "如果用户要求分别说明、逐条说明、每个录音说了什么，或问题是在问某个时间范围内的录音分别说了什么，必须按录音标题分段回答。\n"
        "分段格式必须是：每条录音独立一段，段首写《录音标题》：，该段只总结这条录音的内容，并在段末或关键结论后带对应证据编号。\n"
        "不要把多条录音合并成一段；如果有多条证据，每条证据至少对应一个独立段落。\n"
        "直接输出面向用户的中文回答，不要输出 JSON，不要输出 Markdown 标题。\n"
        "每个关键结论后必须带证据编号，例如 [1]、[2]。\n"
        "如果证据不足，只回答：没有在录音中找到足够依据。\n"
    )
    user = f"用户问题：{query}\n\n证据：\n" + "\n\n".join(items)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt(llm, query, evidence):
    return chat_prompt(llm, build_messages(query, evidence))


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    payload = json.load(sys.stdin)
    evidence = payload.get("evidence") or []
    if not evidence:
        answer = {"text": "没有在录音中找到足够依据。", "citations": [], "notEnoughEvidence": True}
        emit({"type": "delta", "text": answer["text"]})
        emit({"type": "done", "answer": answer})
        return

    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is required for local RAG answers") from exc

    llm = Llama(model_path=payload["modelPath"], n_ctx=int(payload.get("contextSize") or 8192), verbose=False)
    prompt = build_prompt(llm, payload["query"], evidence)
    output = llm(prompt, max_tokens=1200, temperature=0.1, stop=STOP_TOKENS, stream=True)

    text_parts = []
    thinking_parts = []
    thinking = False
    for chunk in output:
        piece = chunk.get("choices", [{}])[0].get("text", "")
        if not piece:
            continue
        cursor = 0
        while cursor < len(piece):
            if thinking:
                end = piece.find("</think>", cursor)
                if end < 0:
                    thinking_parts.append(piece[cursor:])
                    break
                thinking_parts.append(piece[cursor:end])
                emit({"type": "thinking_done", "text": "".join(thinking_parts).strip()})
                thinking = False
                cursor = end + len("</think>")
                continue

            start = piece.find("<think>", cursor)
            if start < 0:
                delta = piece[cursor:]
                text_parts.append(delta)
                emit({"type": "delta", "text": delta})
                break

            delta = piece[cursor:start]
            if delta:
                text_parts.append(delta)
                emit({"type": "delta", "text": delta})
            thinking = True
            thinking_parts = []
            emit({"type": "thinking_start"})
            cursor = start + len("<think>")

    text = "".join(text_parts).strip()
    used_indices = []
    for match in re.findall(r"\[(\d+)\]", text):
        index = int(match)
        if index not in used_indices:
            used_indices.append(index)
    if not used_indices and text:
        used_indices = [item["index"] for item in evidence[: min(3, len(evidence))]]

    answer = {
        "text": text or "没有在录音中找到足够依据。",
        "citations": citation_payload(evidence, used_indices),
        "notEnoughEvidence": not bool(text) or "没有在录音中找到足够依据" in text,
    }
    emit({"type": "done", "answer": answer})


if __name__ == "__main__":
    main()
