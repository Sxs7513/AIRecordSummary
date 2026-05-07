#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lib.local_llm.scripts.chat_template import STOP_TOKENS, chat_prompt


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
        "每个关键结论后必须带 [编号] 引用。只输出 JSON，不要输出 Markdown，不要解释。\n"
        "JSON schema: {\"text\":\"回答文本，关键结论带 [编号]；分段时使用换行换行分隔段落\",\"citations\":[{\"index\":1}],\"notEnoughEvidence\":false}\n"
        "如果证据不足，输出 {\"text\":\"没有在录音中找到足够依据。\",\"citations\":[],\"notEnoughEvidence\":true}\n"
    )
    user = f"用户问题：{query}\n\n证据：\n" + "\n\n".join(items)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt(llm, query, evidence):
    return chat_prompt(llm, build_messages(query, evidence))


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


def asks_for_separate_summary(query):
    keywords = ["分别", "逐条", "每个录音", "每条录音", "各自", "都说了什么", "分别说了什么", "分别都说了什么"]
    return any(keyword in query for keyword in keywords)


def needs_segmented_fallback(text, evidence):
    if len(evidence) <= 1:
        return False
    paragraph_count = len([part for part in text.split("\n\n") if part.strip()])
    title_count = sum(1 for item in evidence if f"《{item['recording']['title']}》" in text)
    citation_count = sum(1 for item in evidence if f"[{item['index']}]" in text)
    return paragraph_count < min(2, len(evidence)) or title_count < len(evidence) or citation_count < len(evidence)


def normalize_answer(parsed, evidence, query):
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
    if not not_enough and asks_for_separate_summary(query) and needs_segmented_fallback(text, evidence):
        text = build_extractive_text(evidence)
        citations = citation_payload(evidence)
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
        parts.append(f"《{item['recording']['title']}》：{item['chunk']['text']} [{item['index']}]")
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

    llm = Llama(model_path=payload["modelPath"], n_ctx=int(payload.get("contextSize") or 8192), verbose=False)
    prompt = build_prompt(llm, payload["query"], evidence)
    output = llm(prompt, max_tokens=1200, temperature=0.1, stop=STOP_TOKENS)
    text = output["choices"][0]["text"].strip()
    json_text = first_json_object(text)
    if json_text:
        try:
            print(json.dumps(normalize_answer(json.loads(json_text), evidence, payload["query"]), ensure_ascii=False))
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
