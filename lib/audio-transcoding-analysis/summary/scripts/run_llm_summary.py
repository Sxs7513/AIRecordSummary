#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lib.local_llm.scripts.chat_template import STOP_TOKENS, chat_prompt


DEFAULT_SYSTEM_PROMPT = (
    "请总结这段录音，只能根据录音文本写，不要编造。\n"
    "开头先写一个全局总结，用 1-2 段话说明这段录音整体在讨论什么、最重要的结论或结果是什么。\n"
    "按照录音里的先后顺序总结，可以自然分成几段，每段围绕一个真实讨论主题或连续发生的事情展开。\n"
    "每段标题要直接写具体主题，不要使用“阶段一/阶段二/片段一/片段二/第一阶段/第二阶段”这类流程标签。\n"
    "总结要比逐句复述更高一层，写清楚这段讨论在解决什么问题、形成了什么看法、有哪些结论或待办。\n"
    "不要机械写成“Speaker A 说……、Speaker B 说……”这种发言记录。只有人物身份本身重要时才提到人。\n"
    "每段都要保留具体事情、数字、结论和待办，不要只写空泛概括。\n"
    "用自然的大白话写，不要写成报告腔，也不要只写空泛概括。\n"
    "用 Markdown 输出。不要使用代码块或缩进代码格式。不要输出思考过程，不要输出 JSON。"
)

def first_json_object(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
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


def parse_json_object(text):
    json_text = first_json_object(text)
    if not json_text:
        return None
    try:
        return json.loads(json_text)
    except Exception:
        return None


def format_time(ms):
    total_seconds = max(0, int(ms or 0) // 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_utterances(utterances):
    text = "\n".join(
        f"[{format_time(item.get('startMs'))}-{format_time(item.get('endMs'))}] {item.get('speakerLabel') or 'Unknown Speaker'}: {item.get('text') or ''}"
        for item in utterances
    )
    return text


def build_messages(title, utterances, system_prompt):
    text = format_utterances(utterances)
    return [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"录音标题：{title}\n\n润色后的录音文本：\n{text}"}
    ]


def build_prompt(llm, title, utterances, system_prompt):
    return chat_prompt(llm, build_messages(title, utterances, system_prompt))


def build_refine_messages(title, chunk, total_chunks, memory, system_prompt):
    text = format_utterances(chunk.get("utterances") or [])
    time_range = f"{format_time(chunk.get('startMs'))}-{format_time(chunk.get('endMs'))}"
    system = (
        f"{system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}\n"
        "你正在做滚动记忆式长录音总结。必须基于当前片段和前文滚动记忆，不能编造。\n"
        "当前步骤只输出严格 JSON，不要 Markdown，不要解释。\n"
        "JSON schema: {\"chunkSummary\":\"当前片段的详细总结\",\"memory\":\"传给后续片段的滚动记忆\"}\n"
        "chunkSummary 按原文顺序概括当前内容的讨论重点，不要用“阶段/片段”作为标题，不要机械写成逐个 speaker 的发言记录。memory 保留后面还会用到的事实、结论和待办。\n"
    )
    user = (
        f"录音标题：{title}\n"
        f"当前片段：{chunk.get('index')}/{total_chunks}，时间范围：{time_range}\n\n"
        f"前文滚动记忆：\n{memory or '无'}\n\n"
        f"当前片段文本：\n{text}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_refine_prompt(llm, title, chunk, total_chunks, memory, system_prompt):
    return chat_prompt(llm, build_refine_messages(title, chunk, total_chunks, memory, system_prompt))


def build_final_messages(title, chunk_summaries, memory, system_prompt):
    summaries_text = "\n\n".join(
        f"片段 {item.get('index')} [{item.get('timeRange')}]\n{item.get('summary')}"
        for item in chunk_summaries
    )
    system = (
        f"{system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}\n"
        "这是长录音的最终综合总结。请基于片段总结和滚动记忆输出给用户看的最终中文总结。\n"
        "片段总结只是内部处理中间结果，最终输出不要出现“片段 1/片段 2/阶段一/阶段二”等内部编号或流程标签。\n"
        "开头先写一个全局总结，用 1-2 段话概括整段录音的核心内容、主要结论或整体结果。\n"
        "按照录音里的先后顺序总结，可以自然分成几段；每段标题要直接写真实主题，而不是写处理阶段。\n"
        "总结要比逐句复述更高一层，不要机械写成逐个 speaker 的发言记录。\n"
        "保留具体事实、数字、结论和待办；只有人物身份本身重要时才提到人。\n"
        "用自然的大白话写，不要写成报告腔，也不要只写空泛概括。\n"
        "不要输出 <think>，不要输出 JSON，不要编造。\n"
    )
    user = (
        f"录音标题：{title}\n\n"
        f"最终滚动记忆：\n{memory or '无'}\n\n"
        f"片段总结：\n{summaries_text}\n"
        "输出要求：输出最终总结，不要把长录音压缩成很短一段。\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_final_prompt(llm, title, chunk_summaries, memory, system_prompt):
    return chat_prompt(llm, build_final_messages(title, chunk_summaries, memory, system_prompt))


def strip_thinking(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(思考过程|推理过程|分析过程)[:：].*?(?=\n\n|$)", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def complete_text(llm, prompt, max_tokens):
    output = llm(prompt, max_tokens=int(max_tokens), temperature=0.1, stop=STOP_TOKENS)
    return strip_thinking(output["choices"][0]["text"].strip())


def truncate_tail(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def truncate_head(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[已截断]"


def fit_final_inputs(chunk_summaries, memory, context_size, max_tokens):
    budget = max(4000, int(context_size) - int(max_tokens) - 1600)
    memory_budget = min(len(memory), max(1200, budget // 3))
    fitted_memory = truncate_tail(memory, memory_budget)
    remaining = max(1200, budget - len(fitted_memory))
    per_chunk = max(400, remaining // max(1, len(chunk_summaries)))
    fitted_summaries = []
    for item in chunk_summaries:
        fitted_summaries.append({
            **item,
            "summary": truncate_head(str(item.get("summary") or ""), per_chunk)
        })
    return fitted_summaries, fitted_memory


def run_rolling_summary(llm, payload, system_prompt):
    chunks = payload.get("chunks") or []
    title = payload.get("title") or "未命名录音"
    max_tokens = int(payload.get("maxTokens") or 2500)
    chunk_max_tokens = int(payload.get("chunkMaxTokens") or 1200)
    memory_max_chars = int(payload.get("memoryMaxChars") or 6000)
    context_size = int(payload.get("contextSize") or 100000)
    memory = ""
    chunk_summaries = []

    for chunk in chunks:
        print(f"[summary] rolling chunk {chunk.get('index')}/{len(chunks)}", file=sys.stderr, flush=True)
        prompt = build_refine_prompt(llm, title, chunk, len(chunks), memory, system_prompt)
        raw = complete_text(llm, prompt, chunk_max_tokens)
        parsed = parse_json_object(raw)
        if parsed:
            chunk_summary = strip_thinking(str(parsed.get("chunkSummary") or ""))
            memory = strip_thinking(str(parsed.get("memory") or memory))
        else:
            chunk_summary = raw
            memory = f"{memory}\n{chunk_summary}".strip()
        memory = truncate_tail(memory, memory_max_chars)
        chunk_summaries.append({
            "index": chunk.get("index"),
            "timeRange": f"{format_time(chunk.get('startMs'))}-{format_time(chunk.get('endMs'))}",
            "summary": chunk_summary or "该片段无明显可总结内容。"
        })

    final_chunk_summaries, final_memory = fit_final_inputs(chunk_summaries, memory, context_size, max_tokens)
    final_prompt = build_final_prompt(llm, title, final_chunk_summaries, final_memory, system_prompt)
    return complete_text(llm, final_prompt, max_tokens)


def main():
    payload = json.load(sys.stdin)
    utterances = payload.get("utterances") or []
    chunks = payload.get("chunks") or []
    if not utterances and not chunks:
        print(json.dumps({"summaryText": "暂无可总结的润色文本。"}, ensure_ascii=False))
        return

    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is required for local recording summary") from exc

    model_path = payload["modelPath"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"local recording summary model file not found: {model_path}")
    print(f"[summary] loading local GGUF model: {model_path}", file=sys.stderr, flush=True)
    llm = Llama(model_path=model_path, n_ctx=int(payload.get("contextSize") or 100000), verbose=False)
    system_prompt = payload.get("systemPrompt") or DEFAULT_SYSTEM_PROMPT
    if payload.get("mode") == "rolling":
        text = run_rolling_summary(llm, payload, system_prompt)
    else:
        prompt = build_prompt(llm, payload.get("title") or "未命名录音", utterances, system_prompt)
        text = complete_text(llm, prompt, int(payload.get("maxTokens") or 2500))
    print(json.dumps({"summaryText": text or "暂无可总结的润色文本。"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
