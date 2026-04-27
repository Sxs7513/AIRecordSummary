#!/usr/bin/env python3
import contextlib
import json
import platform
import re
import sys
from pathlib import Path


def compact_items(items, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    seen = set()
    output = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def build_prompt(text: str, config: dict) -> str:
    llm_config = config.get("llmCorrection") or {}
    max_terms = int(llm_config.get("maxLlmTerms") or llm_config.get("maxProtectTerms") or 160)
    terms = compact_items(config.get("protectTerms"), max_terms)
    phrases = compact_items(config.get("llmPhase"), int(llm_config.get("maxPhrases") or 40))
    people = compact_items(config.get("people"), int(llm_config.get("maxPeople") or 40))
    system_items = llm_config.get("system")
    if not isinstance(system_items, list) or not system_items:
        system_items = [
            "你是半导体录音转写校对器。",
            "只允许修正常见 ASR 同音错词、专业术语错误、简繁体和标点断句。",
            "不要总结，不要扩写，不要改变原意。如果不确定，原样返回。",
            "只输出校对后的文本，不要解释，不要加引号。",
        ]
    system = "\n".join(str(item).strip() for item in system_items if str(item).strip())
    user_template = llm_config.get("userTemplate") or "专业词表：{terms}\n固定表达：{phrases}\n人名：{people}\n原始转写：{text}"
    context = str(user_template).format(
        terms="、".join(terms),
        phrases="；".join(phrases),
        people="、".join(people),
        text=text,
    )
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{context}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_output(output: str) -> str:
    cleaned = output.strip()
    cleaned = re.sub(r"^['\"“”]+|['\"“”]+$", "", cleaned).strip()
    return cleaned


def extract_json_object(output: str):
    cleaned = output.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def correct_one(llm, text: str, config: dict) -> str:
    if not text.strip():
        return text
    result = llm(
        build_prompt(text, config),
        max_tokens=max(64, min(512, len(text) * 3)),
        temperature=0,
        stop=["<|im_end|>"],
        echo=False,
    )
    choices = result.get("choices") or []
    output = choices[0].get("text", "") if choices else ""
    return clean_output(output) or text


def build_merge_prompt(candidate: dict, config: dict) -> str:
    llm_config = config.get("llmCorrection") or {}
    system_items = llm_config.get("mergeSystem")
    if not isinstance(system_items, list) or not system_items:
        system_items = [
            "你是会议转写文本整理助手。",
            "你只能在输入 segments 内做合并、断句、标点整理和少量 ASR 错词修正。",
            "不要新增信息，不要删除信息，不要改变说话人、时间顺序或 sourceIds。",
            "只允许合并相邻且语义连续的 segments；如果不确定就保持分开。",
            "必须输出严格 JSON，不要解释。",
        ]
    system = "\n".join(str(item).strip() for item in system_items if str(item).strip())
    segments = candidate.get("segments") or []
    input_json = json.dumps(
        {
            "groupId": candidate.get("groupId"),
            "speakerLabel": candidate.get("speakerLabel"),
            "segments": segments,
        },
        ensure_ascii=False,
    )
    default_template = (
        "输入是一组同一说话人的相邻 utterance segments。\n"
        "请判断是否需要在这些 segments 内做语义合并和断句。\n"
        "输出格式必须是：\n"
        "{\"groups\":[{\"sourceIds\":[\"segment id\"],\"text\":\"整理后的文本\"}]}\n"
        "规则：\n"
        "1. sourceIds 只能使用输入 segments 的 id。\n"
        "2. 不要遗漏、重复或改变 sourceIds 顺序。\n"
        "3. 每个 groups[i].sourceIds 必须是连续相邻的输入 id。\n"
        "4. 只合并语义上明显连续的相邻片段。\n"
        "5. text 只能整理对应 sourceIds 的原文，不能补充原文没有的信息。\n"
        "输入：{input}"
    )
    merge_user_template = str(llm_config.get("mergeUserTemplate") or default_template)
    instruction = merge_user_template.replace("{input}", input_json)
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def identity_merge(candidate: dict) -> dict:
    return {
        "groupId": candidate.get("groupId"),
        "groups": [
            {"sourceIds": [str(segment.get("id"))], "text": str(segment.get("text") or "")}
            for segment in candidate.get("segments") or []
            if segment.get("id")
        ],
    }


def validate_merge_result(candidate: dict, result: dict) -> dict:
    segments = candidate.get("segments") or []
    ordered_ids = [str(segment.get("id")) for segment in segments if segment.get("id")]
    positions = {source_id: index for index, source_id in enumerate(ordered_ids)}
    seen = []
    groups = []
    cursor = 0

    for group in result.get("groups") or []:
        source_ids = [str(source_id) for source_id in group.get("sourceIds") or []]
        text = str(group.get("text") or "").strip()
        if not source_ids or not text:
            raise ValueError("invalid empty merge group")
        indexes = [positions[source_id] for source_id in source_ids if source_id in positions]
        if len(indexes) != len(source_ids):
            raise ValueError("merge group references unknown source id")
        if indexes != list(range(indexes[0], indexes[-1] + 1)):
            raise ValueError("merge group source ids must be contiguous")
        if indexes[0] != cursor:
            raise ValueError("merge groups must preserve order and cover all segments")
        cursor = indexes[-1] + 1
        seen.extend(source_ids)
        groups.append({"sourceIds": source_ids, "text": text})

    if seen != ordered_ids:
        raise ValueError("merge result must cover every source id exactly once")
    return {"groupId": candidate.get("groupId"), "groups": groups}


def merge_one(llm, candidate: dict, config: dict) -> dict:
    if len(candidate.get("segments") or []) <= 1:
        return identity_merge(candidate)
    result = llm(
        build_merge_prompt(candidate, config),
        max_tokens=768,
        temperature=0,
        stop=["<|im_end|>"],
        echo=False,
    )
    choices = result.get("choices") or []
    output = choices[0].get("text", "") if choices else ""
    try:
        return validate_merge_result(candidate, extract_json_object(output))
    except Exception as exc:
        print(f"LLM merge output rejected for {candidate.get('groupId')}: {exc}", file=sys.stderr)
        return identity_merge(candidate)


def detect_gpu_layers(llama_supports_gpu_offload) -> tuple[int, str]:
    try:
        if llama_supports_gpu_offload():
            if platform.system() == "Darwin":
                return -1, "metal"
            return -1, "gpu"
    except Exception:
        pass
    return 0, "cpu"


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    mode = payload.get("mode") or "correctTexts"
    texts = payload.get("texts") or []
    candidates = payload.get("candidates") or []
    model_path = payload.get("modelPath")
    context_size = int(payload.get("contextSize") or 4096)
    config = payload.get("config") or {}
    if mode == "correctTexts" and not isinstance(texts, list):
        raise RuntimeError("texts must be a list")
    if mode == "mergeUtterances" and not isinstance(candidates, list):
        raise RuntimeError("candidates must be a list")
    if not model_path:
        raise RuntimeError("modelPath is required")
    if "," in str(model_path):
        model_path = str(model_path).split(",")[0].strip()
    if not Path(model_path).exists():
        raise RuntimeError(f"LLM model file not found: {model_path}. Run scripts/install_audio_dependencies.sh first.")

    with contextlib.redirect_stdout(sys.stderr):
        try:
            from llama_cpp import Llama
            from llama_cpp import llama_supports_gpu_offload
        except Exception as exc:
            raise RuntimeError("llama-cpp-python is not installed. Run scripts/install_audio_dependencies.sh first.") from exc

        n_gpu_layers, device = detect_gpu_layers(llama_supports_gpu_offload)
        print(f"local LLM correction device: {device}, n_gpu_layers: {n_gpu_layers}", file=sys.stderr, flush=True)
        llm = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=None,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    if mode == "mergeUtterances":
        results = [merge_one(llm, candidate, config) for candidate in candidates]
        print(json.dumps({"results": results}, ensure_ascii=False))
        return 0

    corrected = [correct_one(llm, str(text), config) for text in texts]
    print(json.dumps({"texts": corrected}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
