import re

STOP_TOKENS = ["</s>", "<|im_end|>"]


def chat_prompt(llm, messages):
    template = getattr(llm, "metadata", {}).get("tokenizer.chat_template")
    if template:
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            eos_token_id = llm.token_eos()
            bos_token_id = llm.token_bos()
            eos_token = llm._model.token_get_text(eos_token_id) if eos_token_id != -1 else "<|im_end|>"
            bos_token = llm._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""
            formatter = Jinja2ChatFormatter(
                template=template,
                eos_token=eos_token,
                bos_token=bos_token,
                stop_token_ids=[eos_token_id] if eos_token_id != -1 else None,
            )
            return formatter(messages=messages, enable_thinking=False).prompt
        except Exception:
            pass

    return "".join(
        f"<|im_start|>{message['role']}\n{message['content'].strip()}\n<|im_end|>\n"
        for message in messages
    ) + "<|im_start|>assistant\n"


def messages_from_chatml(prompt):
    messages = []
    for role, content in re.findall(r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|$)", prompt, flags=re.DOTALL):
        content = content.strip()
        if role == "assistant" and not content:
            continue
        messages.append({"role": role, "content": content})
    if messages:
        return messages
    return [{"role": "user", "content": prompt.strip()}]
