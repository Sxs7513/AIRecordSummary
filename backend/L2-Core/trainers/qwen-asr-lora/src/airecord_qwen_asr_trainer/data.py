from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class QwenAsrDataCollator:
    """Build native Qwen3-ASR supervised inputs and let the processor mask labels."""

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        conversations: list[list[dict[str, object]]] = []
        for feature in features:
            audio_path = str(feature["audio"])
            target = str(feature["text"])
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "audio", "path": audio_path}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target}],
                    },
                ]
            )
        return self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"output_labels": True},
        )
