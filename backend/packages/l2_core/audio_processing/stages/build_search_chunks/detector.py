from __future__ import annotations

import gc
import re
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from l2_core.audio_processing.stages.build_search_chunks.contracts import TopicSection, TopicSectionsOutput
from l2_core.audio_processing.stages.build_search_chunks.prompt import build_topic_boundary_prompt
from l2_core.audio_processing.stages.recording_models import Utterance


class LlamaModel(Protocol):
    def __call__(self, prompt: str, **kwargs: object) -> Mapping[str, object]: ...


class LlamaFactory(Protocol):
    def __call__(self, *, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool) -> LlamaModel: ...


class LlamaCppModule(Protocol):
    Llama: LlamaFactory


class TopicBoundaryDetector:
    """Detect continuous topic sections; invalid model output triggers deterministic fallback."""

    def __init__(self, model_path: Path, context_size: int, verbose: bool = False, max_batch_chars: int = 3500) -> None:
        self._model_path = model_path
        self._context_size = context_size
        self._verbose = verbose
        self._max_batch_chars = max_batch_chars
        self._model: LlamaModel | None = None

    def detect(self, utterances: Sequence[Utterance]) -> list[TopicSection]:
        if not utterances:
            return []
        sections: list[TopicSection] = []
        for batch in self._batches(utterances):
            sections.extend(self._detect_batch(batch))
        self._validate(sections, utterances)
        return sections

    def release(self) -> None:
        self._model = None
        gc.collect()

    def _detect_batch(self, utterances: Sequence[Utterance]) -> list[TopicSection]:
        response = self._load_model()(
            build_topic_boundary_prompt(utterances),
            max_tokens=max(256, min(2048, len(utterances) * 48)),
            temperature=0,
            stop=["<|im_end|>"],
            echo=False,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Topic detector returned no choices")
        raw = str(cast(Mapping[str, object], choices[0]).get("text") or "")
        parsed = TopicSectionsOutput.model_validate_json(self._json_object(raw))
        self._validate(parsed.sections, utterances)
        return parsed.sections

    def _load_model(self) -> LlamaModel:
        if self._model is not None:
            return self._model
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Topic detection model file not found: {self._model_path}")
        module = cast(LlamaCppModule, import_module("llama_cpp"))
        try:
            self._model = module.Llama(
                model_path=str(self._model_path),
                n_ctx=self._context_size,
                n_gpu_layers=-1,
                verbose=self._verbose,
            )
        except Exception:
            self._model = module.Llama(
                model_path=str(self._model_path),
                n_ctx=self._context_size,
                n_gpu_layers=0,
                verbose=self._verbose,
            )
        return self._model

    def _batches(self, utterances: Sequence[Utterance]) -> list[list[Utterance]]:
        batches: list[list[Utterance]] = []
        pending: list[Utterance] = []
        chars = 0
        for utterance in utterances:
            if pending and chars + len(utterance.text) > self._max_batch_chars:
                batches.append(pending)
                pending = []
                chars = 0
            pending.append(utterance)
            chars += len(utterance.text)
        if pending:
            batches.append(pending)
        return batches

    @staticmethod
    def _json_object(raw: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced is not None:
            return fenced.group(1)
        matched = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if matched is None:
            raise ValueError("Topic detector did not return a JSON object")
        return matched.group(0)

    @staticmethod
    def _validate(sections: Sequence[TopicSection], utterances: Sequence[Utterance]) -> None:
        expected = [item.utterance_index for item in utterances]
        covered: list[int] = []
        for section in sections:
            if not section.topic.strip() or section.end_utterance_index < section.start_utterance_index:
                raise ValueError("Invalid topic section")
            covered.extend(range(section.start_utterance_index, section.end_utterance_index + 1))
        if covered != expected:
            raise ValueError("Topic sections must cover every utterance exactly once")
