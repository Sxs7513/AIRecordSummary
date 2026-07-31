from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker import ComputeCommand


class AudioDiarizeTaskInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_storage_path: str = Field(min_length=1)


class AsrInferenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str = Field(min_length=1)
    audio_storage_path: str = Field(min_length=1)


class AsrInferenceBatchInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[AsrInferenceItem] = Field(min_length=1)


class AsrInferenceItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    text: str
    language: str | None = None


class AsrInferenceBatchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model_name: str
    items: list[AsrInferenceItemResult]


class AlignmentInferenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str = Field(min_length=1)
    audio_storage_path: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)


class AlignmentInferenceBatchInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[AlignmentInferenceItem] = Field(min_length=1)


class AlignmentTokenResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    start_time: float
    end_time: float


class AlignmentInferenceItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    tokens: list[AlignmentTokenResult]


class AlignmentInferenceBatchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    items: list[AlignmentInferenceItemResult]


class EmbeddingEncodeTaskInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    texts: list[str] = Field(min_length=1)


class EmbeddingEncodeTaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model_name: str
    dimensions: int = Field(gt=0)
    vectors: list[list[float]]


def audio_diarize_command(audio_storage_path: str) -> ComputeCommand[AudioDiarizeTaskInput]:
    return ComputeCommand(
        task_id=uuid4(),
        operation="diarization.pyannote.infer",
        operation_version="1",
        resource_queue=ResourceQueue.GPU_HIGH,
        input=AudioDiarizeTaskInput(audio_storage_path=audio_storage_path),
    )


def asr_inference_batch_command(
    provider: str,
    audio_storage_paths: Sequence[str],
) -> ComputeCommand[AsrInferenceBatchInput]:
    if provider not in {"qwen_asr", "funasr_nano"}:
        raise ValueError(f"Unsupported ASR provider: {provider}")
    return ComputeCommand(
        task_id=uuid4(),
        operation=f"asr.{provider}.infer_batch",
        operation_version="1",
        resource_queue=ResourceQueue.GPU_HIGH,
        input=AsrInferenceBatchInput(items=[AsrInferenceItem(item_id=str(index), audio_storage_path=path) for index, path in enumerate(audio_storage_paths)]),
    )


def alignment_inference_batch_command(
    items: Sequence[AlignmentInferenceItem],
) -> ComputeCommand[AlignmentInferenceBatchInput]:
    return ComputeCommand(
        task_id=uuid4(),
        operation="alignment.qwen.infer_batch",
        operation_version="1",
        resource_queue=ResourceQueue.GPU_HIGH,
        input=AlignmentInferenceBatchInput(items=list(items)),
    )


def embedding_encode_command(texts: Sequence[str]) -> ComputeCommand[EmbeddingEncodeTaskInput]:
    return ComputeCommand(
        task_id=uuid4(),
        operation="embedding.encode",
        operation_version="1",
        resource_queue=ResourceQueue.GPU_NORMAL,
        input=EmbeddingEncodeTaskInput(texts=list(texts)),
    )
