from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import tempfile
import warnings
import wave
from collections.abc import Callable, Iterable
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import WorkerClient
from l2_core.audio_processing.stages.recording_models import DiarizationOutput, DiarizationSegment, DiarizeInput
from l2_core.audio_processing.worker_tasks import audio_diarize_command

logger = logging.getLogger("audio_processing")


class PyannoteProgressHook:
    """Log pyannote's in-process hook at useful stage-level progress intervals."""

    step_ranges = {
        "segmentation": (45, 68),
        "speaker_counting": (68, 74),
        "embeddings": (74, 90),
        "discrete_diarization": (90, 94),
    }
    step_labels = {
        "segmentation": "语音分割",
        "speaker_counting": "估计说话人数",
        "embeddings": "提取说话人特征",
        "discrete_diarization": "生成说话人时间线",
    }

    def __init__(self, progress: Callable[[int, str], None]) -> None:
        self._last_percent = -1
        self._progress = progress

    def __call__(self, step_name: str, _artifact: object, file: object | None = None, total: int | None = None, completed: int | None = None) -> None:
        start, end = self.step_ranges.get(step_name, (45, 94))
        percent = end if not total or completed is None else int(start + max(0.0, min(1.0, completed / total)) * (end - start))
        if percent < 100 and percent // 5 == self._last_percent // 5:
            return
        self._last_percent = percent
        suffix = "" if not total or completed is None else f" {completed}/{total}"
        message = f"{self.step_labels.get(step_name, step_name)}{suffix}"
        logger.info("pyannote 进度 %d%%：%s", percent, message)
        self._progress(percent, message)


class PyannoteDiarizeStage:
    """Produce project-standard speaker segments from normalized audio."""

    name = "diarize_pyannote"
    version = "2"
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = DiarizeInput

    def __init__(
        self,
        storage_root: Path,
        artifact_store: ArtifactStore,
        model_name: str,
        auth_token: str | None,
        model_cache_dir: Path | None = None,
        use_local_config: bool = True,
        merge_max_gap_ms: int = 3_000,
        merge_max_duration_ms: int = 80_000,
        short_segment_absorb_max_duration_ms: int = 2_000,
        short_segment_absorb_max_gap_ms: int = 2_000,
        worker_client: WorkerClient | None = None,
    ) -> None:
        self._storage_root = storage_root.resolve()
        self._artifact_store = artifact_store
        self._model_name = model_name
        self._auth_token = auth_token
        self._model_cache_dir = model_cache_dir.resolve() if model_cache_dir is not None else None
        self._use_local_config = use_local_config
        self._merge_max_gap_ms = merge_max_gap_ms
        self._merge_max_duration_ms = merge_max_duration_ms
        self._short_segment_absorb_max_duration_ms = short_segment_absorb_max_duration_ms
        self._short_segment_absorb_max_gap_ms = short_segment_absorb_max_gap_ms
        self._worker_client = worker_client
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._temporary_config_path: Path | None = None

    async def try_restore(self, context: StageContext, _input_payload: DiarizeInput) -> StageResult[DiarizationOutput] | None:
        return self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "diarization.pyannote",
            DiarizationOutput,
        )

    async def run(self, context: StageContext, input_payload: DiarizeInput) -> StageResult[DiarizationOutput]:
        audio_descriptor = self._artifact_store.read_json(input_payload.audio)
        storage_path = audio_descriptor.get("storage_path")
        if not isinstance(storage_path, str):
            raise ValueError("Normalized audio artifact is missing storage_path")
        if self._worker_client is None:
            raise RuntimeError("PyannoteDiarizeStage requires WorkerClient")
        inference = await self._worker_client.execute(
            audio_diarize_command(storage_path),
            result_type=DiarizationOutput,
            on_progress=lambda progress, message: context.report_progress(round(progress * 100), message or "说话人分离"),
        )
        output = inference.model_copy(update={"segments": self._postprocess_segments(inference.segments)})
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="diarization.pyannote", data=output.model_dump(mode="json")),),
        )

    def compute(self, audio_path: Path, progress: Callable[[int, str], None]) -> DiarizationOutput:
        raw = self.infer(audio_path, progress)
        return raw.model_copy(update={"segments": self._postprocess_segments(raw.segments)})

    def infer(self, audio_path: Path, progress: Callable[[int, str], None]) -> DiarizationOutput:
        """Run only pyannote model inference and return unsmoothed speaker turns."""
        return DiarizationOutput(provider="pyannote", model_name=self._model_name, segments=self._run_pyannote(audio_path, progress))

    def _run_pyannote(self, audio_path: Path, progress: Callable[[int, str], None]) -> list[DiarizationSegment]:
        logger.info("pyannote：准备依赖和模型（%s）", self._model_name)
        progress(5, "加载 pyannote 依赖")
        self._configure_model_cache()
        try:
            pipeline_module = cast(Any, import_module("pyannote.audio"))
            torch = cast(Any, import_module("torch"))
        except ImportError as error:
            raise RuntimeError("pyannote.audio is not installed in the Python worker environment") from error
        progress(15, "加载 pyannote diarization pipeline")
        if self._pipeline is None:
            loaded_pipeline, self._temporary_config_path = self._load_pipeline(pipeline_module.Pipeline)
            device = self._resolve_device(torch)
            loaded_pipeline.to(device)
            self._pipeline = loaded_pipeline
            self._torch = torch
            logger.info("pyannote：模型加载完成，运行设备 %s", device)
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("pyannote pipeline was not initialized")
        device = self._resolve_device(torch)
        prediction: object | None = None
        audio: dict[str, object] | None = None
        try:
            logger.info("pyannote：运行设备 %s", device)
            progress(25, f"pyannote 运行设备 {device}")
            audio = self._load_audio_waveform(audio_path, torch, progress)
            audio = self._move_audio_to_device(audio, device)
            logger.info("pyannote：开始分离说话人")
            progress(45, "开始分离说话人")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly.*", category=UserWarning)
                prediction = pipeline(audio, hook=PyannoteProgressHook(progress))
            logger.info("pyannote：整理说话人分离结果")
            progress(95, "整理说话人分离结果")
            diarization = self._annotation_from_prediction(prediction)
            return self._to_segments(diarization)
        finally:
            del prediction
            del audio

    def release(self) -> None:
        self._pipeline = None
        torch = self._torch
        self._torch = None
        gc.collect()
        if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch is not None and getattr(torch, "backends", None) is not None and torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        if self._temporary_config_path is not None:
            self._temporary_config_path.unlink(missing_ok=True)
            self._temporary_config_path = None

    def _configure_model_cache(self) -> None:
        if self._model_cache_dir is None:
            return
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self._model_cache_dir.parent))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self._model_cache_dir))

    def _load_pipeline(self, pipeline_factory: Any) -> tuple[Any, Path | None]:
        if self._use_local_config:
            local_config = self._local_pipeline_config()
            if local_config is not None:
                temporary_config = self._patched_local_config(local_config)
                if temporary_config is not None:
                    logger.info("pyannote：使用本地模型快照")
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    return pipeline_factory.from_pretrained(str(temporary_config)), temporary_config
                logger.warning("pyannote 本地 pipeline 配置存在，但依赖快照不完整；将使用标准模型加载")
        if not self._auth_token:
            raise RuntimeError("PYANNOTE_AUTH_TOKEN is required when no complete local pyannote model snapshot is available")
        cache_dir = str(self._model_cache_dir) if self._model_cache_dir else None
        try:
            return pipeline_factory.from_pretrained(self._model_name, token=self._auth_token, cache_dir=cache_dir), None
        except TypeError:
            return pipeline_factory.from_pretrained(self._model_name, use_auth_token=self._auth_token, cache_dir=cache_dir), None

    def _local_pipeline_config(self) -> Path | None:
        if self._model_cache_dir is None:
            return None
        snapshots_dir = self._model_cache_dir / f"models--{self._model_name.replace('/', '--')}" / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
        return next((path / "config.yaml" for path in snapshots if (path / "config.yaml").is_file()), None)

    def _patched_local_config(self, config_path: Path) -> Path | None:
        if self._model_cache_dir is None:
            return None
        segmentation = self._local_snapshot("pyannote/segmentation-3.0", "pytorch_model.bin")
        embedding = self._local_snapshot("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin")
        community_xvec = self._local_snapshot("pyannote/speaker-diarization-community-1", "plda/xvec_transform.npz")
        community_plda = self._local_snapshot("pyannote/speaker-diarization-community-1", "plda/plda.npz")
        if not all((segmentation, embedding, community_xvec, community_plda)):
            return None
        config_text = config_path.read_text(encoding="utf-8")
        config_text = config_text.replace("segmentation: pyannote/segmentation-3.0", f"segmentation: {json.dumps(str(segmentation))}")
        config_text = config_text.replace("embedding: pyannote/wespeaker-voxceleb-resnet34-LM", f"embedding: {json.dumps(str(embedding))}")
        with tempfile.NamedTemporaryFile("w", suffix=".pyannote.config.yaml", delete=False, encoding="utf-8") as file:
            file.write(config_text)
            return Path(file.name)

    def _local_snapshot(self, model_name: str, required_file: str) -> Path | None:
        if self._model_cache_dir is None:
            return None
        snapshots_dir = self._model_cache_dir / f"models--{model_name.replace('/', '--')}" / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
        return next((path.resolve() for path in snapshots if (path / required_file).is_file()), None)

    @staticmethod
    def _resolve_device(torch: Any) -> Any:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _move_audio_to_device(audio: dict[str, object], device: Any) -> dict[str, object]:
        if device.type == "cpu":
            return audio
        return {**audio, "waveform": cast(Any, audio["waveform"]).to(device)}

    @staticmethod
    def _load_audio_waveform(audio_path: Path, torch: Any, progress: Callable[[int, str], None]) -> dict[str, object]:
        progress(30, "转换音频为 pyannote waveform")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            wav_path = Path(file.name)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            with wave.open(str(wav_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
            if sample_width != 2:
                raise RuntimeError(f"Expected 16-bit PCM WAV from ffmpeg, got sample width {sample_width}")
            numpy = cast(Any, import_module("numpy"))
            values = numpy.frombuffer(frames, dtype=numpy.int16).astype(numpy.float32) / 32768.0
            if channels > 1:
                values = values.reshape(-1, channels).mean(axis=1)
            progress(40, "音频 waveform 准备完成")
            return {"waveform": torch.from_numpy(values).unsqueeze(0), "sample_rate": sample_rate}
        finally:
            wav_path.unlink(missing_ok=True)

    @staticmethod
    def _annotation_from_prediction(prediction: object) -> object:
        for attribute in ("exclusive_speaker_diarization", "speaker_diarization"):
            annotation = getattr(prediction, attribute, None)
            if annotation is not None:
                return annotation
        if callable(getattr(prediction, "itertracks", None)):
            return prediction
        raise TypeError("pyannote prediction does not contain a speaker_diarization Annotation")

    def _to_segments(self, diarization: object) -> list[DiarizationSegment]:
        itertracks = getattr(diarization, "itertracks", None)
        if not callable(itertracks):
            raise TypeError("pyannote diarization result does not contain iterable tracks")
        label_by_cluster: dict[str, str] = {}
        segments: list[DiarizationSegment] = []
        tracks = cast(Iterable[tuple[PyannoteTurn, object, str]], itertracks(yield_label=True))
        for turn, _, cluster_id in tracks:
            cluster = str(cluster_id)
            label_by_cluster.setdefault(cluster, self._speaker_label(len(label_by_cluster)))
            start_ms = max(0, round(float(turn.start) * 1000))
            end_ms = max(start_ms, round(float(turn.end) * 1000))
            if end_ms > start_ms:
                segments.append(
                    DiarizationSegment(
                        id=f"{cluster}:{start_ms}:{end_ms}",
                        start_ms=start_ms,
                        end_ms=end_ms,
                        speaker_cluster_id=cluster,
                        speaker_label=label_by_cluster[cluster],
                    )
                )
        return segments

    def _postprocess_segments(self, segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
        merged, absorbed_count = self._smooth_segments(
            segments,
            self._short_segment_absorb_max_duration_ms,
            self._merge_max_gap_ms,
            self._merge_max_duration_ms,
            self._short_segment_absorb_max_gap_ms,
        )
        logger.info(
            "pyannote：完成，原始=%d 个说话人片段，夹心短段吸收=%d 个，合并后=%d 个",
            len(segments),
            absorbed_count,
            len(merged),
        )
        return merged

    @classmethod
    def _smooth_segments(
        cls,
        segments: list[DiarizationSegment],
        short_segment_max_duration_ms: int,
        merge_max_gap_ms: int,
        merge_max_duration_ms: int,
        short_segment_max_gap_ms: int | None = None,
    ) -> tuple[list[DiarizationSegment], int]:
        current = cls._merge_adjacent_segments(segments, merge_max_gap_ms, merge_max_duration_ms)
        absorb_max_gap_ms = merge_max_gap_ms if short_segment_max_gap_ms is None else short_segment_max_gap_ms
        absorbed_total = 0
        while True:
            smoothed, absorbed_count = cls._absorb_short_sandwiched_segments(
                current,
                short_segment_max_duration_ms,
                absorb_max_gap_ms,
            )
            if absorbed_count == 0:
                return current, absorbed_total
            absorbed_total += absorbed_count
            current = cls._merge_adjacent_segments(smoothed, merge_max_gap_ms, merge_max_duration_ms)

    @staticmethod
    def _absorb_short_sandwiched_segments(
        segments: list[DiarizationSegment],
        max_duration_ms: int,
        max_gap_ms: int,
    ) -> tuple[list[DiarizationSegment], int]:
        ordered = sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
        if max_duration_ms <= 0 or len(ordered) < 3:
            return ordered, 0

        smoothed = list(ordered)
        absorbed_count = 0
        for index in range(1, len(ordered) - 1):
            previous = smoothed[index - 1]
            segment = smoothed[index]
            following = smoothed[index + 1]
            if (
                segment.end_ms - segment.start_ms <= max_duration_ms
                and previous.speaker_cluster_id == following.speaker_cluster_id
                and segment.speaker_cluster_id != previous.speaker_cluster_id
                and segment.start_ms - previous.end_ms <= max_gap_ms
                and following.start_ms - segment.end_ms <= max_gap_ms
            ):
                smoothed[index] = segment.model_copy(
                    update={
                        "id": f"{previous.speaker_cluster_id}:{segment.start_ms}:{segment.end_ms}",
                        "speaker_cluster_id": previous.speaker_cluster_id,
                        "speaker_label": previous.speaker_label,
                    }
                )
                absorbed_count += 1
        return smoothed, absorbed_count

    @staticmethod
    def _merge_adjacent_segments(segments: list[DiarizationSegment], max_gap_ms: int, max_duration_ms: int) -> list[DiarizationSegment]:
        merged: list[DiarizationSegment] = []
        for segment in sorted(segments, key=lambda item: (item.start_ms, item.end_ms)):
            if not merged:
                merged.append(segment)
                continue
            previous = merged[-1]
            if (
                previous.speaker_cluster_id == segment.speaker_cluster_id
                and segment.start_ms - previous.end_ms <= max_gap_ms
                and segment.end_ms - previous.start_ms <= max_duration_ms
            ):
                merged[-1] = previous.model_copy(update={"end_ms": segment.end_ms, "id": f"{previous.speaker_cluster_id}:{previous.start_ms}:{segment.end_ms}"})
            else:
                merged.append(segment)
        return merged

    def _resolve_storage_path(self, uri: str) -> Path:
        path = (self._storage_root / uri).resolve()
        if self._storage_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Normalized audio does not exist: {uri}")
        return path

    @staticmethod
    def _speaker_label(index: int) -> str:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"Speaker {alphabet[index]}" if index < len(alphabet) else f"Speaker {index + 1}"


class PyannoteTurn(Protocol):
    start: float
    end: float
