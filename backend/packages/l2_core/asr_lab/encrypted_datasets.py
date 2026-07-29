from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_MAGIC = b"ASRPACK1"
_SALT_SIZE = 16
_NONCE_SIZE = 12
_MAX_PACKAGE_BYTES = 512 * 1024 * 1024
_MAX_SAMPLE_BYTES = 100 * 1024 * 1024
_MAX_SAMPLES = 10_000


@dataclass(frozen=True)
class EncryptedDatasetSample:
    sample_id: str
    text: str
    audio: bytes


class EncryptedDatasetStore:
    """Password-encrypted, repository-local ASR training samples."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def list_packages(self) -> list[dict[str, Any]]:
        if not self._root.is_dir():
            return []
        return [
            {
                "id": path.stem,
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size,
                "updated_at": path.stat().st_mtime,
            }
            for path in sorted(self._root.glob("*.asrpack"), key=lambda item: item.stat().st_mtime, reverse=True)
            if path.is_file()
        ]

    def verify_password(self, package_id: str, password: str) -> None:
        package_path = self._package_path(package_id)
        self._validate_password(password)
        if package_path.is_file():
            self.load(package_id, password)

    def persist_sample(
        self,
        *,
        package_id: str,
        sample_id: str,
        password: str,
        source: Path,
        start_ms: int,
        end_ms: int,
        text: str,
    ) -> Path:
        return self.persist_audio_sample(
            package_id=package_id,
            sample_id=sample_id,
            password=password,
            audio=self._crop_audio(source, start_ms, end_ms),
            text=text,
        )

    def persist_audio_sample(
        self,
        *,
        package_id: str,
        sample_id: str,
        password: str,
        audio: bytes,
        text: str,
    ) -> Path:
        self._validate_password(password)
        if not audio or len(audio) > _MAX_SAMPLE_BYTES:
            raise ValueError("音频切片为空或过大")
        package_path = self._package_path(package_id)
        samples = {sample.sample_id: sample for sample in self.load(package_id, password)} if package_path.is_file() else {}
        samples[sample_id] = EncryptedDatasetSample(
            sample_id=sample_id,
            text=text,
            audio=audio,
        )
        payload = self._build_archive(list(samples.values()))
        encrypted = self._encrypt(payload, password)
        self._root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self._root, prefix=f".{package_id}-", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encrypted)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            temporary_path.replace(package_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return package_path

    def load(self, package_id: str, password: str) -> list[EncryptedDatasetSample]:
        self._validate_password(password)
        package_path = self._package_path(package_id)
        if not package_path.is_file():
            return []
        if package_path.stat().st_size > _MAX_PACKAGE_BYTES:
            raise ValueError("加密数据集文件过大")
        archive = self._decrypt(package_path.read_bytes(), password)
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                manifest_info = bundle.getinfo("dataset.json")
                if manifest_info.file_size > 10 * 1024 * 1024:
                    raise ValueError("加密数据集清单过大")
                decoded_manifest: object = json.loads(bundle.read(manifest_info))
                if not isinstance(decoded_manifest, dict):
                    raise ValueError("不支持的加密数据集格式")
                manifest = cast(dict[str, object], decoded_manifest)
                raw_samples = manifest.get("samples")
                if manifest.get("version") != 1 or not isinstance(raw_samples, list):
                    raise ValueError("不支持的加密数据集格式")
                sample_values = cast(list[object], raw_samples)
                if len(sample_values) > _MAX_SAMPLES:
                    raise ValueError("加密数据集样本数量超过限制")
                samples: list[EncryptedDatasetSample] = []
                for raw_sample_value in sample_values:
                    if not isinstance(raw_sample_value, dict):
                        raise ValueError("加密数据集样本格式错误")
                    raw_sample = cast(dict[str, object], raw_sample_value)
                    sample_id = raw_sample.get("id")
                    text = raw_sample.get("text")
                    audio_name = raw_sample.get("audio")
                    if not isinstance(sample_id, str) or not sample_id:
                        raise ValueError("加密数据集样本缺少必要内容")
                    if not isinstance(text, str) or not text:
                        raise ValueError("加密数据集样本缺少必要内容")
                    if not isinstance(audio_name, str) or not audio_name:
                        raise ValueError("加密数据集样本缺少必要内容")
                    if audio_name != f"audio/{sample_id}.flac":
                        raise ValueError("加密数据集音频路径不合法")
                    audio_info = bundle.getinfo(audio_name)
                    if audio_info.file_size > _MAX_SAMPLE_BYTES:
                        raise ValueError("加密数据集中的音频切片过大")
                    samples.append(EncryptedDatasetSample(sample_id=sample_id, text=text, audio=bundle.read(audio_info)))
                return samples
        except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
            raise ValueError("加密数据集文件已损坏") from error

    def _package_path(self, package_id: str) -> Path:
        if not package_id or any(character not in "0123456789abcdef-" for character in package_id.lower()):
            raise ValueError("加密数据集 ID 不合法")
        path = (self._root / f"{package_id}.asrpack").resolve()
        if path.parent != self._root:
            raise ValueError("加密数据集路径不合法")
        return path

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password) < 8:
            raise ValueError("加密密码至少需要 8 个字符")

    @staticmethod
    def _derive_key(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)

    @classmethod
    def _encrypt(cls, payload: bytes, password: str) -> bytes:
        salt = os.urandom(_SALT_SIZE)
        nonce = os.urandom(_NONCE_SIZE)
        ciphertext = AESGCM(cls._derive_key(password, salt)).encrypt(nonce, payload, _MAGIC)
        return _MAGIC + salt + nonce + ciphertext

    @classmethod
    def _decrypt(cls, package: bytes, password: str) -> bytes:
        header_size = len(_MAGIC) + _SALT_SIZE + _NONCE_SIZE
        if len(package) <= header_size or not package.startswith(_MAGIC):
            raise ValueError("不支持的加密数据集格式")
        salt_start = len(_MAGIC)
        nonce_start = salt_start + _SALT_SIZE
        salt = package[salt_start:nonce_start]
        nonce = package[nonce_start:header_size]
        try:
            return AESGCM(cls._derive_key(password, salt)).decrypt(nonce, package[header_size:], _MAGIC)
        except InvalidTag as error:
            raise ValueError("密码错误或加密数据集已损坏") from error

    @staticmethod
    def _build_archive(samples: list[EncryptedDatasetSample]) -> bytes:
        target = io.BytesIO()
        with zipfile.ZipFile(target, "w") as bundle:
            manifest = {
                "version": 1,
                "samples": [{"id": sample.sample_id, "audio": f"audio/{sample.sample_id}.flac", "text": sample.text} for sample in samples],
            }
            bundle.writestr("dataset.json", json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), compress_type=zipfile.ZIP_DEFLATED)
            for sample in samples:
                bundle.writestr(f"audio/{sample.sample_id}.flac", sample.audio, compress_type=zipfile.ZIP_STORED)
        return target.getvalue()

    @staticmethod
    def _crop_audio(source: Path, start_ms: int, end_ms: int) -> bytes:
        return crop_audio_to_flac(source, start_ms, end_ms)


def crop_audio_to_flac(source: Path, start_ms: int, end_ms: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="asr-lab-slice-") as temporary_directory:
        target = Path(temporary_directory) / "sample.flac"
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-ss",
                f"{start_ms / 1000:.3f}",
                "-to",
                f"{end_ms / 1000:.3f}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "flac",
                str(target),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"音频区间切片失败: {detail}")
        if target.stat().st_size > _MAX_SAMPLE_BYTES:
            raise ValueError("音频切片过大")
        return target.read_bytes()
