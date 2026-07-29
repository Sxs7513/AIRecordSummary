import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from l2_core.asr_lab.encrypted_datasets import EncryptedDatasetStore, crop_audio_to_flac


def _fake_crop(_source: Path, _start: int, _end: int) -> bytes:
    return b"fake-flac"


def test_persist_and_load_encrypted_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = EncryptedDatasetStore(tmp_path)
    monkeypatch.setattr(EncryptedDatasetStore, "_crop_audio", staticmethod(_fake_crop))

    package = store.persist_sample(
        package_id="11111111-1111-1111-1111-111111111111",
        sample_id="sample-1",
        password="correct-password",
        source=tmp_path / "source.wav",
        start_ms=1_000,
        end_ms=2_000,
        text="人工校验文本",
    )

    encrypted = package.read_bytes()
    assert b"fake-flac" not in encrypted
    assert "人工校验文本".encode() not in encrypted
    assert store.list_packages()[0]["id"] == "11111111-1111-1111-1111-111111111111"

    samples = store.load("11111111-1111-1111-1111-111111111111", "correct-password")
    assert [(sample.sample_id, sample.text, sample.audio) for sample in samples] == [
        ("sample-1", "人工校验文本", b"fake-flac")
    ]


def test_wrong_password_cannot_decrypt_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = EncryptedDatasetStore(tmp_path)
    monkeypatch.setattr(EncryptedDatasetStore, "_crop_audio", staticmethod(_fake_crop))
    store.persist_sample(
        package_id="22222222-2222-2222-2222-222222222222",
        sample_id="sample-1",
        password="correct-password",
        source=tmp_path / "source.wav",
        start_ms=0,
        end_ms=1_000,
        text="reference",
    )

    with pytest.raises(ValueError, match="密码错误"):
        store.load("22222222-2222-2222-2222-222222222222", "wrong-password")


def test_persist_existing_audio_slice_without_source_recording(tmp_path: Path) -> None:
    store = EncryptedDatasetStore(tmp_path)
    store.persist_audio_sample(
        package_id="44444444-4444-4444-4444-444444444444",
        sample_id="sample-1",
        password="correct-password",
        audio=b"standalone-flac",
        text="独立切片文本",
    )

    samples = store.load("44444444-4444-4444-4444-444444444444", "correct-password")
    assert len(samples) == 1
    assert samples[0].audio == b"standalone-flac"
    assert samples[0].text == "独立切片文本"


def test_cropped_flac_contains_parseable_duration(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 32_000)

    target = tmp_path / "sample.flac"
    target.write_bytes(crop_audio_to_flac(source, 500, 1_500))
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert 0.9 <= float(result.stdout.strip()) <= 1.1


def test_persisting_same_sample_replaces_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = EncryptedDatasetStore(tmp_path)
    cropped_audio = iter((b"first-audio", b"second-audio"))
    def next_audio(_source: Path, _start: int, _end: int) -> bytes:
        return next(cropped_audio)

    monkeypatch.setattr(EncryptedDatasetStore, "_crop_audio", staticmethod(next_audio))
    package_id = "33333333-3333-3333-3333-333333333333"
    password = "correct-password"
    store.persist_sample(
        package_id=package_id,
        sample_id="sample-1",
        password=password,
        source=tmp_path / "source.wav",
        start_ms=0,
        end_ms=1_000,
        text="first",
    )
    store.persist_sample(
        package_id=package_id,
        sample_id="sample-1",
        password=password,
        source=tmp_path / "source.wav",
        start_ms=0,
        end_ms=1_000,
        text="second",
    )

    samples = store.load(package_id, password)
    assert len(samples) == 1
    assert samples[0].text == "second"
    assert samples[0].audio == b"second-audio"
