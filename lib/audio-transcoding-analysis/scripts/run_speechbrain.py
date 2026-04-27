#!/usr/bin/env python3
import contextlib
import json
import sys
import tempfile
from pathlib import Path


def crop_wav_with_ffmpeg(source: str, start_ms: int, end_ms: int) -> str:
    import subprocess

    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-ss",
            str(start_ms / 1000),
            "-to",
            str(end_ms / 1000),
            "-ar",
            "16000",
            "-ac",
            "1",
            output.name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return output.name


def convert_wav_with_ffmpeg(source: str) -> str:
    import subprocess

    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-ar",
            "16000",
            "-ac",
            "1",
            output.name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return output.name


def concat_wavs_with_ffmpeg(wav_paths: list[str]) -> str:
    import subprocess

    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output.close()
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        for wav_path in wav_paths:
            escaped = wav_path.replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_file.close()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file.name,
                "-ar",
                "16000",
                "-ac",
                "1",
                output.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return output.name
    finally:
        Path(list_file.name).unlink(missing_ok=True)


def load_payload() -> dict:
    if len(sys.argv) >= 3 and sys.argv[1] == "--payload-file":
        return json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if len(sys.argv) >= 2:
        return json.loads(sys.argv[1])
    raise RuntimeError("Missing speechbrain payload.")


def speaker_key(segment: dict) -> str:
    return segment.get("speakerClusterId") or segment.get("speakerLabel") or "unknown"


def create_speaker_wav(
    recording_path: str,
    segments: list[dict],
    min_segment_ms: int,
    min_speaker_ms: int,
    max_speaker_ms: int,
):
    clip_paths = []
    selected_duration_ms = 0
    try:
        for segment in sorted(segments, key=lambda item: int(item["startMs"])):
            start_ms = int(segment["startMs"])
            end_ms = int(segment["endMs"])
            duration_ms = end_ms - start_ms
            if duration_ms < min_segment_ms:
                continue
            if selected_duration_ms >= max_speaker_ms:
                break

            remaining_ms = max_speaker_ms - selected_duration_ms
            cropped_end_ms = start_ms + min(duration_ms, remaining_ms)
            clip_paths.append(crop_wav_with_ffmpeg(recording_path, start_ms, cropped_end_ms))
            selected_duration_ms += cropped_end_ms - start_ms

        if selected_duration_ms < min_speaker_ms or not clip_paths:
            return None, clip_paths, selected_duration_ms
        if len(clip_paths) == 1:
            return clip_paths[0], [], selected_duration_ms
        return concat_wavs_with_ffmpeg(clip_paths), clip_paths, selected_duration_ms
    except Exception:
        for clip_path in clip_paths:
            Path(clip_path).unlink(missing_ok=True)
        raise


def main() -> int:
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except Exception as exc:
            raise RuntimeError("speechbrain is not installed. Run scripts/install_audio_dependencies.sh first.") from exc

    payload = load_payload()
    cache_dir = payload.get("cacheDir") or "model-cache/speechbrain/spkrec-ecapa-voxceleb"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=cache_dir)

    threshold = float(payload.get("threshold", 0.7))
    min_segment_ms = int(payload.get("minSegmentMs", 1000))
    min_speaker_ms = int(payload.get("minSpeakerMs", 3000))
    max_speaker_ms = int(payload.get("maxSpeakerMs", 30000))
    prepared_profiles = []
    temp_sample_paths = []
    temp_speaker_paths = []
    matches = []

    try:
        for profile in payload["profiles"]:
            sample_paths = []
            for sample_path in profile["samplePaths"]:
                try:
                    wav_path = convert_wav_with_ffmpeg(sample_path)
                    sample_paths.append(wav_path)
                    temp_sample_paths.append(wav_path)
                except Exception as exc:
                    print(f"Skipping speaker sample that cannot be converted: {sample_path}: {exc}", file=sys.stderr)
            if sample_paths:
                prepared_profiles.append({**profile, "samplePaths": sample_paths})

        grouped_segments = {}
        for segment in payload["diarizationSegments"]:
            grouped_segments.setdefault(speaker_key(segment), []).append(segment)

        cluster_results = {}
        for cluster_key, segments in grouped_segments.items():
            if not prepared_profiles:
                cluster_results[cluster_key] = (False, None, None)
                continue

            try:
                speaker_wav_path, cleanup_paths, selected_duration_ms = create_speaker_wav(
                    payload["recordingPath"],
                    segments,
                    min_segment_ms,
                    min_speaker_ms,
                    max_speaker_ms,
                )
                temp_speaker_paths.extend(cleanup_paths)
                if speaker_wav_path:
                    temp_speaker_paths.append(speaker_wav_path)
            except Exception as exc:
                print(f"Skipping speaker cluster that cannot be converted: {cluster_key}: {exc}", file=sys.stderr)
                cluster_results[cluster_key] = (False, None, None)
                continue

            if not speaker_wav_path:
                print(
                    f"Skipping speaker cluster with insufficient usable audio: {cluster_key}, selected_duration_ms={selected_duration_ms}",
                    file=sys.stderr,
                )
                cluster_results[cluster_key] = (False, None, None)
                continue

            best_profile_id = None
            best_score = None
            for profile in prepared_profiles:
                for sample_path in profile["samplePaths"]:
                    try:
                        with contextlib.redirect_stdout(sys.stderr):
                            score, _prediction = verifier.verify_files(sample_path, speaker_wav_path)
                        numeric_score = float(score.squeeze().item())
                        if best_score is None or numeric_score > best_score:
                            best_score = numeric_score
                            best_profile_id = profile["id"]
                    except Exception as exc:
                        print(
                            f"Skipping failed speaker comparison: speaker={cluster_key} sample={sample_path}: {exc}",
                            file=sys.stderr,
                        )

            is_target = best_score is not None and best_score >= threshold
            cluster_results[cluster_key] = (
                is_target,
                best_score,
                best_profile_id if is_target else None,
            )

        for segment in payload["diarizationSegments"]:
            is_target, confidence, profile_id = cluster_results.get(speaker_key(segment), (False, None, None))
            matches.append(
                {
                    "diarizationSegmentId": segment["id"],
                    "isTargetPerson": is_target,
                    "confidence": confidence,
                    "speakerProfileId": profile_id,
                }
            )
    finally:
        for sample_path in temp_sample_paths:
            Path(sample_path).unlink(missing_ok=True)
        for speaker_path in temp_speaker_paths:
            Path(speaker_path).unlink(missing_ok=True)

    print(json.dumps(matches, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
