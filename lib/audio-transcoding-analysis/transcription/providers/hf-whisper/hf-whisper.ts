import path from "node:path";
import { getAppConfig } from "../../../../config/app-config.ts";
import { type AudioProgressEvent, runPythonJson } from "../../../runtime/python-json.ts";
import { audioProcessEnv } from "../../../runtime/runtime-env.ts";
import type { Recording, TranscriptionOutput } from "../../../../types/models.ts";

export async function transcribeWithHfWhisper(recording: Recording, startedAt: number, onProgress?: (progress: AudioProgressEvent) => void): Promise<TranscriptionOutput> {
  const config = getAppConfig();
  const pythonBin = config.audio.hfWhisperPythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "transcription", "providers", "hf-whisper", "scripts", "run_hf_whisper.py");
  const audioPath = path.join(process.cwd(), recording.storagePath);
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "hf-whisper");
  const args = [
    script,
    audioPath,
    "--model",
    config.audio.hfWhisperModel,
    "--cache-dir",
    cacheDir,
    "--language",
    config.audio.hfWhisperLanguage,
    "--chunk-length-s",
    String(config.audio.hfWhisperChunkLengthSeconds),
    "--batch-size",
    String(config.audio.hfWhisperBatchSize),
    "--vad-model",
    config.audio.hfWhisperVadModel,
    "--vad-max-segment-ms",
    String(config.audio.hfWhisperVadMaxSegmentMs),
    "--merge-length-s",
    String(config.audio.hfWhisperMergeLengthSeconds)
  ];
  if (config.audio.hfWhisperMergeVad) args.push("--merge-vad");

  console.log("[transcription] starting hf whisper", {
    recordingId: recording.id,
    pythonBin,
    model: config.audio.hfWhisperModel,
    language: config.audio.hfWhisperLanguage,
    chunkLengthSeconds: config.audio.hfWhisperChunkLengthSeconds,
    batchSize: config.audio.hfWhisperBatchSize,
    vadModel: config.audio.hfWhisperVadModel,
    cacheDir,
    audioPath
  });
  const output = await runPythonJson<TranscriptionOutput>({
    pythonBin,
    args,
    env: audioProcessEnv(pythonBin, config.audio.modelCacheRoot),
    logPrefix: "[transcription]",
    onProgress
  });
  console.log("[transcription] hf whisper complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    segmentCount: output.segments.length,
    textLength: output.fullText.length
  });
  return output;
}
