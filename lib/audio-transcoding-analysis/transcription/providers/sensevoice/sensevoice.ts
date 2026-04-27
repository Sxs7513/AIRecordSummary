import path from "node:path";
import { getAppConfig } from "../../../../config/app-config.ts";
import { type AudioProgressEvent, runPythonJson } from "../../../runtime/python-json.ts";
import { audioProcessEnv } from "../../../runtime/runtime-env.ts";
import type { Recording, TranscriptionOutput } from "../../../../types/models.ts";

export async function transcribeWithSenseVoice(recording: Recording, startedAt: number, onProgress?: (progress: AudioProgressEvent) => void): Promise<TranscriptionOutput> {
  const config = getAppConfig();
  const pythonBin = config.audio.senseVoicePythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "transcription", "providers", "sensevoice", "scripts", "run_sensevoice.py");
  const audioPath = path.join(process.cwd(), recording.storagePath);
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "sensevoice");
  const args = [
    script,
    audioPath,
    "--model",
    config.audio.senseVoiceModel,
    "--cache-dir",
    cacheDir,
    "--language",
    config.audio.senseVoiceLanguage,
    "--vad-model",
    config.audio.senseVoiceVadModel,
    "--vad-max-segment-ms",
    String(config.audio.senseVoiceVadMaxSegmentMs),
    "--merge-length-s",
    String(config.audio.senseVoiceMergeLengthSeconds)
  ];
  if (config.audio.senseVoiceUseItn) args.push("--use-itn");
  if (config.audio.senseVoiceMergeVad) args.push("--merge-vad");

  console.log("[transcription] starting sensevoice", {
    recordingId: recording.id,
    pythonBin,
    model: config.audio.senseVoiceModel,
    language: config.audio.senseVoiceLanguage,
    vadModel: config.audio.senseVoiceVadModel,
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
  console.log("[transcription] sensevoice complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    segmentCount: output.segments.length,
    textLength: output.fullText.length
  });
  return output;
}
