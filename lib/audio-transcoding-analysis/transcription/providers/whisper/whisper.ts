import path from "node:path";
import { getAppConfig } from "../../../../config/app-config.ts";
import { type AudioProgressEvent, runPythonJson } from "../../../runtime/python-json.ts";
import { audioProcessEnv } from "../../../runtime/runtime-env.ts";
import { loadWhisperInitialPrompt } from "../../prompt-config.ts";
import type { Recording, TranscriptionOutput } from "../../../../types/models.ts";

export async function transcribeWithWhisper(recording: Recording, startedAt: number, onProgress?: (progress: AudioProgressEvent) => void): Promise<TranscriptionOutput> {
  const config = getAppConfig();
  const pythonBin = config.audio.whisperPythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "transcription", "providers", "whisper", "scripts", "run_whisper.py");
  const audioPath = path.join(process.cwd(), recording.storagePath);
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "whisper");
  const initialPrompt = await loadWhisperInitialPrompt(config.audio.whisperInitialPromptConfigPath);
  const args = [script, audioPath, "--model", config.audio.whisperModel, "--cache-dir", cacheDir];
  if (config.audio.whisperLanguage) args.push("--language", config.audio.whisperLanguage);
  if (initialPrompt) args.push("--initial-prompt", initialPrompt);

  console.log("[transcription] starting whisper", {
    recordingId: recording.id,
    pythonBin,
    model: config.audio.whisperModel,
    language: config.audio.whisperLanguage || "auto",
    hasInitialPrompt: Boolean(initialPrompt),
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
  console.log("[transcription] whisper complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    segmentCount: output.segments.length,
    textLength: output.fullText.length
  });
  return output;
}
