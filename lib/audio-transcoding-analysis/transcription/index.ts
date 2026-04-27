import { getAppConfig } from "../../config/app-config.ts";
import type { AudioProgressEvent } from "../runtime/python-json.ts";
import type { Recording, TranscriptionOutput } from "../../types/models.ts";
import { transcribeWithParaformer } from "./providers/paraformer/paraformer.ts";
import { transcribeWithHfWhisper } from "./providers/hf-whisper/hf-whisper.ts";
import { transcribeWithQwenAsr } from "./providers/qwen-asr/qwen-asr.ts";
import { transcribeWithSenseVoice } from "./providers/sensevoice/sensevoice.ts";
import { transcribeWithWhisper } from "./providers/whisper/whisper.ts";

export async function transcribeRecording(recording: Recording, onProgress?: (progress: AudioProgressEvent) => void): Promise<TranscriptionOutput> {
  const startedAt = Date.now();
  const config = getAppConfig();
  if (config.audio.asrProvider === "sensevoice") {
    return transcribeWithSenseVoice(recording, startedAt, onProgress);
  }
  if (config.audio.asrProvider === "paraformer") {
    return transcribeWithParaformer(recording, startedAt, onProgress);
  }
  if (config.audio.asrProvider === "hf_whisper") {
    return transcribeWithHfWhisper(recording, startedAt, onProgress);
  }
  if (config.audio.asrProvider === "qwen_asr") {
    return transcribeWithQwenAsr(recording, startedAt, onProgress);
  }
  return transcribeWithWhisper(recording, startedAt, onProgress);
}
