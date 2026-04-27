import path from "node:path";
import { getAppConfig } from "../../../../config/app-config.ts";
import { type AudioProgressEvent, runPythonJson } from "../../../runtime/python-json.ts";
import { audioProcessEnv } from "../../../runtime/runtime-env.ts";
import { loadAsrHotwords } from "../../prompt-config.ts";
import type { Recording, TranscriptionOutput } from "../../../../types/models.ts";

export async function transcribeWithParaformer(recording: Recording, startedAt: number, onProgress?: (progress: AudioProgressEvent) => void): Promise<TranscriptionOutput> {
  const config = getAppConfig();
  const pythonBin = config.audio.paraformerPythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "transcription", "providers", "paraformer", "scripts", "run_paraformer.py");
  const audioPath = path.join(process.cwd(), recording.storagePath);
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "paraformer");
  const hotwords = await loadAsrHotwords(config.audio.paraformerHotwordConfigPath, config.audio.paraformerMaxHotwords);
  const args = [
    script,
    audioPath,
    "--model",
    config.audio.paraformerModel,
    "--model-revision",
    config.audio.paraformerModelRevision,
    "--cache-dir",
    cacheDir,
    "--vad-model",
    config.audio.paraformerVadModel,
    "--vad-model-revision",
    config.audio.paraformerVadModelRevision,
    "--punc-model",
    config.audio.paraformerPuncModel,
    "--punc-model-revision",
    config.audio.paraformerPuncModelRevision,
    "--vad-max-segment-ms",
    String(config.audio.paraformerVadMaxSegmentMs),
    "--merge-length-s",
    String(config.audio.paraformerMergeLengthSeconds)
  ];
  if (config.audio.paraformerMergeVad) args.push("--merge-vad");
  if (hotwords.length > 0) args.push("--hotword", hotwords.join(" "));

  console.log("[transcription] starting paraformer", {
    recordingId: recording.id,
    pythonBin,
    model: config.audio.paraformerModel,
    modelRevision: config.audio.paraformerModelRevision,
    vadModel: config.audio.paraformerVadModel,
    vadModelRevision: config.audio.paraformerVadModelRevision,
    puncModel: config.audio.paraformerPuncModel,
    puncModelRevision: config.audio.paraformerPuncModelRevision,
    hotwordCount: hotwords.length,
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
  console.log("[transcription] paraformer complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    segmentCount: output.segments.length,
    textLength: output.fullText.length
  });
  return output;
}
