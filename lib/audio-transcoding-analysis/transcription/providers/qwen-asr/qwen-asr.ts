import path from "node:path";
import { existsSync, readdirSync, statSync } from "node:fs";
import { getAppConfig } from "../../../../config/app-config.ts";
import { type AudioProgressEvent, runPythonJson } from "../../../runtime/python-json.ts";
import { audioProcessEnv } from "../../../runtime/runtime-env.ts";
import { loadQwenAsrContext } from "../../prompt-config.ts";
import type { Recording, TranscriptionOutput } from "../../../../types/models.ts";

function getHuggingFaceCacheRepoDir(modelName: string, modelCacheRoot: string): string {
  return path.join(process.cwd(), modelCacheRoot, "huggingface", "hub", `models--${modelName.replace("/", "--")}`);
}

function hasUsableHuggingFaceSnapshot(modelName: string, modelCacheRoot: string): boolean {
  const snapshotsDir = path.join(getHuggingFaceCacheRepoDir(modelName, modelCacheRoot), "snapshots");
  if (!existsSync(snapshotsDir)) return false;

  for (const snapshot of readdirSync(snapshotsDir)) {
    const snapshotDir = path.join(snapshotsDir, snapshot);
    if (!statSync(snapshotDir).isDirectory()) continue;
    const hasModelConfig = existsSync(path.join(snapshotDir, "config.json"));
    const hasFeatureExtractor = existsSync(path.join(snapshotDir, "preprocessor_config.json"));
    const hasTokenizerConfig = existsSync(path.join(snapshotDir, "tokenizer_config.json"));
    const hasTokenizer = ["tokenizer.json", "tokenizer.model", "vocab.json"].some((fileName) =>
      existsSync(path.join(snapshotDir, fileName))
    );
    if (hasModelConfig && hasFeatureExtractor && hasTokenizerConfig && hasTokenizer) return true;
  }
  return false;
}

function qwenAsrProcessEnv(pythonBin: string, modelCacheRoot: string, models: string[]): NodeJS.ProcessEnv {
  const env = audioProcessEnv(pythonBin, modelCacheRoot);
  const canUseOfflineMode = models.filter(Boolean).every((model) => hasUsableHuggingFaceSnapshot(model, modelCacheRoot));
  if (!canUseOfflineMode) return env;

  return {
    ...env,
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    HF_DATASETS_OFFLINE: "1"
  };
}

export async function transcribeWithQwenAsr(recording: Recording, startedAt: number, onProgress?: (progress: AudioProgressEvent) => void): Promise<TranscriptionOutput> {
  const config = getAppConfig();
  const pythonBin = config.audio.qwenAsrPythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "transcription", "providers", "qwen-asr", "scripts", "run_qwen_asr.py");
  const audioPath = path.join(process.cwd(), recording.storagePath);
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "qwen-asr");
  const context = await loadQwenAsrContext(config.audio.qwenAsrContextConfigPath, config.audio.qwenAsrMaxContextItems, config.audio.qwenAsrContext);
  const args = [
    script,
    audioPath,
    "--model",
    config.audio.qwenAsrModel,
    "--cache-dir",
    cacheDir,
    "--language",
    config.audio.qwenAsrLanguage,
    "--max-new-tokens",
    String(config.audio.qwenAsrMaxNewTokens),
    "--max-inference-batch-size",
    String(config.audio.qwenAsrMaxInferenceBatchSize),
    "--vad-model",
    config.audio.qwenAsrVadModel,
    "--vad-max-segment-ms",
    String(config.audio.qwenAsrVadMaxSegmentMs),
    "--merge-length-s",
    String(config.audio.qwenAsrMergeLengthSeconds)
  ];
  if (config.audio.qwenAsrUseOwnSegments) args.push("--forced-aligner-model", config.audio.qwenAsrForcedAlignerModel);
  if (context) args.push("--context", context);
  if (config.audio.qwenAsrUseOwnSegments) args.push("--use-own-segments");
  if (config.audio.qwenAsrMergeVad) args.push("--merge-vad");
  const env = qwenAsrProcessEnv(
    pythonBin,
    config.audio.modelCacheRoot,
    config.audio.qwenAsrUseOwnSegments ? [config.audio.qwenAsrModel, config.audio.qwenAsrForcedAlignerModel] : [config.audio.qwenAsrModel]
  );

  console.log("[transcription] starting qwen asr", {
    recordingId: recording.id,
    pythonBin,
    model: config.audio.qwenAsrModel,
    language: config.audio.qwenAsrLanguage,
    useOwnSegments: config.audio.qwenAsrUseOwnSegments,
    contextConfigPath: config.audio.qwenAsrContextConfigPath,
    contextLength: context?.length ?? 0,
    maxNewTokens: config.audio.qwenAsrMaxNewTokens,
    maxInferenceBatchSize: config.audio.qwenAsrMaxInferenceBatchSize,
    forcedAlignerModel: config.audio.qwenAsrForcedAlignerModel,
    vadModel: config.audio.qwenAsrVadModel,
    offlineMode: env.HF_HUB_OFFLINE === "1",
    cacheDir,
    audioPath
  });
  const output = await runPythonJson<TranscriptionOutput>({
    pythonBin,
    args,
    env,
    logPrefix: "[transcription]",
    onProgress
  });
  console.log("[transcription] qwen asr complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    segmentCount: output.segments.length,
    textLength: output.fullText.length
  });
  return output;
}
