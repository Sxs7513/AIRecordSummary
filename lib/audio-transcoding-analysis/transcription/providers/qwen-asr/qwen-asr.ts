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

function positiveIntegerOrDefault(value: unknown, fallback: number) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return fallback;
  return Math.floor(number);
}

export async function transcribeWithQwenAsr(
  recording: Recording,
  startedAt: number,
  onProgress?: (progress: AudioProgressEvent) => void
): Promise<TranscriptionOutput> {
  const config = getAppConfig();
  const pythonBin = config.audio.qwenAsrPythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "transcription", "providers", "qwen-asr", "scripts", "run_qwen_asr.py");
  const audioPath = path.join(process.cwd(), recording.storagePath);
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "qwen-asr");
  const pyannoteCacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "huggingface", "hub");
  const context = await loadQwenAsrContext(config.audio.qwenAsrContextConfigPath, config.audio.qwenAsrMaxContextItems, config.audio.qwenAsrContext);
  const segmentTimeoutSeconds = positiveIntegerOrDefault(config.audio.qwenAsrSegmentTimeoutSeconds, 180);
  const args = [
    script,
    audioPath,
    "--model",
    config.audio.qwenAsrModel,
    "--cache-dir",
    cacheDir,
    "--pyannote-cache-dir",
    pyannoteCacheDir,
    "--language",
    config.audio.qwenAsrLanguage,
    "--max-new-tokens",
    String(config.audio.qwenAsrMaxNewTokens),
    "--max-inference-batch-size",
    String(config.audio.qwenAsrMaxInferenceBatchSize),
    "--segment-timeout-s",
    String(segmentTimeoutSeconds),
    "--low-volume-rms-threshold",
    String(config.audio.qwenAsrLowVolumeRmsThreshold),
    "--low-volume-peak-threshold",
    String(config.audio.qwenAsrLowVolumePeakThreshold),
    "--speaker-segment-merge-max-gap-ms",
    String(config.audio.qwenAsrSpeakerSegmentMergeMaxGapMs),
    "--speaker-segment-merge-max-duration-ms",
    String(config.audio.qwenAsrSpeakerSegmentMergeMaxDurationMs),
    "--speaker-segment-min-duration-ms",
    String(config.audio.qwenAsrSpeakerSegmentMinDurationMs),
    "--vad-model",
    config.audio.qwenAsrVadModel,
    "--vad-max-segment-ms",
    String(config.audio.qwenAsrVadMaxSegmentMs),
    "--vad-merge-max-gap-ms",
    String(config.audio.qwenAsrVadMergeMaxGapMs),
    "--vad-min-segment-ms",
    String(config.audio.qwenAsrVadMinSegmentMs),
    "--merge-length-s",
    String(config.audio.qwenAsrMergeLengthSeconds)
  ];
  if (config.audio.pyannoteAuthToken) args.push("--pyannote-auth-token", config.audio.pyannoteAuthToken);
  if (!config.audio.pyannoteUseLocalConfig) args.push("--no-pyannote-local-config");
  if (config.audio.qwenAsrUseOwnSegments) args.push("--forced-aligner-model", config.audio.qwenAsrForcedAlignerModel);
  if (context) args.push("--context", context);
  if (config.audio.qwenAsrUseOwnSegments) args.push("--use-own-segments");
  if (config.audio.qwenAsrEnhanceLowVolumeSegments) args.push("--enhance-low-volume-segments");
  if (config.audio.qwenAsrMergeVad) args.push("--merge-vad");
  if (config.audio.qwenAsrStripTrailingPunctuation) args.push("--strip-trailing-punctuation");
  if (config.audio.qwenAsrBreakOnSentenceEnd) args.push("--break-on-sentence-end");
  const env = qwenAsrProcessEnv(
    pythonBin,
    config.audio.modelCacheRoot,
    config.audio.qwenAsrUseOwnSegments ? [config.audio.qwenAsrModel, config.audio.qwenAsrForcedAlignerModel, "pyannote/speaker-diarization-3.1"] : [config.audio.qwenAsrModel, "pyannote/speaker-diarization-3.1"]
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
    segmentTimeoutSeconds,
    enhanceLowVolumeSegments: config.audio.qwenAsrEnhanceLowVolumeSegments,
    lowVolumeRmsThreshold: config.audio.qwenAsrLowVolumeRmsThreshold,
    lowVolumePeakThreshold: config.audio.qwenAsrLowVolumePeakThreshold,
    hasPyannoteAuthToken: Boolean(config.audio.pyannoteAuthToken),
    pyannoteUseLocalConfig: config.audio.pyannoteUseLocalConfig,
    pyannoteCacheDir,
    forcedAlignerModel: config.audio.qwenAsrForcedAlignerModel,
    vadModel: config.audio.qwenAsrVadModel,
    vadMergeMaxGapMs: config.audio.qwenAsrVadMergeMaxGapMs,
    vadMinSegmentMs: config.audio.qwenAsrVadMinSegmentMs,
    speakerSegmentMinDurationMs: config.audio.qwenAsrSpeakerSegmentMinDurationMs,
    speakerSegmentMergeMaxGapMs: config.audio.qwenAsrSpeakerSegmentMergeMaxGapMs,
    speakerSegmentMergeMaxDurationMs: config.audio.qwenAsrSpeakerSegmentMergeMaxDurationMs,
    stripTrailingPunctuation: config.audio.qwenAsrStripTrailingPunctuation,
    breakOnSentenceEnd: config.audio.qwenAsrBreakOnSentenceEnd,
    offlineMode: env.HF_HUB_OFFLINE === "1",
    cacheDir,
    audioPath
  });
  const output = await runPythonJson<TranscriptionOutput>({
    pythonBin,
    args,
    env: {
      ...env,
      PYANNOTE_AUTH_TOKEN: config.audio.pyannoteAuthToken
    },
    logPrefix: "[transcription]",
    onProgress
  });
  console.log("[transcription] qwen asr complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    segmentCount: output.segments.length,
    diarizationSegmentCount: output.diarization?.segments.length ?? 0,
    textLength: output.fullText.length
  });
  return output;
}
