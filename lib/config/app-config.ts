import { getDatabaseConfig, type DatabaseConfig } from "./database.ts";

export interface AppConfig {
  database: DatabaseConfig;
  storage: {
    localRoot: string;
    publicFileBaseUrl: string;
  };
  audio: {
    autoInstallDependencies: boolean;
    asrProvider: "whisper" | "sensevoice" | "paraformer" | "hf_whisper" | "qwen_asr";
    whisperPythonBin: string;
    whisperModel: string;
    whisperLanguage: string;
    whisperInitialPromptConfigPath: string;
    hfWhisperPythonBin: string;
    hfWhisperModel: string;
    hfWhisperLanguage: string;
    hfWhisperChunkLengthSeconds: number;
    hfWhisperBatchSize: number;
    hfWhisperVadModel: string;
    hfWhisperVadMaxSegmentMs: number;
    hfWhisperMergeVad: boolean;
    hfWhisperMergeLengthSeconds: number;
    qwenAsrPythonBin: string;
    qwenAsrModel: string;
    qwenAsrLanguage: string;
    qwenAsrForcedAlignerModel: string;
    qwenAsrUseOwnSegments: boolean;
    qwenAsrContextConfigPath: string;
    qwenAsrContext: string;
    qwenAsrMaxContextItems: number;
    qwenAsrMaxNewTokens: number;
    qwenAsrMaxInferenceBatchSize: number;
    qwenAsrVadModel: string;
    qwenAsrVadMaxSegmentMs: number;
    qwenAsrMergeVad: boolean;
    qwenAsrMergeLengthSeconds: number;
    senseVoicePythonBin: string;
    senseVoiceModel: string;
    senseVoiceLanguage: string;
    senseVoiceUseItn: boolean;
    senseVoiceVadModel: string;
    senseVoiceVadMaxSegmentMs: number;
    senseVoiceMergeVad: boolean;
    senseVoiceMergeLengthSeconds: number;
    paraformerPythonBin: string;
    paraformerModel: string;
    paraformerModelRevision: string;
    paraformerVadModel: string;
    paraformerVadModelRevision: string;
    paraformerPuncModel: string;
    paraformerPuncModelRevision: string;
    paraformerHotwordConfigPath: string;
    paraformerMaxHotwords: number;
    paraformerVadMaxSegmentMs: number;
    paraformerMergeVad: boolean;
    paraformerMergeLengthSeconds: number;
    transcriptionCorrectionEnabled: boolean;
    pyannotePythonBin: string;
    pyannoteAuthToken: string;
    speechbrainPythonBin: string;
    modelCacheRoot: string;
    targetSpeakerThreshold: number;
    utteranceMaxDurationMs: number;
    utteranceMaxTextChars: number;
    utteranceLlmMergeMaxGapMs: number;
    utteranceLlmMergeMaxDurationMs: number;
    utteranceLlmMergeMaxTextChars: number;
    llmCorrectionEnabled: boolean;
    llmCorrectionPythonBin: string;
    llmCorrectionModelRepo: string;
    llmCorrectionModelFile: string;
    llmCorrectionContextSize: number;
    llmCorrectionTimeoutMs: number;
  };
  jobs: {
    embeddedWorkerEnabled: boolean;
    workerConcurrency: number;
    intervalMs: number;
    batchSize: number;
  };
  search: {
    embeddingEnabled: boolean;
    embeddingProvider: "local_qwen3";
    embeddingModel: string;
    embeddingDimensions: number;
    embeddingPythonBin: string;
    embeddingDevice: "auto" | "cuda" | "mps" | "cpu";
    embeddingBatchSize: number;
    embeddingModelCacheDir: string;
    vectorTopK: number;
    finalTopK: number;
    minScore: number;
    chunkMaxDurationMs: number;
    chunkMaxTextChars: number;
    chunkMaxGapMs: number;
    answerEnabled: boolean;
    answerProvider: "local_llm" | "deepseek_api" | "extractive";
    answerModelRepo: string;
    answerModelFile: string;
    answerContextSize: number;
    answerTimeoutMs: number;
    deepseekApiKey: string;
    deepseekBaseUrl: string;
    deepseekModel: string;
  };
}

declare global {
  // eslint-disable-next-line no-var
  var __aiRecordSummaryConfig: AppConfig | undefined;
}

export function buildAppConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  return {
    database: getDatabaseConfig(env),
    storage: {
      localRoot: env.LOCAL_STORAGE_ROOT || "uploads",
      publicFileBaseUrl: env.PUBLIC_FILE_BASE_URL || ""
    },
    audio: {
      autoInstallDependencies: env.AUDIO_DEPS_AUTO_INSTALL !== "false",
      asrProvider:
        env.ASR_PROVIDER === "sensevoice"
          ? "sensevoice"
          : env.ASR_PROVIDER === "paraformer"
            ? "paraformer"
            : env.ASR_PROVIDER === "hf_whisper"
              ? "hf_whisper"
              : env.ASR_PROVIDER === "qwen_asr"
                ? "qwen_asr"
                : "whisper",
      whisperPythonBin: env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      whisperModel: env.WHISPER_MODEL || "base",
      whisperLanguage: env.WHISPER_LANGUAGE || "",
      whisperInitialPromptConfigPath: env.WHISPER_INITIAL_PROMPT_CONFIG || "config/initial-prompt.json",
      hfWhisperPythonBin: env.HF_WHISPER_PYTHON_BIN || env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      hfWhisperModel: env.HF_WHISPER_MODEL || "BELLE-2/Belle-whisper-large-v3-turbo-zh",
      hfWhisperLanguage: env.HF_WHISPER_LANGUAGE || "zh",
      hfWhisperChunkLengthSeconds: Number(env.HF_WHISPER_CHUNK_LENGTH_S || 30),
      hfWhisperBatchSize: Number(env.HF_WHISPER_BATCH_SIZE || 4),
      hfWhisperVadModel: env.HF_WHISPER_VAD_MODEL || "fsmn-vad",
      hfWhisperVadMaxSegmentMs: Number(env.HF_WHISPER_VAD_MAX_SEGMENT_MS || 30000),
      hfWhisperMergeVad: env.HF_WHISPER_MERGE_VAD !== "false",
      hfWhisperMergeLengthSeconds: Number(env.HF_WHISPER_MERGE_LENGTH_S || 15),
      qwenAsrPythonBin: env.QWEN_ASR_PYTHON_BIN || env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      qwenAsrModel: env.QWEN_ASR_MODEL || "Qwen/Qwen3-ASR-1.7B",
      qwenAsrLanguage: env.QWEN_ASR_LANGUAGE || "auto",
      qwenAsrForcedAlignerModel: env.QWEN_ASR_FORCED_ALIGNER_MODEL || "Qwen/Qwen3-ForcedAligner-0.6B",
      qwenAsrUseOwnSegments: env.QWEN_ASR_USE_OWN_SEGMENTS === "true",
      qwenAsrContextConfigPath: env.QWEN_ASR_CONTEXT_CONFIG || env.WHISPER_INITIAL_PROMPT_CONFIG || "config/initial-prompt.json",
      qwenAsrContext: env.QWEN_ASR_CONTEXT || "",
      qwenAsrMaxContextItems: Number(env.QWEN_ASR_MAX_CONTEXT_ITEMS || 80),
      qwenAsrMaxNewTokens: Number(env.QWEN_ASR_MAX_NEW_TOKENS || 4096),
      qwenAsrMaxInferenceBatchSize: Number(env.QWEN_ASR_MAX_INFERENCE_BATCH_SIZE || 4),
      qwenAsrVadModel: env.QWEN_ASR_VAD_MODEL || "fsmn-vad",
      qwenAsrVadMaxSegmentMs: Number(env.QWEN_ASR_VAD_MAX_SEGMENT_MS || 30000),
      qwenAsrMergeVad: env.QWEN_ASR_MERGE_VAD !== "false",
      qwenAsrMergeLengthSeconds: Number(env.QWEN_ASR_MERGE_LENGTH_S || 15),
      senseVoicePythonBin: env.SENSEVOICE_PYTHON_BIN || env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      senseVoiceModel: env.SENSEVOICE_MODEL || "FunAudioLLM/SenseVoiceSmall",
      senseVoiceLanguage: env.SENSEVOICE_LANGUAGE || "auto",
      senseVoiceUseItn: env.SENSEVOICE_USE_ITN !== "false",
      senseVoiceVadModel: env.SENSEVOICE_VAD_MODEL || "fsmn-vad",
      senseVoiceVadMaxSegmentMs: Number(env.SENSEVOICE_VAD_MAX_SEGMENT_MS || 30000),
      senseVoiceMergeVad: env.SENSEVOICE_MERGE_VAD !== "false",
      senseVoiceMergeLengthSeconds: Number(env.SENSEVOICE_MERGE_LENGTH_S || 15),
      paraformerPythonBin: env.PARAFORMER_PYTHON_BIN || env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      paraformerModel: env.PARAFORMER_MODEL || "paraformer-zh",
      paraformerModelRevision: env.PARAFORMER_MODEL_REVISION || "v2.0.4",
      paraformerVadModel: env.PARAFORMER_VAD_MODEL || "fsmn-vad",
      paraformerVadModelRevision: env.PARAFORMER_VAD_MODEL_REVISION || "v2.0.4",
      paraformerPuncModel: env.PARAFORMER_PUNC_MODEL || "ct-punc-c",
      paraformerPuncModelRevision: env.PARAFORMER_PUNC_MODEL_REVISION || "v2.0.4",
      paraformerHotwordConfigPath: env.PARAFORMER_HOTWORD_CONFIG || env.WHISPER_INITIAL_PROMPT_CONFIG || "config/initial-prompt.json",
      paraformerMaxHotwords: Number(env.PARAFORMER_MAX_HOTWORDS || 80),
      paraformerVadMaxSegmentMs: Number(env.PARAFORMER_VAD_MAX_SEGMENT_MS || 30000),
      paraformerMergeVad: env.PARAFORMER_MERGE_VAD !== "false",
      paraformerMergeLengthSeconds: Number(env.PARAFORMER_MERGE_LENGTH_S || 15),
      transcriptionCorrectionEnabled: env.TRANSCRIPTION_CORRECTION_ENABLED !== "false",
      pyannotePythonBin: env.PYANNOTE_PYTHON_BIN || ".venv-audio/bin/python",
      pyannoteAuthToken: env.PYANNOTE_AUTH_TOKEN || "",
      speechbrainPythonBin: env.SPEECHBRAIN_PYTHON_BIN || ".venv-audio/bin/python",
      modelCacheRoot: env.AUDIO_MODEL_CACHE_ROOT || "model-cache",
      targetSpeakerThreshold: Number(env.TARGET_SPEAKER_THRESHOLD || 0.7),
      utteranceMaxDurationMs: Number(env.UTTERANCE_MAX_DURATION_MS || 30000),
      utteranceMaxTextChars: Number(env.UTTERANCE_MAX_TEXT_CHARS || 500),
      utteranceLlmMergeMaxGapMs: Number(env.UTTERANCE_LLM_MERGE_MAX_GAP_MS || 1200),
      utteranceLlmMergeMaxDurationMs: Number(env.UTTERANCE_LLM_MERGE_MAX_DURATION_MS || 20000),
      utteranceLlmMergeMaxTextChars: Number(env.UTTERANCE_LLM_MERGE_MAX_TEXT_CHARS || 300),
      llmCorrectionEnabled: env.LLM_CORRECTION_ENABLED === "true",
      llmCorrectionPythonBin: env.LLM_CORRECTION_PYTHON_BIN || env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      llmCorrectionModelRepo: env.LLM_CORRECTION_MODEL_REPO || "Qwen/Qwen2.5-7B-Instruct-GGUF",
      llmCorrectionModelFile: env.LLM_CORRECTION_MODEL_FILE || "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf,qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
      llmCorrectionContextSize: Number(env.LLM_CORRECTION_CONTEXT_SIZE || 8192),
      llmCorrectionTimeoutMs: Number(env.LLM_CORRECTION_TIMEOUT_MS || 600000)
    },
    jobs: {
      embeddedWorkerEnabled: env.EMBEDDED_WORKER_ENABLED !== "false",
      workerConcurrency: Math.max(1, Number(env.EMBEDDED_WORKER_CONCURRENCY || 2)),
      intervalMs: Number(env.EMBEDDED_WORKER_INTERVAL_MS || 5000),
      batchSize: Math.max(1, Number(env.EMBEDDED_WORKER_BATCH_SIZE || 25))
    },
    search: {
      embeddingEnabled: env.EMBEDDING_ENABLED !== "false",
      embeddingProvider: "local_qwen3",
      embeddingModel: env.EMBEDDING_MODEL || "Qwen/Qwen3-Embedding-0.6B",
      embeddingDimensions: Number(env.EMBEDDING_DIMENSIONS || 1024),
      embeddingPythonBin: env.EMBEDDING_PYTHON_BIN || env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      embeddingDevice:
        env.EMBEDDING_DEVICE === "cuda" || env.EMBEDDING_DEVICE === "mps" || env.EMBEDDING_DEVICE === "cpu"
          ? env.EMBEDDING_DEVICE
          : "auto",
      embeddingBatchSize: Math.max(1, Number(env.EMBEDDING_BATCH_SIZE || 16)),
      embeddingModelCacheDir: env.EMBEDDING_MODEL_CACHE_DIR || "model-cache/embedding",
      vectorTopK: Math.max(1, Number(env.SEARCH_VECTOR_TOP_K || 30)),
      finalTopK: Math.max(1, Number(env.SEARCH_FINAL_TOP_K || 10)),
      minScore: Number(env.SEARCH_MIN_SCORE || 0.25),
      chunkMaxDurationMs: Number(env.SEARCH_CHUNK_MAX_DURATION_MS || 60000),
      chunkMaxTextChars: Number(env.SEARCH_CHUNK_MAX_TEXT_CHARS || 800),
      chunkMaxGapMs: Number(env.SEARCH_CHUNK_MAX_GAP_MS || 3000),
      answerEnabled: env.RAG_ANSWER_ENABLED !== "false",
      answerProvider:
        env.RAG_ANSWER_PROVIDER === "deepseek_api"
          ? "deepseek_api"
          : env.RAG_ANSWER_PROVIDER === "extractive"
            ? "extractive"
            : "local_llm",
      answerModelRepo: env.RAG_ANSWER_MODEL_REPO || "Qwen/Qwen3-8B-GGUF",
      answerModelFile: env.RAG_ANSWER_MODEL_FILE || "Qwen3-8B-Q4_K_M.gguf",
      answerContextSize: Number(env.RAG_ANSWER_CONTEXT_SIZE || 8192),
      answerTimeoutMs: Number(env.RAG_ANSWER_TIMEOUT_MS || 600000),
      deepseekApiKey: env.DEEPSEEK_API_KEY || "",
      deepseekBaseUrl: env.DEEPSEEK_BASE_URL || "https://api.deepseek.com",
      deepseekModel: env.DEEPSEEK_MODEL || "deepseek-chat"
    }
  };
}

export function setGlobalAppConfig(config = buildAppConfig()): AppConfig {
  globalThis.__aiRecordSummaryConfig = config;
  process.env.AI_RECORD_SUMMARY_CONFIG = JSON.stringify(config);
  return config;
}

export function getAppConfig(): AppConfig {
  return globalThis.__aiRecordSummaryConfig ?? setGlobalAppConfig();
}
