import { getDatabaseConfig, type DatabaseConfig } from "./database.ts";

export interface AppConfig {
  // PostgreSQL 连接配置。
  database: DatabaseConfig;
  storage: {
    // 本地音频文件保存根目录。
    localRoot: string;
    // 对外访问本地文件时使用的 URL 前缀；为空时使用应用内 /uploads 路由。
    publicFileBaseUrl: string;
  };
  audio: {
    // 应用启动时是否运行音频依赖安装脚本。
    installDependenciesOnStartup: boolean;
    // 启动时是否自动检查/安装音频处理依赖。
    autoInstallDependencies: boolean;
    // 当前使用的 ASR 转写提供方。
    asrProvider: "whisper" | "sensevoice" | "paraformer" | "hf_whisper" | "qwen_asr";
    // Whisper 使用的 Python 解释器路径。
    whisperPythonBin: string;
    // Whisper 模型名称。
    whisperModel: string;
    // Whisper 识别语言；为空表示自动识别。
    whisperLanguage: string;
    // 转写上下文、热词、纠错和总结 prompt 的主配置文件路径。
    whisperInitialPromptConfigPath: string;
    // HuggingFace Whisper 使用的 Python 解释器路径。
    hfWhisperPythonBin: string;
    // HuggingFace Whisper 模型名称。
    hfWhisperModel: string;
    // HuggingFace Whisper 识别语言。
    hfWhisperLanguage: string;
    // HuggingFace Whisper 单次推理音频块长度，单位秒。
    hfWhisperChunkLengthSeconds: number;
    // HuggingFace Whisper 推理批大小。
    hfWhisperBatchSize: number;
    // HuggingFace Whisper 前置 VAD 模型。
    hfWhisperVadModel: string;
    // HuggingFace Whisper VAD 片段最大时长，单位毫秒。
    hfWhisperVadMaxSegmentMs: number;
    // HuggingFace Whisper 是否合并相邻 VAD 片段。
    hfWhisperMergeVad: boolean;
    // HuggingFace Whisper 合并后单段最长时长，单位秒。
    hfWhisperMergeLengthSeconds: number;
    // Qwen ASR 使用的 Python 解释器路径。
    qwenAsrPythonBin: string;
    // Qwen ASR 模型名称。
    qwenAsrModel: string;
    // Qwen ASR 识别语言；auto 表示自动识别。
    qwenAsrLanguage: string;
    // Qwen 自带分段模式使用的 forced aligner 模型。
    qwenAsrForcedAlignerModel: string;
    // 是否使用 Qwen 自带 timestamp/forced aligner 分段；关闭时走 VAD 切片。
    qwenAsrUseOwnSegments: boolean;
    // Qwen ASR 上下文配置文件路径，默认复用 initial-prompt。
    qwenAsrContextConfigPath: string;
    // 额外注入给 Qwen ASR 的上下文文本。
    qwenAsrContext: string;
    // 从上下文配置里最多取多少条术语/短语/人名提示。
    qwenAsrMaxContextItems: number;
    // Qwen ASR 单次生成最大 token 数。
    qwenAsrMaxNewTokens: number;
    // Qwen ASR 最大推理 batch size。
    qwenAsrMaxInferenceBatchSize: number;
    // Qwen ASR 单个片段转写超时时间，单位秒；用于防止个别片段生成跑飞卡死。
    qwenAsrSegmentTimeoutSeconds: number;
    // Qwen ASR 是否对低音量片段使用增强音频进行转写。
    qwenAsrEnhanceLowVolumeSegments: boolean;
    // Qwen ASR 片段 RMS 低于该阈值时认为音量偏低。
    qwenAsrLowVolumeRmsThreshold: number;
    // Qwen ASR 片段峰值低于该阈值时认为音量偏低。
    qwenAsrLowVolumePeakThreshold: number;
    // Qwen ASR VAD 切片模型。
    qwenAsrVadModel: string;
    // VAD 合并/切分后，送给 Qwen ASR 的单个音频片段最长时长。
    // 片段越长，上下文越完整，但推理更慢、资源占用更高。
    qwenAsrVadMaxSegmentMs: number;
    // 相邻 VAD 片段之间允许合并的最大静音间隔。
    // 这是实际决定短暂停顿是否断句的阈值。
    qwenAsrVadMergeMaxGapMs: number;
    // 小于该时长的 VAD 短片段会尽量并入相邻片段。
    // 用于减少 VAD 抖动导致的碎片。
    qwenAsrVadMinSegmentMs: number;
    // 是否在送给 Qwen ASR 前先合并相邻 VAD 片段。
    qwenAsrMergeVad: boolean;
    // 合并后的单个 VAD 音频片段最长时长，单位秒。
    // 这是片段长度上限，不是停顿阈值，也不是断句阈值。
    qwenAsrMergeLengthSeconds: number;
    // 清理 Qwen ASR 容易在每个片段末尾补上的标点。
    // 避免后续连续发言合并时把每个片段都当作完整句子。
    qwenAsrStripTrailingPunctuation: boolean;
    // 仅用于 Qwen timestamp 模式。开启后，句尾标点可触发 timestamp 片段断开。
    // 如果 ASR 标点过于激进，应该保持关闭。
    qwenAsrBreakOnSentenceEnd: boolean;
    // Qwen ASR 使用 pyannote 说话人片段时，相同说话人片段允许合并的最大间隔，单位毫秒。
    qwenAsrSpeakerSegmentMergeMaxGapMs: number;
    // Qwen ASR 使用 pyannote 说话人片段时，相同说话人合并后的单段最长时长，单位毫秒。
    qwenAsrSpeakerSegmentMergeMaxDurationMs: number;
    // Qwen ASR 使用 pyannote 说话人片段时，小于该时长的短片段会优先吸收到相邻片段，单位毫秒。
    qwenAsrSpeakerSegmentMinDurationMs: number;
    // SenseVoice 使用的 Python 解释器路径。
    senseVoicePythonBin: string;
    // SenseVoice 模型名称。
    senseVoiceModel: string;
    // SenseVoice 识别语言；auto 表示自动识别。
    senseVoiceLanguage: string;
    // SenseVoice 是否启用 ITN 文本规范化。
    senseVoiceUseItn: boolean;
    // SenseVoice 前置 VAD 模型。
    senseVoiceVadModel: string;
    // SenseVoice VAD 片段最大时长，单位毫秒。
    senseVoiceVadMaxSegmentMs: number;
    // SenseVoice 是否合并相邻 VAD 片段。
    senseVoiceMergeVad: boolean;
    // SenseVoice 合并后单段最长时长，单位秒。
    senseVoiceMergeLengthSeconds: number;
    // Paraformer 使用的 Python 解释器路径。
    paraformerPythonBin: string;
    // Paraformer 模型名称。
    paraformerModel: string;
    // Paraformer 模型版本。
    paraformerModelRevision: string;
    // Paraformer 前置 VAD 模型。
    paraformerVadModel: string;
    // Paraformer VAD 模型版本。
    paraformerVadModelRevision: string;
    // Paraformer 标点恢复模型。
    paraformerPuncModel: string;
    // Paraformer 标点恢复模型版本。
    paraformerPuncModelRevision: string;
    // Paraformer 热词配置文件路径。
    paraformerHotwordConfigPath: string;
    // Paraformer 最多注入多少个热词。
    paraformerMaxHotwords: number;
    // Paraformer VAD 片段最大时长，单位毫秒。
    paraformerVadMaxSegmentMs: number;
    // Paraformer 是否合并相邻 VAD 片段。
    paraformerMergeVad: boolean;
    // Paraformer 合并后单段最长时长，单位秒。
    paraformerMergeLengthSeconds: number;
    // 是否启用转写后的文本校正/润色任务。
    transcriptionCorrectionEnabled: boolean;
    // Pyannote diarization 使用的 Python 解释器路径。
    pyannotePythonBin: string;
    // Pyannote/HuggingFace 鉴权 token。
    pyannoteAuthToken: string;
    // 是否优先消费本地 pyannote pipeline config。
    // 开启后会在本地缓存完整时使用 patched config，避免断网时触发 HuggingFace HEAD 请求。
    pyannoteUseLocalConfig: boolean;
    // SpeechBrain 声纹识别使用的 Python 解释器路径。
    speechbrainPythonBin: string;
    // 本地模型缓存根目录。
    modelCacheRoot: string;
    // 目标人物声纹匹配阈值。
    targetSpeakerThreshold: number;
    // 连续发言合并后的最大时长，单位毫秒。
    utteranceMaxDurationMs: number;
    // 连续发言合并后的最大文本长度。
    utteranceMaxTextChars: number;
    // LLM 二次合并连续发言时允许的最大间隔，单位毫秒。
    utteranceLlmMergeMaxGapMs: number;
    // LLM 二次合并连续发言时单组最大时长，单位毫秒。
    utteranceLlmMergeMaxDurationMs: number;
    // LLM 二次合并连续发言时单组最大文本长度。
    utteranceLlmMergeMaxTextChars: number;
    // 是否启用 LLM 文本纠错。
    llmCorrectionEnabled: boolean;
    // LLM 文本纠错使用的 Python 解释器路径。
    llmCorrectionPythonBin: string;
    // LLM 文本纠错模型仓库。
    llmCorrectionModelRepo: string;
    // LLM 文本纠错模型文件名。
    llmCorrectionModelFile: string;
    // LLM 文本纠错上下文窗口大小。
    llmCorrectionContextSize: number;
    // LLM 文本纠错超时时间，单位毫秒。
    llmCorrectionTimeoutMs: number;
  };
  jobs: {
    // 是否在 Next.js 进程内启动嵌入式后台 worker。
    embeddedWorkerEnabled: boolean;
    // 后台 worker 并发处理录音任务数量。
    workerConcurrency: number;
    // 后台 worker 轮询间隔，单位毫秒。
    intervalMs: number;
    // 后台 worker 单次拉取任务数量。
    batchSize: number;
  };
  search: {
    // 是否启用 embedding 和向量检索。
    embeddingEnabled: boolean;
    // embedding 提供方，目前固定为本地 Qwen3 embedding。
    embeddingProvider: "local_qwen3";
    // embedding 模型名称。
    embeddingModel: string;
    // embedding 向量维度，必须与模型输出和数据库 vector 维度一致。
    embeddingDimensions: number;
    // embedding 使用的 Python 解释器路径。
    embeddingPythonBin: string;
    // embedding 推理设备，auto 会优先尝试 cuda、mps，再回退 cpu。
    embeddingDevice: "auto" | "cuda" | "mps" | "cpu";
    // embedding 批处理大小。
    embeddingBatchSize: number;
    // embedding 模型缓存目录。
    embeddingModelCacheDir: string;
    // 向量库原始召回 topK。
    vectorTopK: number;
    // 最终返回给 RAG 的 evidence 数量。
    finalTopK: number;
    // 向量命中后向前扩展的证据上下文窗口，单位毫秒。
    evidenceContextBeforeMs: number;
    // 向量命中后向后扩展的证据上下文窗口，单位毫秒。
    evidenceContextAfterMs: number;
    // 单条 evidence 最多拼接多少字符，避免一次命中塞入过长上下文。
    evidenceContextMaxChars: number;
    // 同一录音内相邻证据窗口间隔小于该值时合并，单位毫秒。
    evidenceMergeGapMs: number;
    // 召回结果最低相似度阈值。
    minScore: number;
    // 建索引时单个 search chunk 最大时长，单位毫秒。
    chunkMaxDurationMs: number;
    // 建索引时单个 search chunk 最大文本长度。
    chunkMaxTextChars: number;
    // 建索引时允许合并相邻 utterance 的最大间隔，单位毫秒。
    chunkMaxGapMs: number;
    // 是否启用生成式问答；关闭后只返回抽取式答案。
    answerEnabled: boolean;
    // RAG 问答模型提供方。
    answerProvider: "local_llm" | "deepseek_api" | "extractive";
    // 共享本地 LLM 模型仓库，用于 router、answer、validate 和本地录音总结。
    localLlmModelRepo: string;
    // 共享本地 LLM 模型文件名。
    localLlmModelFile: string;
    // RAG 问答上下文窗口大小。
    answerContextSize: number;
    // RAG 问答超时时间，单位毫秒。
    answerTimeoutMs: number;
    // DeepSeek API Key。
    deepseekApiKey: string;
    // DeepSeek API base URL。
    deepseekBaseUrl: string;
    // DeepSeek 模型名称。
    deepseekModel: string;
    // 录音总结模型提供方。
    summaryProvider: "local_llm" | "deepseek_api";
    // 录音总结 prompt 配置文件路径，默认复用 initial-prompt.json 的 summary 字段。
    summaryPromptConfigPath: string;
    // 录音总结上下文窗口大小，16384、262144
    summaryContextSize: number;
    // 录音总结最大输出 token 数。
    summaryMaxTokens: number;
    // 是否对长录音启用滚动记忆式分段总结。
    summaryRollingEnabled: boolean;
    // 超过该时长的录音会启用滚动记忆式分段总结，单位毫秒。
    summaryRollingThresholdMs: number;
    // 滚动总结每个分块的目标时长，单位毫秒。
    summaryRollingChunkDurationMs: number;
    // 滚动总结每个分块的最大文本字符数。
    summaryRollingChunkMaxChars: number;
    // 滚动总结每个分块输出 token 上限。
    summaryRollingChunkMaxTokens: number;
    // 传给下一块的滚动记忆最大字符数。
    summaryRollingMemoryMaxChars: number;
    // 录音总结超时时间，单位毫秒；0 表示不启用硬超时。
    summaryTimeoutMs: number;
  };
}

declare global {
  // eslint-disable-next-line no-var
  var __aiRecordSummaryConfig: AppConfig | undefined;
}

function numberFromEnv(value: string | undefined, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function buildAppConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  return {
    database: getDatabaseConfig(env),
    storage: {
      localRoot: env.LOCAL_STORAGE_ROOT || "uploads",
      publicFileBaseUrl: env.PUBLIC_FILE_BASE_URL || ""
    },
    audio: {
      installDependenciesOnStartup: env.AUDIO_DEPS_INSTALL_ON_STARTUP
        ? env.AUDIO_DEPS_INSTALL_ON_STARTUP !== "false"
        : env.AUDIO_DEPS_AUTO_INSTALL !== "false",
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
      qwenAsrSegmentTimeoutSeconds: numberFromEnv(env.QWEN_ASR_SEGMENT_TIMEOUT_S, 180),
      qwenAsrEnhanceLowVolumeSegments: env.QWEN_ASR_ENHANCE_LOW_VOLUME_SEGMENTS !== "false",
      qwenAsrLowVolumeRmsThreshold: Number(env.QWEN_ASR_LOW_VOLUME_RMS_THRESHOLD || 0.015),
      qwenAsrLowVolumePeakThreshold: Number(env.QWEN_ASR_LOW_VOLUME_PEAK_THRESHOLD || 0.12),
      qwenAsrVadModel: env.QWEN_ASR_VAD_MODEL || "fsmn-vad",
      qwenAsrVadMaxSegmentMs: Number(env.QWEN_ASR_VAD_MAX_SEGMENT_MS || 120000),
      qwenAsrVadMergeMaxGapMs: Number(env.QWEN_ASR_VAD_MERGE_MAX_GAP_MS || 2000),
      qwenAsrVadMinSegmentMs: Number(env.QWEN_ASR_VAD_MIN_SEGMENT_MS || 1200),
      qwenAsrMergeVad: env.QWEN_ASR_MERGE_VAD !== "false",
      qwenAsrMergeLengthSeconds: Number(env.QWEN_ASR_MERGE_LENGTH_S || 60000),
      qwenAsrStripTrailingPunctuation: env.QWEN_ASR_STRIP_TRAILING_PUNCTUATION !== "false",
      qwenAsrBreakOnSentenceEnd: env.QWEN_ASR_BREAK_ON_SENTENCE_END === "true",
      qwenAsrSpeakerSegmentMergeMaxGapMs: Number(env.QWEN_ASR_SPEAKER_SEGMENT_MERGE_MAX_GAP_MS || 2000),
      qwenAsrSpeakerSegmentMergeMaxDurationMs: Number(env.QWEN_ASR_SPEAKER_SEGMENT_MERGE_MAX_DURATION_MS || 60000),
      qwenAsrSpeakerSegmentMinDurationMs: Number(env.QWEN_ASR_SPEAKER_SEGMENT_MIN_DURATION_MS || 1200),
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
      pyannoteUseLocalConfig: env.PYANNOTE_USE_LOCAL_CONFIG !== "false",
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
      embeddingModel: env.EMBEDDING_MODEL || "Qwen/Qwen3-Embedding-4B",
      embeddingDimensions: Number(env.EMBEDDING_DIMENSIONS || 2560),
      embeddingPythonBin: env.EMBEDDING_PYTHON_BIN || env.WHISPER_PYTHON_BIN || ".venv-audio/bin/python",
      embeddingDevice:
        env.EMBEDDING_DEVICE === "cuda" || env.EMBEDDING_DEVICE === "mps" || env.EMBEDDING_DEVICE === "cpu"
          ? env.EMBEDDING_DEVICE
          : "auto",
      embeddingBatchSize: Math.max(1, Number(env.EMBEDDING_BATCH_SIZE || 16)),
      embeddingModelCacheDir: env.EMBEDDING_MODEL_CACHE_DIR || "model-cache/embedding",
      vectorTopK: Math.max(1, Number(env.SEARCH_VECTOR_TOP_K || 30)),
      finalTopK: Math.max(1, Number(env.SEARCH_FINAL_TOP_K || 10)),
      evidenceContextBeforeMs: Math.max(0, Number(env.SEARCH_EVIDENCE_CONTEXT_BEFORE_MS || 60000)),
      evidenceContextAfterMs: Math.max(0, Number(env.SEARCH_EVIDENCE_CONTEXT_AFTER_MS || 60000)),
      evidenceContextMaxChars: Math.max(200, Number(env.SEARCH_EVIDENCE_CONTEXT_MAX_CHARS || 1600)),
      evidenceMergeGapMs: Math.max(0, Number(env.SEARCH_EVIDENCE_MERGE_GAP_MS || 30000)),
      minScore: Number(env.SEARCH_MIN_SCORE || 0.25),
      chunkMaxDurationMs: Number(env.SEARCH_CHUNK_MAX_DURATION_MS || 60000),
      chunkMaxTextChars: Number(env.SEARCH_CHUNK_MAX_TEXT_CHARS || 800),
      chunkMaxGapMs: Number(env.SEARCH_CHUNK_MAX_GAP_MS || 10000),
      answerEnabled: env.RAG_ANSWER_ENABLED !== "false",
      answerProvider:
        env.RAG_ANSWER_PROVIDER === "deepseek_api"
          ? "deepseek_api"
          : env.RAG_ANSWER_PROVIDER === "extractive"
            ? "extractive"
            : "local_llm",
      localLlmModelRepo: env.LOCAL_LLM_MODEL_REPO || env.RAG_ANSWER_MODEL_REPO || "DevQuasar/Qwen.Qwen3.5-9B-GGUF",
      localLlmModelFile: env.LOCAL_LLM_MODEL_FILE || env.RAG_ANSWER_MODEL_FILE || "Qwen.Qwen3.5-9B.Q8_0.gguf",
      answerContextSize: Number(env.RAG_ANSWER_CONTEXT_SIZE || 8192),
      answerTimeoutMs: Number(env.RAG_ANSWER_TIMEOUT_MS || 600000),
      deepseekApiKey: env.DEEPSEEK_API_KEY || "",
      deepseekBaseUrl: env.DEEPSEEK_BASE_URL || "https://api.deepseek.com",
      deepseekModel: env.DEEPSEEK_MODEL || "deepseek-chat",
      summaryProvider: env.RECORDING_SUMMARY_PROVIDER === "deepseek_api" ? "deepseek_api" : "local_llm",
      summaryPromptConfigPath: env.RECORDING_SUMMARY_PROMPT_CONFIG || env.WHISPER_INITIAL_PROMPT_CONFIG || "config/initial-prompt.json",
      summaryContextSize: Number(env.RECORDING_SUMMARY_CONTEXT_SIZE || 262144),
      summaryMaxTokens: Number(env.RECORDING_SUMMARY_MAX_TOKENS || 4096),
      summaryRollingEnabled: env.RECORDING_SUMMARY_ROLLING_ENABLED !== "false",
      summaryRollingThresholdMs: Number(env.RECORDING_SUMMARY_ROLLING_THRESHOLD_MS || 30 * 60 * 1000),
      summaryRollingChunkDurationMs: Number(env.RECORDING_SUMMARY_ROLLING_CHUNK_DURATION_MS || 10 * 60 * 1000),
      summaryRollingChunkMaxChars: Number(env.RECORDING_SUMMARY_ROLLING_CHUNK_MAX_CHARS || 8000),
      summaryRollingChunkMaxTokens: Number(env.RECORDING_SUMMARY_ROLLING_CHUNK_MAX_TOKENS || 1800),
      summaryRollingMemoryMaxChars: Number(env.RECORDING_SUMMARY_ROLLING_MEMORY_MAX_CHARS || 6000),
      summaryTimeoutMs: Number(env.RECORDING_SUMMARY_TIMEOUT_MS || 0)
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
