import path from "node:path";
import { spawn } from "node:child_process";
import { getAppConfig } from "../../config/app-config";
import { resolveSharedLocalLlmModelPath } from "../../local-llm/model-path";
import { audioProcessEnv } from "../runtime/runtime-env";
import { loadSummarySystemPrompt } from "./prompt-config";
import type { Recording, UtteranceSegment } from "../../types/models";

export interface RecordingSummaryResult {
  provider: "local_llm" | "deepseek_api";
  modelName: string;
  summaryText: string;
}

function setOptionalTimeout(callback: () => void, timeoutMs: number) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return null;
  return setTimeout(callback, timeoutMs);
}

function clearOptionalTimeout(timer: NodeJS.Timeout | null) {
  if (timer) clearTimeout(timer);
}

interface SummaryUtteranceInput {
  speakerLabel: string | null;
  startMs: number;
  endMs: number;
  text: string;
}

interface SummaryChunk {
  index: number;
  startMs: number;
  endMs: number;
  utterances: SummaryUtteranceInput[];
}

function formatTime(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function utteranceInputSize(utterance: SummaryUtteranceInput) {
  return utterance.text.length + (utterance.speakerLabel?.length ?? 0) + 32;
}

function toSummaryUtterance(utterance: UtteranceSegment): SummaryUtteranceInput {
  return {
    speakerLabel: utterance.speakerLabel,
    startMs: utterance.startMs,
    endMs: utterance.endMs,
    text: utterance.text
  };
}

function truncateUtterances(utterances: UtteranceSegment[], maxChars: number) {
  let used = 0;
  const output: SummaryUtteranceInput[] = [];
  for (const utterance of utterances) {
    const remaining = maxChars - used;
    if (remaining <= 0) break;
    const text = utterance.text.length > remaining ? utterance.text.slice(0, remaining) : utterance.text;
    const item = { ...toSummaryUtterance(utterance), text };
    output.push(item);
    used += utteranceInputSize(item);
  }
  return output;
}

function recordingDurationMs(utterances: UtteranceSegment[]) {
  if (utterances.length === 0) return 0;
  const start = Math.min(...utterances.map((item) => item.startMs));
  const end = Math.max(...utterances.map((item) => item.endMs));
  return Math.max(0, end - start);
}

function shouldUseRollingSummary(utterances: UtteranceSegment[]) {
  const config = getAppConfig().search;
  return config.summaryRollingEnabled && recordingDurationMs(utterances) >= config.summaryRollingThresholdMs;
}

function buildSummaryChunks(utterances: UtteranceSegment[]): SummaryChunk[] {
  const config = getAppConfig().search;
  const effectiveChunkMaxChars = Math.min(
    config.summaryRollingChunkMaxChars,
    Math.max(3000, config.summaryContextSize - config.summaryRollingChunkMaxTokens - config.summaryRollingMemoryMaxChars - 1800)
  );
  const chunks: SummaryChunk[] = [];
  let current: SummaryUtteranceInput[] = [];
  let currentStartMs: number | null = null;
  let currentEndMs = 0;
  let currentChars = 0;

  for (const utterance of utterances) {
    const item = toSummaryUtterance(utterance);
    const itemSize = utteranceInputSize(item);
    const chunkDuration = currentStartMs === null ? 0 : Math.max(0, item.endMs - currentStartMs);
    const shouldFlush = current.length > 0
      && (chunkDuration > config.summaryRollingChunkDurationMs || currentChars + itemSize > effectiveChunkMaxChars);
    if (shouldFlush) {
      chunks.push({
        index: chunks.length + 1,
        startMs: currentStartMs ?? current[0].startMs,
        endMs: currentEndMs,
        utterances: current
      });
      current = [];
      currentStartMs = null;
      currentEndMs = 0;
      currentChars = 0;
    }
    currentStartMs ??= item.startMs;
    currentEndMs = item.endMs;
    currentChars += itemSize;
    current.push(item);
  }

  if (current.length > 0) {
    chunks.push({
      index: chunks.length + 1,
      startMs: currentStartMs ?? current[0].startMs,
      endMs: currentEndMs,
      utterances: current
    });
  }
  return chunks;
}

function summaryInputCharBudget(contextSize: number, maxTokens: number) {
  const reservedForPrompt = 1200;
  const available = contextSize - maxTokens - reservedForPrompt;
  return Math.max(1200, available);
}

function stripThinking(text: string) {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<\/?think>/gi, "")
    .replace(/^\s*(思考过程|推理过程|分析过程)[:：][\s\S]*?(?=\n\n|$)/, "")
    .trim();
}

async function generateWithLocalLlm(recording: Recording, utterances: UtteranceSegment[]): Promise<RecordingSummaryResult> {
  const appConfig = getAppConfig();
  const config = appConfig.search;
  const systemPrompt = await loadSummarySystemPrompt(config.summaryPromptConfigPath);
  const { modelFile, modelPath } = resolveSharedLocalLlmModelPath({
    modelCacheRoot: appConfig.audio.modelCacheRoot,
    modelRepo: config.localLlmModelRepo,
    modelFile: config.localLlmModelFile
  });
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "summary", "scripts", "run_llm_summary.py");
  const contextSize = config.summaryContextSize;
  const useRolling = shouldUseRollingSummary(utterances);
  const summarizedUtterances = useRolling ? [] : truncateUtterances(utterances, summaryInputCharBudget(contextSize, config.summaryMaxTokens));
  const chunks = useRolling ? buildSummaryChunks(utterances) : [];

  return new Promise<RecordingSummaryResult>((resolve, reject) => {
    const child = spawn(config.embeddingPythonBin, [script], {
      cwd: process.cwd(),
      env: audioProcessEnv(config.embeddingPythonBin, appConfig.audio.modelCacheRoot)
    });
    const timer = setOptionalTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`local recording summary timed out after ${config.summaryTimeoutMs}ms`));
    }, config.summaryTimeoutMs);
    let stdout = "";
    let stderr = "";
    let settled = false;
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearOptionalTimeout(timer);
      reject(error);
    });
    child.on("exit", (code) => {
      if (settled) return;
      settled = true;
      clearOptionalTimeout(timer);
      if (code !== 0) {
        reject(new Error(`local recording summary exited with code ${code}\n${stderr}`));
        return;
      }
      try {
        const payload = JSON.parse(stdout) as { summaryText?: string };
        resolve({ provider: "local_llm", modelName: modelFile, summaryText: payload.summaryText || "暂无可总结的润色文本。" });
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(JSON.stringify({
      title: recording.title,
      mode: useRolling ? "rolling" : "single",
      utterances: summarizedUtterances,
      chunks,
      systemPrompt,
      modelPath,
      contextSize,
      maxTokens: config.summaryMaxTokens,
      chunkMaxTokens: config.summaryRollingChunkMaxTokens,
      memoryMaxChars: config.summaryRollingMemoryMaxChars
    }));
  });
}

function buildDeepSeekSingleUserPayload(recording: Recording, utterances: SummaryUtteranceInput[]) {
  return JSON.stringify({
    title: recording.title,
    utterances: utterances.map((item) => ({
      time: `${formatTime(item.startMs)}-${formatTime(item.endMs)}`,
      speakerLabel: item.speakerLabel,
      text: item.text
    }))
  }, null, 2);
}

async function callDeepSeekSummary(input: { systemPrompt: string; userContent: string; maxTokens: number; signal?: AbortSignal }) {
  const config = getAppConfig().search;
  const response = await fetch(`${config.deepseekBaseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${config.deepseekApiKey}`
    },
    body: JSON.stringify({
      model: config.deepseekModel,
      max_tokens: input.maxTokens,
      messages: [
        {
          role: "system",
          content: input.systemPrompt
        },
        {
          role: "user",
          content: input.userContent
        }
      ]
    }),
    signal: input.signal
  });
  if (!response.ok) throw new Error(`DeepSeek summary failed: ${response.status} ${await response.text()}`);
  const payload = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
  return stripThinking(payload.choices?.[0]?.message?.content ?? "");
}

function extractJsonObject(text: string) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    return JSON.parse(match[0]) as { chunkSummary?: string; memory?: string };
  } catch {
    return null;
  }
}

async function generateDeepSeekRolling(recording: Recording, utterances: UtteranceSegment[], systemPrompt: string, signal: AbortSignal): Promise<string> {
  const config = getAppConfig().search;
  const chunks = buildSummaryChunks(utterances);
  const chunkSummaries: Array<{ index: number; timeRange: string; summary: string }> = [];
  let memory = "";

  for (const chunk of chunks) {
    if (signal.aborted) throw new Error("DeepSeek summary aborted");
    const userContent = JSON.stringify({
      task: "请总结当前片段，并更新传给后续片段的滚动记忆。只输出 JSON。",
      outputSchema: { chunkSummary: "当前片段总结", memory: "更新后的滚动记忆" },
      recordingTitle: recording.title,
      chunkIndex: chunk.index,
      totalChunks: chunks.length,
      timeRange: `${formatTime(chunk.startMs)}-${formatTime(chunk.endMs)}`,
      previousMemory: memory,
      utterances: chunk.utterances.map((item) => ({
        time: `${formatTime(item.startMs)}-${formatTime(item.endMs)}`,
        speakerLabel: item.speakerLabel,
        text: item.text
      }))
    }, null, 2);
    const raw = await callDeepSeekSummary({
      systemPrompt: `${systemPrompt}\n你正在做滚动记忆式长录音总结。当前步骤必须输出严格 JSON，不要 Markdown。chunkSummary 按原文顺序概括当前内容的讨论重点，不要用“阶段/片段”作为标题，不要机械写成逐个 speaker 的发言记录。memory 保留后面还会用到的事实、结论和待办。`,
      userContent,
      maxTokens: config.summaryRollingChunkMaxTokens,
      signal
    });
    const parsed = extractJsonObject(raw);
    const chunkSummary = stripThinking(parsed?.chunkSummary || raw || "");
    memory = stripThinking(parsed?.memory || `${memory}\n${chunkSummary}`).slice(-config.summaryRollingMemoryMaxChars);
    chunkSummaries.push({ index: chunk.index, timeRange: `${formatTime(chunk.startMs)}-${formatTime(chunk.endMs)}`, summary: chunkSummary });
  }

  const finalUserContent = JSON.stringify({
    task: "请基于所有片段总结和最终滚动记忆，输出给用户看的最终录音总结。片段总结只是内部处理中间结果，最终输出不要出现“片段1/片段2/阶段一/阶段二”等内部编号或流程标签。开头先写一个全局总结，用1-2段话概括整段录音的核心内容、主要结论或整体结果。然后按照录音里的先后顺序总结，可以自然分成几段；每段标题要直接写真实主题，而不是写处理阶段。总结要比逐句复述更高一层，不要机械写成逐个 speaker 的发言记录。保留具体事实、数字、结论和待办；只有人物身份本身重要时才提到人。用自然的大白话写，不要写成报告腔，也不要只写空泛概括。不要把长录音压缩成很短一段。",
    recordingTitle: recording.title,
    finalMemory: memory,
    chunkSummaries
  }, null, 2);
  return await callDeepSeekSummary({ systemPrompt, userContent: finalUserContent, maxTokens: config.summaryMaxTokens, signal });
}

async function generateWithDeepSeek(recording: Recording, utterances: UtteranceSegment[]): Promise<RecordingSummaryResult> {
  const config = getAppConfig().search;
  if (!config.deepseekApiKey) throw new Error("DEEPSEEK_API_KEY is required for recording summary");
  const systemPrompt = await loadSummarySystemPrompt(config.summaryPromptConfigPath);
  const controller = new AbortController();
  const timer = setOptionalTimeout(() => controller.abort(), config.summaryTimeoutMs);
  const summarizedUtterances = truncateUtterances(utterances, summaryInputCharBudget(config.summaryContextSize, config.summaryMaxTokens));
  try {
    const summaryText = shouldUseRollingSummary(utterances)
      ? await generateDeepSeekRolling(recording, utterances, systemPrompt, controller.signal)
      : await callDeepSeekSummary({ systemPrompt, userContent: buildDeepSeekSingleUserPayload(recording, summarizedUtterances), maxTokens: config.summaryMaxTokens, signal: controller.signal });
    return { provider: "deepseek_api", modelName: config.deepseekModel, summaryText: summaryText || "暂无可总结的润色文本。" };
  } finally {
    clearOptionalTimeout(timer);
  }
}

export async function generateRecordingSummary(recording: Recording, utterances: UtteranceSegment[]): Promise<RecordingSummaryResult> {
  const config = getAppConfig().search;
  if (config.summaryProvider === "deepseek_api") return generateWithDeepSeek(recording, utterances);
  return generateWithLocalLlm(recording, utterances);
}
