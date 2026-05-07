import path from "node:path";
import { spawn } from "node:child_process";
import { z } from "zod";
import { getAppConfig } from "../../config/app-config";
import { audioProcessEnv } from "../../audio-transcoding-analysis/runtime/runtime-env";
import { resolveSharedLocalLlmModelPath } from "../../local-llm/model-path";
import { ragLog, textPreview } from "../debug";
import { firstJsonObject } from "../router/prompt";
import type { AnswerValidation } from "../graph-state";
import type { RagAnswer, SearchEvidence } from "../../types/models";

const llmValidationSchema = z.object({
  valid: z.boolean(),
  failureKind: z.enum(["none", "bad_citation", "unsupported_claim", "wrong_evidence", "not_enough_evidence"]),
  reason: z.string().default(""),
  unsupportedClaims: z.array(z.string()).default([]),
  badCitations: z.array(z.number()).default([]),
  rewriteInstruction: z.string().nullable().default(null),
  rewriteQuery: z.string().nullable().default(null)
});

function truncateText(text: string, maxChars: number) {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, Math.max(0, maxChars))}\n[已截断]`;
}

function evidenceForValidation(evidence: SearchEvidence[], answer: RagAnswer) {
  const citedIndexes = new Set(answer.citations.map((citation) => citation.index));
  if (citedIndexes.size === 0) return evidence.slice(0, 8);
  const citedEvidence = evidence.filter((item) => citedIndexes.has(item.index));
  return citedEvidence.length > 0 ? citedEvidence : evidence.slice(0, 8);
}

function buildValidationPrompt(input: { query: string; evidence: SearchEvidence[]; answer: RagAnswer }, maxPromptChars: number) {
  const validationEvidence = evidenceForValidation(input.evidence, input.answer);
  const fixedBudget = input.query.length + input.answer.text.length + 1800;
  const evidenceBudget = Math.max(1600, maxPromptChars - fixedBudget);
  const perEvidenceBudget = Math.max(500, Math.floor(evidenceBudget / Math.max(1, validationEvidence.length)));
  const evidence = validationEvidence
    .map((item) => {
      const header = `[${item.index}] ${item.recording.title} / ${item.recording.location ?? "未配置地点"} ${item.chunk.startMs}-${item.chunk.endMs}ms`;
      return `${header}\n${truncateText(item.chunk.text, perEvidenceBudget)}`;
    })
    .join("\n\n");
  const answerText = truncateText(input.answer.text, Math.max(1200, Math.floor(maxPromptChars * 0.25)));
  return (
    "<|im_start|>system\n" +
    "你是 RAG 答案校验器。只输出 JSON，不要回答用户问题。\n" +
    "判断回答是否完全由证据支撑，并判断失败原因。证据可能已截断；如果无法判断，不要扩大事实，只根据可见证据判断。\n" +
    "failureKind 只能是 none、bad_citation、unsupported_claim、wrong_evidence、not_enough_evidence。\n" +
    "如果证据不够或检索明显不对，使用 wrong_evidence 或 not_enough_evidence，并给 rewriteQuery。\n" +
    "如果证据够但回答引用错或说了证据没有的话，使用 bad_citation 或 unsupported_claim，并给 rewriteInstruction。\n" +
    "JSON schema: {\"valid\":true,\"failureKind\":\"none\",\"reason\":\"...\",\"unsupportedClaims\":[],\"badCitations\":[],\"rewriteInstruction\":null,\"rewriteQuery\":null}\n" +
    "<|im_end|>\n" +
    "<|im_start|>user\n" +
    `用户问题：${input.query}\n\n证据：\n${evidence}\n\n回答：\n${answerText}\n\n引用：${JSON.stringify(input.answer.citations)}\n` +
    "<|im_end|>\n<|im_start|>assistant\n"
  );
}

export async function validateAnswerWithLocalLlm(input: { query: string; evidence: SearchEvidence[]; answer: RagAnswer }): Promise<AnswerValidation | null> {
  const appConfig = getAppConfig();
  const config = appConfig.search;
  if (!config.answerEnabled || config.answerProvider !== "local_llm" || input.evidence.length === 0 || input.answer.notEnoughEvidence) return null;

  const script = path.join(process.cwd(), "lib", "search", "grading", "scripts", "run_llm_validator.py");
  const { modelPath } = resolveSharedLocalLlmModelPath({
    modelCacheRoot: appConfig.audio.modelCacheRoot,
    modelRepo: config.localLlmModelRepo,
    modelFile: config.localLlmModelFile
  });
  const contextSize = Math.min(config.answerContextSize, 8192);
  const prompt = buildValidationPrompt(input, Math.max(6000, contextSize * 3 - 2000));
  ragLog("answer.llm_validate_start", {
    evidenceCount: input.evidence.length,
    citationCount: input.answer.citations.length,
    answerLength: input.answer.text.length,
    promptLength: prompt.length,
    contextSize
  });

  try {
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(config.embeddingPythonBin, [script], {
        cwd: process.cwd(),
        env: audioProcessEnv(config.embeddingPythonBin, appConfig.audio.modelCacheRoot)
      });
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        reject(new Error("answer validator timed out"));
      }, Math.min(config.answerTimeoutMs, 120000));
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("error", (error) => {
        clearTimeout(timer);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`answer validator exited with code ${code}\n${stderr}`));
          return;
        }
        const payload = JSON.parse(stdout) as { text?: string };
        resolve(payload.text ?? "");
      });
      child.stdin.end(JSON.stringify({
        prompt,
        modelPath,
        contextSize,
        maxTokens: 500,
        temperature: 0,
        stop: ["</s>", "<|im_end|>"]
      }));
    });
    const jsonText = firstJsonObject(output);
    if (!jsonText) throw new Error("validator did not return JSON");
    const validation = llmValidationSchema.parse(JSON.parse(jsonText));
    ragLog("answer.llm_validate_done", {
      valid: validation.valid,
      failureKind: validation.failureKind,
      reason: validation.reason,
      rewriteQuery: validation.rewriteQuery ? textPreview(validation.rewriteQuery) : null
    });
    return validation;
  } catch (error) {
    ragLog("answer.llm_validate_error", {
      message: error instanceof Error ? error.message : String(error)
    });
    return null;
  }
}
