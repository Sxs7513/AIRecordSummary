import { getAppConfig } from "../config/app-config";
import { getAnswerProvider } from "./answering";
import { ExtractiveAnswerProvider } from "./answering/extractive";
import { elapsedMs, ragLog, textPreview } from "./debug";
import { retrieveSearchEvidence } from "./search";
import type { RagQueryInput } from "./types";
import type { RagAnswer, RagCitation, RagQueryResponse, SearchEvidence } from "../types/models";

export function validateRagAnswer(answer: RagAnswer, evidence: SearchEvidence[]): RagAnswer {
  const startedAt = Date.now();
  const byIndex = new Map(evidence.map((item) => [item.index, item]));
  const citations: RagCitation[] = [];
  for (const citation of answer.citations) {
    const item = byIndex.get(citation.index);
    if (!item) continue;
    if (citation.chunkId !== item.chunk.id || citation.recordingId !== item.recording.id) continue;
    citations.push({
      index: item.index,
      chunkId: item.chunk.id,
      recordingId: item.recording.id,
      startMs: item.chunk.startMs,
      endMs: item.chunk.endMs
    });
  }
  if (!answer.notEnoughEvidence && citations.length === 0 && evidence.length > 0) {
    ragLog("answer.validate_failed", {
      rawCitationCount: answer.citations.length,
      evidenceCount: evidence.length,
      answerPreview: textPreview(answer.text)
    });
    throw new Error("RAG answer citation validation failed");
  }
  ragLog("answer.validate_done", {
    rawCitationCount: answer.citations.length,
    validCitationCount: citations.length,
    evidenceCount: evidence.length,
    notEnoughEvidence: answer.notEnoughEvidence,
    answerLength: answer.text.length,
    durationMs: elapsedMs(startedAt)
  });
  return {
    text: answer.text.trim(),
    citations,
    notEnoughEvidence: answer.notEnoughEvidence || evidence.length === 0
  };
}

export async function runRagQuery(input: RagQueryInput): Promise<RagQueryResponse> {
  const config = getAppConfig().search;
  const startedAt = Date.now();
  ragLog("query.start", {
    mode: input.mode ?? "answer",
    provider: config.answerProvider,
    answerEnabled: config.answerEnabled,
    queryPreview: textPreview(input.query),
    limit: input.limit ?? null,
    filters: input.filters ?? {}
  });
  const retrieved = await retrieveSearchEvidence(input);
  ragLog("query.retrieved", {
    queryId: retrieved.queryId,
    evidenceCount: retrieved.evidence.length,
    evidence: retrieved.evidence.slice(0, 5).map((item) => ({
      index: item.index,
      chunkId: item.chunk.id,
      recordingId: item.recording.id,
      title: item.recording.title,
      score: Number(item.score.toFixed(4)),
      time: `${item.chunk.startMs}-${item.chunk.endMs}`,
      textPreview: textPreview(item.chunk.text, 80)
    }))
  });
  if (input.mode === "retrieve_only") {
    ragLog("query.done", {
      queryId: retrieved.queryId,
      mode: "retrieve_only",
      durationMs: elapsedMs(startedAt)
    });
    return {
      queryId: retrieved.queryId,
      query: input.query,
      answer: null,
      evidence: retrieved.evidence,
      message: retrieved.message
    };
  }

  if (retrieved.evidence.length === 0) {
    ragLog("query.no_evidence", {
      queryId: retrieved.queryId,
      durationMs: elapsedMs(startedAt)
    });
    return {
      queryId: retrieved.queryId,
      query: input.query,
      answer: {
        text: "没有在录音中找到足够依据。",
        citations: [],
        notEnoughEvidence: true
      },
      evidence: [],
      message: retrieved.message
    };
  }

  let answer: RagAnswer;
  try {
    const provider = config.answerEnabled ? getAnswerProvider() : new ExtractiveAnswerProvider();
    const answerStartedAt = Date.now();
    ragLog("answer.start", {
      queryId: retrieved.queryId,
      provider: config.answerEnabled ? config.answerProvider : "extractive_disabled_answer",
      evidenceCount: retrieved.evidence.length,
      evidenceTokenApprox: retrieved.evidence.reduce((sum, item) => sum + Math.ceil(item.chunk.text.length / 2), 0)
    });
    answer = validateRagAnswer(await provider.generateAnswer({ query: input.query, evidence: retrieved.evidence, outputLanguage: "zh-CN" }), retrieved.evidence);
    ragLog("answer.done", {
      queryId: retrieved.queryId,
      provider: config.answerEnabled ? config.answerProvider : "extractive_disabled_answer",
      answerLength: answer.text.length,
      citationCount: answer.citations.length,
      notEnoughEvidence: answer.notEnoughEvidence,
      durationMs: elapsedMs(answerStartedAt),
      answerPreview: textPreview(answer.text)
    });
  } catch (error) {
    ragLog("answer.error", {
      queryId: retrieved.queryId,
      message: error instanceof Error ? error.message : String(error)
    });
    console.error("[rag] answer generation failed, falling back to extractive answer", error);
    const fallbackStartedAt = Date.now();
    answer = validateRagAnswer(await new ExtractiveAnswerProvider().generateAnswer({ query: input.query, evidence: retrieved.evidence }), retrieved.evidence);
    ragLog("answer.fallback_done", {
      queryId: retrieved.queryId,
      answerLength: answer.text.length,
      citationCount: answer.citations.length,
      durationMs: elapsedMs(fallbackStartedAt)
    });
  }

  ragLog("query.done", {
    queryId: retrieved.queryId,
    mode: input.mode ?? "answer",
    evidenceCount: retrieved.evidence.length,
    answerLength: answer.text.length,
    citationCount: answer.citations.length,
    durationMs: elapsedMs(startedAt)
  });
  return {
    queryId: retrieved.queryId,
    query: input.query,
    answer,
    evidence: retrieved.evidence
  };
}
