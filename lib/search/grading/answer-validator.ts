import { elapsedMs, ragLog, textPreview } from "../debug";
import type { RagAnswer, RagCitation, SearchEvidence } from "../../types/models";

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
