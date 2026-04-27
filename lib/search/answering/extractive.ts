import type { RagAnswerInput, RagAnswerOutput } from "../types";
import type { RagAnswerProvider } from "./provider";

export class ExtractiveAnswerProvider implements RagAnswerProvider {
  async generateAnswer(input: RagAnswerInput): Promise<RagAnswerOutput> {
    if (input.evidence.length === 0) {
      return {
        text: "没有在录音中找到足够依据。",
        citations: [],
        notEnoughEvidence: true
      };
    }
    const topEvidence = input.evidence.slice(0, 3);
    return {
      text: topEvidence
        .map((evidence) => `根据录音《${evidence.recording.title}》${formatRange(evidence.chunk.startMs, evidence.chunk.endMs)}，${evidence.chunk.text} [${evidence.index}]`)
        .join("\n\n"),
      citations: topEvidence.map((evidence) => ({
        index: evidence.index,
        chunkId: evidence.chunk.id,
        recordingId: evidence.recording.id,
        startMs: evidence.chunk.startMs,
        endMs: evidence.chunk.endMs
      })),
      notEnoughEvidence: false
    };
  }
}

function formatRange(startMs: number, endMs: number) {
  return `${formatMs(startMs)}-${formatMs(endMs)}`;
}

function formatMs(value: number) {
  const totalSeconds = Math.floor(value / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}
