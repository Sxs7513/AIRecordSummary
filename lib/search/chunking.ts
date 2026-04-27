import { normalizeSearchText } from "./normalize";
import type { ChunkableUtterance, SearchChunkDraft } from "./types";

function unique<T>(items: Array<T | null | undefined>): T[] {
  return Array.from(new Set(items.filter((item): item is T => item !== null && item !== undefined && item !== "")));
}

function joinChunkText(parts: string[]) {
  return parts
    .map((part) => part.trim())
    .filter(Boolean)
    .join("\n")
    .replace(/\s+\n/g, "\n")
    .trim();
}

function speakerKey(utterance: ChunkableUtterance) {
  return utterance.speakerClusterId || utterance.speakerLabel || "unknown";
}

function canAppend(current: ChunkableUtterance[], next: ChunkableUtterance, options: { maxDurationMs: number; maxTextChars: number; maxGapMs: number }) {
  const first = current[0];
  const previous = current[current.length - 1];
  if (!first || !previous) return true;
  if (speakerKey(previous) !== speakerKey(next)) return false;
  if (next.startMs - previous.endMs > options.maxGapMs) return false;
  if (next.endMs - first.startMs > options.maxDurationMs) return false;
  const text = joinChunkText([...current.map((item) => item.text), next.text]);
  if (options.maxTextChars > 0 && text.length > options.maxTextChars) return false;
  return true;
}

function buildDraft(index: number, group: ChunkableUtterance[]): SearchChunkDraft {
  const first = group[0];
  const last = group[group.length - 1];
  const text = joinChunkText(group.map((utterance) => {
    const speaker = utterance.speakerLabel ? `${utterance.speakerLabel}: ` : "";
    return `${speaker}${utterance.text}`;
  }));
  return {
    chunkIndex: index,
    text,
    normalizedText: normalizeSearchText(text),
    startMs: first.startMs,
    endMs: last.endMs,
    speakerLabels: unique(group.map((utterance) => utterance.speakerLabel)),
    speakerClusterIds: unique(group.map((utterance) => utterance.speakerClusterId)),
    sourceUtteranceSegmentIds: group.map((utterance) => utterance.id),
    sourceTranscriptionSegmentIds: unique(group.flatMap((utterance) => utterance.sourceTranscriptionSegmentIds)),
    isTargetPerson: group.some((utterance) => utterance.isTargetPerson),
    matchedSpeakerProfileIds: unique(group.map((utterance) => utterance.matchedSpeakerProfileId)),
    metadata: {
      utteranceCount: group.length,
      mergeStrategy: "same_speaker_gap_limited"
    }
  };
}

export function buildSearchChunks(utterances: ChunkableUtterance[], options: { maxDurationMs: number; maxTextChars: number; maxGapMs: number }): SearchChunkDraft[] {
  const chunks: SearchChunkDraft[] = [];
  let group: ChunkableUtterance[] = [];

  const flush = () => {
    if (group.length === 0) return;
    chunks.push(buildDraft(chunks.length, group));
    group = [];
  };

  for (const utterance of utterances) {
    if (!utterance.text.trim()) continue;
    if (group.length === 0 || canAppend(group, utterance, options)) {
      group.push(utterance);
      continue;
    }
    flush();
    group.push(utterance);
  }
  flush();
  return chunks;
}
