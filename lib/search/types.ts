import type { RagAnswer, SearchEvidence, SearchFilters, UtteranceSegment } from "../types/models";

export interface SearchChunkDraft {
  chunkIndex: number;
  text: string;
  normalizedText: string;
  startMs: number;
  endMs: number;
  speakerLabels: string[];
  speakerClusterIds: string[];
  sourceUtteranceSegmentIds: string[];
  sourceTranscriptionSegmentIds: string[];
  isTargetPerson: boolean;
  matchedSpeakerProfileIds: string[];
  metadata: Record<string, unknown>;
}

export interface SearchChunkRow extends SearchChunkDraft {
  id: string;
  recordingId: string;
  embeddingModelId: string;
  createdAt: string;
  updatedAt: string;
}

export interface SearchInput {
  query: string;
  limit?: number;
  filters?: SearchFilters;
}

export interface SearchOutput {
  queryId: string;
  evidence: SearchEvidence[];
  message?: string;
}

export interface RagQueryInput extends SearchInput {
  mode?: "answer" | "retrieve_only";
}

export interface RagAnswerInput {
  query: string;
  evidence: SearchEvidence[];
  outputLanguage?: string;
}

export type RagAnswerOutput = RagAnswer;

export type ChunkableUtterance = UtteranceSegment;
