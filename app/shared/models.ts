export type RecordingStatus = "uploaded" | "processing" | "completed" | "failed";
export type SpeakerProfileStatus = "active" | "inactive";
export type SampleStatus = RecordingStatus;

export type Recording = {
  id: string;
  title: string;
  fileName: string;
  storagePath: string;
  location: string | null;
  mimeType: string;
  fileSizeBytes: number;
  durationSeconds: number | null;
  processingDurationMs: number | null;
  status: RecordingStatus;
  errorMessage: string | null;
  uploadedAt: string;
  createdAt: string;
  updatedAt: string;
};

export type RecordingSummary = { id: string; recordingId: string; provider: "local_llm" | "deepseek_api"; modelName: string; summaryText: string; createdAt: string; updatedAt: string };
export type Transcription = { id: string; recordingId: string; language: string | null; modelName: string; fullText: string; segmentCount: number; createdAt: string; updatedAt: string };
export type TranscriptionSegment = { id: string; recordingId: string; transcriptionId: string; segmentIndex: number; startMs: number; endMs: number; text: string; speakerLabel: string | null; speakerClusterId: string | null; speakerConfidence: number | null; isTargetPerson: boolean; targetPersonConfidence: number | null; diarizationSegmentId: string | null; matchedSpeakerProfileId: string | null; createdAt: string };
export type TranscriptionToken = { id: string; recordingId: string; transcriptionId: string; tokenIndex: number; sourceWindowIndex: number; text: string; startMs: number; endMs: number; speakerClusterId: string | null; speakerLabel: string | null; attributionStatus: string };
export type SpeakerDiarizationSegment = { id: string; recordingId: string; speakerClusterId: string; speakerLabel: string; startMs: number; endMs: number; confidence: number | null; isTargetPerson: boolean; targetPersonConfidence: number | null; matchedSpeakerProfileId: string | null; createdAt: string };
export type UtteranceSegment = { id: string; recordingId: string; utteranceIndex: number; startMs: number; endMs: number; text: string; speakerLabel: string | null; speakerClusterId: string | null; sourceTranscriptionSegmentIds: string[]; isTargetPerson: boolean; targetPersonConfidence: number | null; matchedSpeakerProfileId: string | null; mergeReason: string; createdAt: string };
export type SpeakerProfile = { id: string; displayName: string; status: SpeakerProfileStatus; notes: string | null; createdAt: string; updatedAt: string };
export type SpeakerProfileSample = { id: string; speakerProfileId: string; fileName: string; storagePath: string; mimeType: string; fileSizeBytes: number; durationSeconds: number | null; status: SampleStatus; errorMessage: string | null; createdAt: string; updatedAt: string };
export type SpeakerProfileWithSamples = SpeakerProfile & { samples: SpeakerProfileSample[] };

export type SearchEvidence = {
  index: number;
  recording: { id: string; title: string; fileName: string; location: string | null; durationSeconds: number | null };
  chunk: { id: string; text?: string; startMs: number; endMs: number; speakerLabels: string[]; isTargetPerson: boolean; matchedSpeakerProfiles: Array<{ id: string; displayName: string }> };
  score: number;
  matchType: "vector" | "keyword" | "hybrid";
  url: string;
};
export type RagAnswer = { text: string; citations: Array<{ index: number; chunkId: string; recordingId: string; startMs: number; endMs: number }>; notEnoughEvidence: boolean };
export type RagQueryResponse = { queryId: string; query: string; answer: RagAnswer | null; evidence: SearchEvidence[]; message?: string };
