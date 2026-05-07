export type RecordingStatus = "uploaded" | "processing" | "completed" | "failed";
export type JobType = "transcription" | "speaker_diarization" | "speaker_identification" | "text_correction" | "embedding_indexing" | "summary";
export type JobStatus = "pending" | "running" | "completed" | "failed";
export type SpeakerProfileStatus = "active" | "inactive";
export type SampleStatus = "uploaded" | "processing" | "completed" | "failed";

export interface Recording {
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
}

export interface Transcription {
  id: string;
  recordingId: string;
  language: string | null;
  modelName: string;
  fullText: string;
  segmentCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface RecordingSummary {
  id: string;
  recordingId: string;
  provider: "local_llm" | "deepseek_api";
  modelName: string;
  summaryText: string;
  createdAt: string;
  updatedAt: string;
}

export interface TranscriptionSegment {
  id: string;
  recordingId: string;
  transcriptionId: string;
  segmentIndex: number;
  startMs: number;
  endMs: number;
  text: string;
  speakerLabel: string | null;
  speakerClusterId: string | null;
  speakerConfidence: number | null;
  isTargetPerson: boolean;
  targetPersonConfidence: number | null;
  diarizationSegmentId: string | null;
  matchedSpeakerProfileId: string | null;
  createdAt: string;
}

export interface SpeakerDiarizationSegment {
  id: string;
  recordingId: string;
  speakerClusterId: string;
  speakerLabel: string;
  startMs: number;
  endMs: number;
  confidence: number | null;
  isTargetPerson: boolean;
  targetPersonConfidence: number | null;
  matchedSpeakerProfileId: string | null;
  createdAt: string;
}

export interface UtteranceSegment {
  id: string;
  recordingId: string;
  utteranceIndex: number;
  startMs: number;
  endMs: number;
  text: string;
  speakerLabel: string | null;
  speakerClusterId: string | null;
  sourceTranscriptionSegmentIds: string[];
  isTargetPerson: boolean;
  targetPersonConfidence: number | null;
  matchedSpeakerProfileId: string | null;
  mergeReason: string;
  createdAt: string;
}

export interface SpeakerProfile {
  id: string;
  displayName: string;
  status: SpeakerProfileStatus;
  notes: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface SpeakerProfileSample {
  id: string;
  speakerProfileId: string;
  fileName: string;
  storagePath: string;
  mimeType: string;
  fileSizeBytes: number;
  durationSeconds: number | null;
  status: SampleStatus;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface SpeakerProfileWithSamples extends SpeakerProfile {
  samples: SpeakerProfileSample[];
}

export interface ProcessingJob {
  id: string;
  recordingId: string;
  jobType: JobType;
  status: JobStatus;
  attemptCount: number;
  errorMessage: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  processingDurationMs: number | null;
  createdAt: string;
  updatedAt: string;
}

export interface RecordingDetail {
  recording: Recording;
  summary: RecordingSummary | null;
  transcription: Transcription | null;
  transcriptionSegments: TranscriptionSegment[];
  speakerDiarizationSegments: SpeakerDiarizationSegment[];
  utteranceSegments: UtteranceSegment[];
  jobs: ProcessingJob[];
  speakerProfiles: SpeakerProfile[];
}

export interface CreatedRecording {
  recording: Recording;
  job: ProcessingJob;
}

export interface TranscriptionOutput {
  language: string | null;
  modelName: string;
  fullText: string;
  diarization?: DiarizationOutput | null;
  segments: Array<{
    startMs: number;
    endMs: number;
    text: string;
    speakerLabel?: string | null;
    speakerClusterId?: string | null;
    speakerConfidence?: number | null;
  }>;
}

export interface DiarizationOutput {
  modelName: string;
  segments: Array<{
    speakerClusterId: string;
    speakerLabel: string;
    startMs: number;
    endMs: number;
    confidence: number | null;
  }>;
}

export interface SpeakerIdentificationMatch {
  diarizationSegmentId: string;
  isTargetPerson: boolean;
  confidence: number | null;
  speakerProfileId: string | null;
}

export interface AudioProgressEvent {
  stage: string;
  message: string;
  percent: number;
}

export interface SearchFilters {
  recordingIds?: string[];
  speakerProfileIds?: string[];
  personNames?: string[];
  locations?: string[];
  targetPersonOnly?: boolean;
  createdFrom?: string | null;
  createdTo?: string | null;
  uploadedFrom?: string | null;
  uploadedTo?: string | null;
}

export interface SearchEvidence {
  index: number;
  recording: {
    id: string;
    title: string;
    fileName: string;
    location: string | null;
    durationSeconds: number | null;
  };
  chunk: {
    id: string;
    text: string;
    startMs: number;
    endMs: number;
    speakerLabels: string[];
    isTargetPerson: boolean;
    matchedSpeakerProfiles: Array<{
      id: string;
      displayName: string;
    }>;
  };
  score: number;
  matchType: "vector" | "keyword" | "hybrid";
  url: string;
}

export interface RagCitation {
  index: number;
  chunkId: string;
  recordingId: string;
  startMs: number;
  endMs: number;
}

export interface RagAnswer {
  text: string;
  citations: RagCitation[];
  notEnoughEvidence: boolean;
}

export interface RagQueryResponse {
  queryId: string;
  query: string;
  answer: RagAnswer | null;
  evidence: SearchEvidence[];
  message?: string;
}
