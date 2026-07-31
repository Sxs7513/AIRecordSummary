export type RagEvalDataset = {
  id: string;
  name: string;
  description: string | null;
  case_count: number;
  version_count: number;
  latest_version_number: number | null;
  updated_at: string;
};

export type RagEvalEvidence = {
  id: string;
  case_draft_id: string;
  source_recording_id: string;
  source_chunk_id: string | null;
  quote: string;
  start_ms: number;
  end_ms: number;
  relevance: number;
  recording_title: string;
  recording_file_name: string;
};

export type RagEvalCase = {
  id: string;
  dataset_id: string;
  query: string;
  scope: { recording_ids?: string[] };
  tags: string[];
  status: "draft" | "reviewed" | "approved";
  archived_at: string | null;
  revision: number;
  evidence: RagEvalEvidence[];
};

export type RagEvalDatasetVersion = {
  id: string;
  dataset_id: string;
  version_number: number;
  status: "building" | "frozen";
  case_count: number;
  checksum: string | null;
  frozen_at: string | null;
};

export type RagEvalDatasetDetail = {
  dataset: RagEvalDataset;
  cases: RagEvalCase[];
  versions: RagEvalDatasetVersion[];
};

export type SearchChunk = {
  id: string;
  recording_id: string;
  chunk_index: number;
  text: string;
  start_ms: number;
  end_ms: number;
  recording_title: string;
  file_name: string;
  score: number;
};

export type RagEvalRecording = {
  id: string;
  title: string;
  file_name: string;
  created_at: string;
  chunk_count: number;
};

export type VersionPreview = {
  case_count: number;
  evidence_count: number;
  checksum: string;
};

export type RagEvalRun = {
  id: string;
  dataset_version_id: string;
  dataset_name: string;
  version_number: number;
  pipeline_name: string | null;
  config_hash: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  total_case_count: number;
  completed_case_count: number;
  failed_case_count: number;
  error_message: string | null;
  created_at: string;
};

export type RagEvalMetric = {
  id: string;
  scope: "run" | "tag" | "case" | "operation" | "step";
  scope_key: string | null;
  operation: string | null;
  metric_name: string;
  value: number | string;
  sample_count: number | null;
};

export type RankedResult = {
  rank: number;
  recording_id: string;
  recording_title: string;
  source_chunk_id: string | null;
  score: number | string | null;
  matched_relevance: number;
  match_kind: string;
  text: string | null;
  start_ms: number | null;
  end_ms: number | null;
  details: { text?: string; start_ms?: number; end_ms?: number };
};

export type StepResult = {
  id: string;
  operation: string;
  sequence: number;
  status: string;
  latency_ms: number | null;
  output: { candidate_count?: number };
  details: Record<string, unknown>;
  ranked_results: RankedResult[];
};

export type CaseResult = {
  id: string;
  evaluation_case_id: string;
  query: string;
  tags: string[];
  status: string;
  latency_ms: number | null;
  error_message: string | null;
  steps: StepResult[];
};

export type RagEvalRunDetail = {
  run: RagEvalRun & { pipeline_config: Record<string, unknown> };
  metrics: RagEvalMetric[];
  cases: CaseResult[];
};
