export type AdjudicationDataset = {
  id: string;
  name: string;
  description: string | null;
  case_count: number;
  version_count: number;
  latest_version_number: number | null;
};

export type GoldCorrection = {
  id: string;
  target_evidence_draft_id: string;
  start_char: number;
  end_char: number;
  original_expression: string;
  accepted_expressions: string[];
};

export type AdjudicationEvidence = {
  id: string;
  case_draft_id: string;
  role: "target" | "reference";
  position: number;
  source_recording_id: string;
  source_chunk_id: string | null;
  recording_title: string;
  recording_file_name: string;
  chunk_index: number;
  text: string;
  start_ms: number;
  end_ms: number;
  corrections: GoldCorrection[];
};

export type AdjudicationCase = {
  id: string;
  dataset_id: string;
  query: string;
  tags: string[];
  status: "draft" | "reviewed" | "approved";
  revision: number;
  evidence: AdjudicationEvidence[];
};

export type AdjudicationVersion = {
  id: string;
  dataset_id: string;
  version_number: number;
  status: "building" | "frozen";
  case_count: number;
};

export type AdjudicationDatasetDetail = {
  dataset: AdjudicationDataset;
  cases: AdjudicationCase[];
  versions: AdjudicationVersion[];
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

export type RecordingOption = {
  id: string;
  title: string;
  file_name: string;
  chunk_count: number;
};

export type VersionPreview = {
  case_count: number;
  target_count: number;
  correction_count: number;
  checksum: string;
};

export type AdjudicationRun = {
  id: string;
  dataset_version_id: string;
  dataset_name: string;
  version_number: number;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  total_case_count: number;
  completed_case_count: number;
  failed_case_count: number;
  error_message: string | null;
  created_at: string;
};

export type CorrectionResult = {
  gold_correction_id: string;
  passed: boolean;
  matched_proposal_id: string | null;
  actual_expression: string | null;
  start_char: number;
  end_char: number;
  original_expression: string;
  accepted_expressions: string[];
  details: {
    match_kind?: "exact" | "fuzzy" | "unmatched";
    similarity?: number | string | null;
    matched_accepted_expression?: string | null;
    actual_local_text?: string | null;
    expected_local_text?: string | null;
  };
};

export type PredictionResult = {
  id: string;
  matched_gold_correction_id: string | null;
  proposal_id: string;
  evidence_index: number;
  chunk_id: string;
  start_char: number;
  end_char: number;
  original_expression: string;
  resolved_expression: string;
  match_kind: "exact" | "fuzzy" | "unmatched";
  similarity: number | string | null;
};

export type AdjudicationCaseResult = {
  id: string;
  query: string;
  status: "running" | "succeeded" | "failed";
  latency_ms: number | null;
  token_usage: number;
  agent_state: Record<string, unknown> | null;
  error_type: string | null;
  error_message: string | null;
  corrections: CorrectionResult[];
  predictions: PredictionResult[];
};

export type AdjudicationMetric = {
  metric_name:
    | "correction_precision_strict"
    | "correction_recall_strict"
    | "correction_f1_strict"
    | "correction_precision_relaxed"
    | "correction_recall_relaxed"
    | "correction_f1_relaxed";
  value: number | string;
  passed_count: number;
  sample_count: number;
  details: {
    scope: "strict" | "relaxed";
    true_positive: number;
    false_positive: number;
    false_negative: number;
    exact_count: number;
    fuzzy_count: number;
    gold_count: number;
    prediction_count: number;
  };
};

export type AdjudicationRunDetail = {
  run: AdjudicationRun;
  metrics: AdjudicationMetric[];
  cases: AdjudicationCaseResult[];
};
