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

export type AnswerVerdict = "direct_answer" | "qualified_answer" | "abstain";

export type AnswerKeyPoint = {
  id: string;
  text: string;
  evidence_ids: string[];
};

export type AnswerAnnotation = {
  expected_verdict: AnswerVerdict;
  reference_answer: string | null;
  key_points: AnswerKeyPoint[];
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
  answer_annotation: AnswerAnnotation | null;
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
  evaluation_case_id: string | null;
  step_result_id: string | null;
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
  error_message: string | null;
  output: {
    candidate_count?: number;
    answer?: string;
    verdict?: AnswerVerdict;
    reason?: string;
    sources?: Array<{
      index: number;
      recording: { id: string; title?: string; fileName: string };
      chunk: { id: string; startMs: number; endMs: number };
    }>;
    answerability_correct?: boolean;
    verdict_correct?: boolean; // Legacy answer_judge_v1/v2 output.
    key_point_results?: Array<{ key_point_id: string; covered: boolean; correct: boolean; reason: string }>;
    claim_results?: Array<{
      claim: string;
      factual_status: "supported" | "unsupported" | "contradicted";
      citation_status: "supported" | "missing" | "misaligned";
      citation_indexes: number[];
      reason: string;
    }>;
    // Legacy answer_judge_v1/v2 output retained for historical run rendering.
    citation_results?: Array<{ claim: string; citation_indexes: number[]; supported: boolean; reason: string }>;
    unsupported_claims?: string[];
    contradictions?: string[];
  };
  details: Record<string, unknown>;
  ranked_results: RankedResult[];
};

export type GoldStageStatus = "hit" | "miss" | "skipped" | "unknown";

export type GoldStageObservation = {
  status: GoldStageStatus;
  best_rank: number | null;
  match_kind?: string | null;
  operation?: string | null;
  channels?: Record<string, GoldStageObservation>;
};

export type EvidenceJourney = {
  diagnostic_version: string;
  evidence_id: string;
  recording_id: string;
  recording_title: string | null;
  source_chunk_id: string | null;
  quote: string;
  start_ms: number;
  end_ms: number;
  relevance: number;
  final_covered: boolean;
  last_visible_node: string | null;
  first_loss_node: string | null;
  stages: Record<string, GoldStageObservation>;
};

export type EvidenceDiagnosisSummary = {
  diagnostic_version: string;
  gold_count: number;
  covered_gold_count: number;
  uncovered_gold_count: number;
  unknown_gold_count: number;
  coverage_status: "full_coverage" | "partial_coverage" | "no_hit" | "not_applicable";
  first_loss_node_counts: Record<string, number>;
};

export type CaseResult = {
  id: string;
  evaluation_case_id: string;
  query: string;
  tags: string[];
  answer_annotation: AnswerAnnotation | null;
  status: string;
  latency_ms: number | null;
  error_message: string | null;
  details: Record<string, unknown>;
  evidence_journeys: EvidenceJourney[];
  steps: StepResult[];
};

export type RagEvalRunDetail = {
  run: RagEvalRun & { pipeline_config: Record<string, unknown> };
  metrics: RagEvalMetric[];
  cases: CaseResult[];
  evidence_diagnosis: EvidenceDiagnosisSummary;
};
