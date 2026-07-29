export type Dataset = {
  id: string;
  name: string;
  description: string | null;
  status: "active" | "archived";
  draft_count?: number;
  reviewed_count?: number;
  approved_count?: number;
  asset_count?: number;
  latest_version_number?: number | null;
  created_at: string;
  updated_at: string;
};

export type SourceAsset = {
  id: string;
  dataset_id: string;
  recording_id: string | null;
  file_name: string;
  mime_type: string;
  file_size_bytes: number;
  duration_ms: number;
  annotation_count?: number;
  approved_count?: number;
  created_at: string;
};

export type Annotation = {
  id: string;
  dataset_id: string;
  source_asset_id: string;
  start_ms: number;
  end_ms: number;
  reference_text: string;
  language: string | null;
  status: "draft" | "reviewed" | "approved";
  train_allowed: boolean;
  evaluation_allowed: boolean;
  contains_sensitive_data: boolean;
  revision: number;
  created_at: string;
  updated_at: string;
};

export type DatasetVersion = {
  id: string;
  dataset_id: string;
  version_number: number;
  status: "building" | "frozen";
  normalization_name: string;
  normalization_version: string;
  case_count: number;
  checksum: string | null;
  frozen_at: string | null;
};

export type DatasetDetail = {
  dataset: Dataset;
  assets: SourceAsset[];
  annotations: Annotation[];
  versions: DatasetVersion[];
};

export type EncryptedProjectDataset = {
  id: string;
  file_name: string;
  file_size_bytes: number;
  updated_at: number;
};

export type SplitSummary = {
  group_count: number;
  case_count: number;
  duration_ms: number;
};

export type DatasetPreview = {
  train: SplitSummary;
  validation: SplitSummary;
  test: SplitSummary;
  excluded_count: number;
  checksum: string;
  cases: DatasetPreviewCase[];
};

export type DatasetPreviewCase = {
  annotation: {
    id: string;
    source_asset_id: string;
    start_ms: number;
    end_ms: number;
    reference_text: string;
    language: string | null;
    train_allowed: boolean;
    evaluation_allowed: boolean;
  };
  split: "train" | "validation" | "test";
  normalized_reference_text: string;
};

export type ModelVersion = {
  id: string;
  name: string;
  version: string;
  base_model_name: string;
  status: "candidate" | "validated" | "approved" | "deployed" | "retired";
  training_run_id: string | null;
  created_at: string;
};

export type TrainingRun = {
  id: string;
  dataset_version_id: string;
  base_model_version_id: string;
  dataset_name: string;
  dataset_version_number: number;
  base_model_name: string;
  candidate_model_name: string;
  preset_name: string;
  status: string;
  progress_percent: number | null;
  progress_message: string | null;
  error_message: string | null;
  created_at: string;
};

export type EvaluationRun = {
  id: string;
  dataset_version_id: string;
  dataset_name: string;
  dataset_version_number: number;
  split: "validation" | "test";
  status: string;
  total_case_count: number;
  completed_case_count: number;
  failed_case_count: number;
  models: Array<{ id: string; name: string; version: string; role: string; position: number }>;
  error_message: string | null;
  created_at: string;
};

export type MetricValue = {
  id: string;
  model_version_id: string | null;
  metric_name: string;
  metric_version: string;
  value: number | string;
  sample_count: number | null;
  details: Record<string, unknown>;
};

export type CaseResult = {
  id: string;
  model_version_id: string;
  evaluation_case_id: string;
  source_asset_id: string;
  file_name: string;
  start_ms: number;
  end_ms: number;
  reference_text_raw: string;
  reference_text_normalized: string;
  hypothesis_text_raw: string | null;
  hypothesis_text_normalized: string | null;
  status: "succeeded" | "failed";
  error_message: string | null;
  details: { cer?: { value?: number; operations?: EditOperation[] } } & Record<string, unknown>;
};

export type EditOperation = {
  kind: "equal" | "substitute" | "delete" | "insert";
  reference: string | null;
  hypothesis: string | null;
};

export type EvaluationRunDetail = {
  run: EvaluationRun;
  models: ModelVersion[];
  metrics: MetricValue[];
  case_results: CaseResult[];
};
