export type ObservabilityStatus = "running" | "succeeded" | "failed" | "cancelled" | "abandoned";

export type ObservabilityOverview = {
  run_count: number;
  invocation_count: number;
  failed_invocation_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  average_invocation_elapsed_ms: number | null;
  token_p90_by_operation: Array<{
    operation: string;
    sample_run_count: number;
    invocation_count: number;
    prompt_tokens_p90: number;
    completion_tokens_p90: number;
    total_tokens_p90: number;
  }>;
  start: string;
  end: string;
};

export type ObservabilityRun = {
  generation_run_id: string;
  conversation_id: string | null;
  conversation_navigable: boolean;
  conversation_deleted: boolean;
  started_at: string;
  finished_at: string | null;
  invocation_count: number;
  failed_invocation_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  status: ObservabilityStatus;
};

export type ExecutionSpan = {
  id: string;
  parent_span_id: string | null;
  operation: string;
  operation_version: string;
  attempt: number;
  status: ObservabilityStatus;
  started_at: string;
  finished_at: string | null;
  elapsed_ms: number | null;
  error_type: string | null;
  metadata: Record<string, unknown>;
};

export type ModelInvocation = {
  id: string;
  span_id: string | null;
  operation: string;
  provider: string;
  model: string | null;
  stream: boolean;
  status: ObservabilityStatus;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  usage_source: "provider" | "local_tokenizer" | "estimated" | "unavailable";
  finish_reason: string | null;
  provider_request_id: string | null;
  error_type: string | null;
  started_at: string;
  finished_at: string | null;
  elapsed_ms: number | null;
};

export type ObservabilityRunDetail = {
  generation_run_id: string;
  spans: ExecutionSpan[];
  model_invocations: ModelInvocation[];
};

export type ObservabilityConversationMessage = {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  sequence: number;
  content_blocks: Array<{ type: "text"; value: string }>;
  sources: Record<string, unknown>[];
  generation_run_id: string | null;
  status: string;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type ObservabilityConversationSnapshot = {
  conversation: {
    id: string;
    title: string;
    owner_user_id: string | null;
    archived_at: string | null;
    created_at: string;
    updated_at: string;
    deleted: boolean;
  };
  messages: ObservabilityConversationMessage[];
};
