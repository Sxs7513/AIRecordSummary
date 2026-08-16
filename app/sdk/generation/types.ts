export type TextBlock = {
  type: "text";
  value: string;
};

export type SubMessageStatus = "pending" | "streaming" | "completed" | "failed" | "cancelled";

export type MessageGroup = {
  id: string;
  sub_message_ids: string[];
  primary_sub_message_id: string;
};

export type SubMessage = {
  id: string;
  variant: "original" | "corrected";
  title: string;
  status: SubMessageStatus;
  blocks: TextBlock[];
  sources: Record<string, unknown>[];
  error: string | null;
};

export type AggreMessageBlock = {
  type: "AGGRE_MSG";
  id: string;
  sub_message: {
    message_group: MessageGroup;
    sub_message_list: SubMessage[];
  };
};

export type AdjudicationConfirmationCandidate = {
  id: string;
  expression: string;
  confidence: number;
  source_urls: string[];
};

export type ExpressionTargetSpan = {
  start_char: number;
  end_char: number;
};

export type AdjudicationConfirmationItem = {
  id: string;
  evidence_index: number;
  recording_id: string;
  chunk_id: string;
  start_ms: number;
  end_ms: number;
  original_expression: string;
  target_spans: ExpressionTargetSpan[];
  candidates: AdjudicationConfirmationCandidate[];
  reason: string;
};

export type AdjudicationConfirmationBlock = {
  type: "adjudication_confirmation";
  request_id: string;
  source_generation_id: string;
  items: AdjudicationConfirmationItem[];
};

export type ContentBlock = TextBlock | AggreMessageBlock | AdjudicationConfirmationBlock;

export type GenerationKind = "text";
export type GenerationStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type ConnectionStatus = "connecting" | "connected" | "reconnecting" | "closed";

export type GenerationPhase = { name: string; label: string };
export type GenerationError = { code: string; message: string; retryable?: boolean };

export type GenerationEvent = {
  v: 1;
  run_id: string;
  seq: number;
  type: "conversation.ready" | "snapshot" | "run.status" | "phase" | "content.delta" | "output.final" | "run.error" | "run.cancelled";
  at: string;
  data: Record<string, unknown>;
};

export type GenerationViewState = {
  runId: string;
  kind: GenerationKind | null;
  status: GenerationStatus | null;
  phase: GenerationPhase | null;
  blocks: ContentBlock[];
  sources: Record<string, unknown>[];
  output: Record<string, unknown> | null;
  error: GenerationError | null;
  lastSequence: number;
  connection: ConnectionStatus;
};
