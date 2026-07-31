export type TextBlock = {
  type: "text";
  value: string;
};

export type ContentBlock = TextBlock;

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
