"use client";

import { create } from "zustand";
import type { ContentBlock, ConnectionStatus, GenerationError, GenerationEvent, GenerationPhase, GenerationStatus, GenerationViewState } from "./types";

type GenerationStore = {
  runs: Record<string, GenerationViewState>;
  consume: (event: GenerationEvent) => void;
  consumeMany: (events: GenerationEvent[]) => void;
  setConnection: (runId: string, connection: ConnectionStatus) => void;
  clear: (runId: string) => void;
};

const emptyState = (runId: string): GenerationViewState => ({
  runId,
  kind: null,
  status: null,
  phase: null,
  blocks: [],
  sources: [],
  output: null,
  error: null,
  lastSequence: -1,
  connection: "connecting"
});

export function reduceGenerationEvent(current: GenerationViewState | undefined, event: GenerationEvent): GenerationViewState {
  const state = current ?? emptyState(event.run_id);
  if (event.type !== "snapshot" && event.seq <= state.lastSequence) return state;

  if (event.type === "snapshot") {
    const data = event.data;
    return {
      ...state,
      kind: asKind(data.kind),
      status: asStatus(data.status),
      phase: asPhase(data.phase),
      blocks: asBlocks(data.blocks),
      sources: asRecords(data.sources),
      output: asRecord(data.output),
      error: asError(data.error),
      lastSequence: event.seq
    };
  }
  if (event.type === "content.delta") {
    return { ...state, blocks: [...state.blocks, ...asBlocks(event.data.blocks)], lastSequence: event.seq };
  }
  if (event.type === "run.status") {
    return { ...state, status: asStatus(event.data.status), lastSequence: event.seq };
  }
  if (event.type === "phase") {
    return { ...state, phase: asPhase(event.data), lastSequence: event.seq };
  }
  if (event.type === "output.final") {
    const output = asRecord(event.data.output);
    const finalBlocks = output === null ? [] : asBlocks(output.content_blocks);
    return {
      ...state,
      status: "succeeded",
      blocks: finalBlocks.length > 0 ? finalBlocks : state.blocks,
      output,
      sources: asRecords(event.data.sources),
      lastSequence: event.seq
    };
  }
  if (event.type === "run.error") {
    return { ...state, status: "failed", error: asError(event.data), lastSequence: event.seq };
  }
  if (event.type === "run.cancelled") {
    return { ...state, status: "cancelled", lastSequence: event.seq };
  }
  return { ...state, lastSequence: event.seq };
}

export const useGenerationStore = create<GenerationStore>((set) => ({
  runs: {},
  consume: (event) => set((state) => ({ runs: { ...state.runs, [event.run_id]: reduceGenerationEvent(state.runs[event.run_id], event) } })),
  consumeMany: (events) => {
    if (events.length === 0) return;
    set((state) => {
      let runs = state.runs;
      for (const event of events) {
        const current = runs[event.run_id];
        const next = reduceGenerationEvent(current, event);
        if (next === current) continue;
        if (runs === state.runs) runs = { ...runs };
        runs[event.run_id] = next;
      }
      return runs === state.runs ? state : { runs };
    });
  },
  setConnection: (runId, connection) => set((state) => ({
    runs: { ...state.runs, [runId]: { ...(state.runs[runId] ?? emptyState(runId)), connection } }
  })),
  clear: (runId) => set((state) => {
    const { [runId]: _discarded, ...runs } = state.runs;
    return { runs };
  })
}));

function asBlocks(value: unknown): ContentBlock[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is ContentBlock =>
    typeof item === "object" && item !== null && (item as { type?: unknown }).type === "text" && typeof (item as { value?: unknown }).value === "string"
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asRecords(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => asRecord(item) !== null) : [];
}

function asPhase(value: unknown): GenerationPhase | null {
  const data = asRecord(value);
  return data && typeof data.name === "string" && typeof data.label === "string" ? { name: data.name, label: data.label } : null;
}

function asError(value: unknown): GenerationError | null {
  const data = asRecord(value);
  return data && typeof data.code === "string" && typeof data.message === "string"
    ? { code: data.code, message: data.message, retryable: typeof data.retryable === "boolean" ? data.retryable : undefined }
    : null;
}

function asStatus(value: unknown): GenerationStatus | null {
  return value === "queued" || value === "running" || value === "succeeded" || value === "failed" || value === "cancelled" ? value : null;
}

function asKind(value: unknown): GenerationViewState["kind"] {
  return value === "text" ? value : null;
}
