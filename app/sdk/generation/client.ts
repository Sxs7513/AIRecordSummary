"use client";

import { HttpSseTransport, type GenerationTransportOptions } from "./transport";
import { useGenerationStore } from "./store";
import type { GenerationEvent } from "./types";

const pythonApiOrigin = process.env.NEXT_PUBLIC_PYTHON_API_ORIGIN ?? "http://localhost:8000";

export type GenerationStartResponse = { generation_run_id: string };

export class GenerationStreamClient {
  private readonly transport: HttpSseTransport;
  private readonly pendingEvents = new Map<string, GenerationEvent[]>();
  private animationFrame: number | null = null;
  private flushTimer: number | null = null;

  constructor(private readonly options: GenerationTransportOptions = {}) {
    this.transport = new HttpSseTransport(options);
  }

  connect(runId: string): void {
    const store = useGenerationStore.getState();
    store.setConnection(runId, "connecting");
    this.transport.connect(runId, {
      onEvent: (event) => this.consume(event),
      onConnection: (connection) => useGenerationStore.getState().setConnection(runId, connection),
      onError: () => undefined
    });
  }

  close(runId: string): void {
    this.flush(runId);
    this.transport.close();
    useGenerationStore.getState().setConnection(runId, "closed");
  }

  async cancel(runId: string): Promise<void> {
    await fetch(`${pythonApiOrigin}/api/generations/${encodeURIComponent(runId)}`, {
      method: "DELETE",
      headers: await this.options.getHeaders?.(),
      credentials: "include"
    });
  }

  async start(path: string, payload: unknown): Promise<string> {
    const response = await fetch(`${pythonApiOrigin}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", ...(await this.options.getHeaders?.()) },
      body: JSON.stringify(payload),
      credentials: "include"
    });
    const body = await response.json() as Partial<GenerationStartResponse> & { detail?: string };
    if (!response.ok || typeof body.generation_run_id !== "string") {
      throw new Error(body.detail || `Generation start request failed: ${response.status}`);
    }
    this.connect(body.generation_run_id);
    return body.generation_run_id;
  }

  private consume(event: GenerationEvent): void {
    if (event.type !== "content.delta") {
      this.flush(event.run_id);
      useGenerationStore.getState().consume(event);
      return;
    }
    const events = this.pendingEvents.get(event.run_id) ?? [];
    events.push(event);
    this.pendingEvents.set(event.run_id, events);
    this.scheduleFlush();
  }

  private scheduleFlush(): void {
    if (this.animationFrame === null) {
      this.animationFrame = window.requestAnimationFrame(() => {
        this.animationFrame = null;
        this.flush();
      });
    }
    if (this.flushTimer === null) {
      this.flushTimer = window.setTimeout(() => {
        this.flushTimer = null;
        this.flush();
      }, 100);
    }
  }

  private flush(runId?: string): void {
    const events = runId === undefined
      ? [...this.pendingEvents.values()].flat()
      : this.pendingEvents.get(runId) ?? [];
    if (runId === undefined) this.pendingEvents.clear();
    else this.pendingEvents.delete(runId);
    if (events.length > 0) useGenerationStore.getState().consumeMany(events);

    if (this.pendingEvents.size > 0) return;
    if (this.animationFrame !== null) {
      window.cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    if (this.flushTimer !== null) {
      window.clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }
}
