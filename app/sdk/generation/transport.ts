"use client";

import type { ConnectionStatus, GenerationEvent } from "./types";

const pythonApiOrigin = process.env.NEXT_PUBLIC_PYTHON_API_ORIGIN ?? "http://localhost:8000";

export type GenerationTransportOptions = {
  /** Supplies future account credentials without coupling the SDK to a specific auth provider. */
  getHeaders?: () => HeadersInit | Promise<HeadersInit>;
  reconnectDelayMs?: number;
};

export type EventHandlers = {
  onEvent: (event: GenerationEvent) => void;
  onConnection: (state: ConnectionStatus) => void;
  onError: (error: Error) => void;
};

/**
 * A fetch-based HTTP SSE client.
 *
 * It deliberately owns frame parsing and Last-Event-ID replay, so future
 * bearer-token authentication only changes getHeaders, not the stream protocol.
 */
export class HttpSseTransport {
  private abortController: AbortController | null = null;
  private closed = true;
  private lastEventId: string | null = null;

  constructor(private readonly options: GenerationTransportOptions = {}) {}

  connect(runId: string, handlers: EventHandlers): void {
    this.close();
    this.closed = false;
    this.lastEventId = null;
    this.abortController = new AbortController();
    void this.run(runId, handlers, this.abortController.signal);
  }

  connectPost(path: string, payload: unknown, handlers: EventHandlers): void {
    this.close();
    this.closed = false;
    this.lastEventId = null;
    this.abortController = new AbortController();
    void this.runPost(path, payload, handlers, this.abortController.signal);
  }

  close(): void {
    this.closed = true;
    this.abortController?.abort();
    this.abortController = null;
  }

  private async run(runId: string, handlers: EventHandlers, signal: AbortSignal): Promise<void> {
    let reconnecting = false;
    while (!this.closed) {
      try {
        handlers.onConnection(reconnecting ? "reconnecting" : "connecting");
        const headers = new Headers(await this.options.getHeaders?.());
        headers.set("Accept", "text/event-stream");
        if (this.lastEventId !== null) headers.set("Last-Event-ID", this.lastEventId);
        const response = await fetch(`${pythonApiOrigin}/api/generations/${encodeURIComponent(runId)}/events`, {
          headers,
          signal,
          credentials: "include"
        });
        if (!response.ok) throw new Error(`Generation SSE request failed: ${response.status}`);
        if (response.body === null) throw new Error("Generation SSE response has no body");
        handlers.onConnection("connected");
        if (await this.readFrames(response.body, handlers)) {
          handlers.onConnection("closed");
          return;
        }
        if (this.closed) return;
        throw new Error("Generation SSE connection ended before a terminal event");
      } catch (reason) {
        if (this.closed || signal.aborted) return;
        handlers.onError(reason instanceof Error ? reason : new Error("Generation SSE connection failed"));
        handlers.onConnection("reconnecting");
        reconnecting = true;
        await delay(this.options.reconnectDelayMs ?? 1_000, signal);
      }
    }
  }

  private async runPost(path: string, payload: unknown, handlers: EventHandlers, signal: AbortSignal): Promise<void> {
    let reconnecting = false;
    while (!this.closed) {
      try {
        handlers.onConnection(reconnecting ? "reconnecting" : "connecting");
        const headers = new Headers(await this.options.getHeaders?.());
        headers.set("Accept", "text/event-stream");
        headers.set("Content-Type", "application/json");
        if (this.lastEventId !== null) headers.set("Last-Event-ID", this.lastEventId);
        const response = await fetch(`${pythonApiOrigin}${path}`, {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
          signal,
          credentials: "include"
        });
        if (!response.ok) throw new Error(`Generation SSE request failed: ${response.status}`);
        if (response.body === null) throw new Error("Generation SSE response has no body");
        handlers.onConnection("connected");
        if (await this.readFrames(response.body, handlers)) {
          handlers.onConnection("closed");
          return;
        }
        if (this.closed) return;
        throw new Error("Generation SSE connection ended before a terminal event");
      } catch (reason) {
        if (this.closed || signal.aborted) return;
        handlers.onError(reason instanceof Error ? reason : new Error("Generation SSE connection failed"));
        handlers.onConnection("reconnecting");
        reconnecting = true;
        await delay(this.options.reconnectDelayMs ?? 1_000, signal);
      }
    }
  }

  private async readFrames(stream: ReadableStream<Uint8Array>, handlers: EventHandlers): Promise<boolean> {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (!this.closed) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let boundary: RegExpMatchArray | null;
        while ((boundary = buffer.match(/\r?\n\r?\n/)) !== null) {
          const index = boundary.index ?? 0;
          const frame = buffer.slice(0, index);
          buffer = buffer.slice(index + boundary[0].length);
          const event = parseFrame(frame);
          if (event === null) continue;
          this.lastEventId = event.id ?? String(event.value.seq);
          handlers.onEvent(event.value);
          if (isTerminal(event.value)) return true;
        }
        if (done) return false;
      }
      return false;
    } finally {
      reader.releaseLock();
    }
  }
}

type ParsedFrame = { id: string | null; value: GenerationEvent };

function parseFrame(frame: string): ParsedFrame | null {
  let id: string | null = null;
  const data: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    const value = separator < 0 ? "" : line.slice(separator + 1).replace(/^ /, "");
    if (field === "id") id = value;
    if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  try {
    const value: unknown = JSON.parse(data.join("\n"));
    if (!isGenerationEvent(value)) return null;
    return { id, value };
  } catch {
    return null;
  }
}

function isGenerationEvent(value: unknown): value is GenerationEvent {
  if (typeof value !== "object" || value === null) return false;
  const event = value as Partial<GenerationEvent>;
  return event.v === 1 && typeof event.run_id === "string" && typeof event.seq === "number" && typeof event.type === "string" && typeof event.at === "string" && typeof event.data === "object";
}

function isTerminal(event: GenerationEvent): boolean {
  return event.type === "output.final" || event.type === "run.error" || event.type === "run.cancelled" ||
    (event.type === "snapshot" && (event.data.status === "succeeded" || event.data.status === "failed" || event.data.status === "cancelled"));
}

function delay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}
