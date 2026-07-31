import type { ObservabilityConversationSnapshot, ObservabilityOverview, ObservabilityRun, ObservabilityRunDetail } from "./types";

const observabilityApiOrigin = process.env.NEXT_PUBLIC_OBSERVABILITY_API_ORIGIN ?? "http://localhost:8003";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${observabilityApiOrigin}${path}`, { credentials: "include", cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Observability API request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const getObservabilityOverview = () => request<ObservabilityOverview>("/api/observability/overview");
export const listObservabilityRuns = () => request<ObservabilityRun[]>("/api/observability/runs?limit=50");
export const getObservabilityRun = (runId: string) =>
  request<ObservabilityRunDetail>(`/api/observability/runs/${encodeURIComponent(runId)}`);
export const getObservabilityRunConversation = (runId: string) =>
  request<ObservabilityConversationSnapshot>(`/api/observability/runs/${encodeURIComponent(runId)}/conversation`);
