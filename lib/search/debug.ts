export function textPreview(value: string, max = 120) {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > max ? `${normalized.slice(0, max)}...` : normalized;
}

export function elapsedMs(startedAt: number) {
  return Date.now() - startedAt;
}

export function ragLog(event: string, data: Record<string, unknown> = {}) {
  console.log(`[rag] ${event}`, data);
}
