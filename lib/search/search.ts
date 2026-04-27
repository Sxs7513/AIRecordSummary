import { getAppConfig } from "../config/app-config";
import { searchChunks } from "../db/search";
import { elapsedMs, ragLog, textPreview } from "./debug";
import { normalizeSearchText } from "./normalize";
import { getEmbeddingProvider } from "../audio-transcoding-analysis/embedding";
import type { SearchInput, SearchOutput } from "./types";

const queryEmbeddingCache = new Map<string, { value: number[]; expiresAt: number }>();

async function embedQuery(query: string) {
  const config = getAppConfig().search;
  const normalized = normalizeSearchText(query);
  const key = `${config.embeddingProvider}:${config.embeddingModel}:${normalized}`;
  const cached = queryEmbeddingCache.get(key);
  if (cached && cached.expiresAt > Date.now()) {
    ragLog("embedding.cache_hit", {
      provider: config.embeddingProvider,
      model: config.embeddingModel,
      queryPreview: textPreview(query),
      dimensions: cached.value.length
    });
    return cached.value;
  }
  const startedAt = Date.now();
  ragLog("embedding.start", {
    provider: config.embeddingProvider,
    model: config.embeddingModel,
    device: config.embeddingDevice,
    queryPreview: textPreview(query)
  });
  const embedding = await getEmbeddingProvider().embedQuery(query);
  ragLog("embedding.done", {
    provider: config.embeddingProvider,
    model: config.embeddingModel,
    dimensions: embedding.length,
    durationMs: elapsedMs(startedAt)
  });
  if (queryEmbeddingCache.size >= 100) {
    const firstKey = queryEmbeddingCache.keys().next().value;
    if (firstKey) queryEmbeddingCache.delete(firstKey);
  }
  queryEmbeddingCache.set(key, { value: embedding, expiresAt: Date.now() + 10 * 60 * 1000 });
  return embedding;
}

export async function retrieveSearchEvidence(input: SearchInput): Promise<SearchOutput> {
  const config = getAppConfig().search;
  const startedAt = Date.now();
  const query = input.query.trim();
  if (!query) throw new Error("Query is required");
  if (!config.embeddingEnabled) throw new Error("Embedding search is disabled");
  ragLog("retrieve.start", {
    queryPreview: textPreview(query),
    requestedLimit: input.limit ?? null,
    finalTopK: config.finalTopK,
    vectorTopK: config.vectorTopK,
    minScore: config.minScore,
    filters: input.filters ?? {}
  });
  const embedding = await embedQuery(query);
  const result = await searchChunks({
    queryText: query,
    embedding,
    limit: Math.min(input.limit ?? config.finalTopK, config.finalTopK),
    filters: input.filters,
    latencyStartedAt: startedAt
  });
  ragLog("retrieve.done", {
    queryId: result.queryId,
    evidenceCount: result.evidence.length,
    topScores: result.evidence.slice(0, 5).map((item) => Number(item.score.toFixed(4))),
    durationMs: elapsedMs(startedAt)
  });
  return {
    ...result,
    message: result.evidence.length === 0 ? "没有找到足够相关的录音片段" : undefined
  };
}
