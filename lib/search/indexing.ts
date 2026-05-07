import { getAppConfig } from "../config/app-config";
import { getEmbeddingProvider } from "../audio-transcoding-analysis/embedding";
import { listUtterancesForIndex, replaceRecordingSearchChunks } from "../db/search";
import { buildSearchChunks } from "./chunking";

export async function indexRecordingForSearch(recordingId: string) {
  const config = getAppConfig().search;
  if (!config.embeddingEnabled) {
    return { chunkCount: 0, skipped: true };
  }
  const utterances = await listUtterancesForIndex(recordingId);
  const chunks = buildSearchChunks(utterances, {
    maxDurationMs: config.chunkMaxDurationMs,
    maxTextChars: config.chunkMaxTextChars,
    maxGapMs: config.chunkMaxGapMs
  });
  if (chunks.length === 0) {
    await replaceRecordingSearchChunks(recordingId, [], []);
    return { chunkCount: 0, skipped: false };
  }

  const provider = getEmbeddingProvider();
  const embeddings = await provider.embedTexts(chunks.map((chunk) => chunk.text));
  await replaceRecordingSearchChunks(recordingId, chunks, embeddings);
  return { chunkCount: chunks.length, skipped: false };
}
