import { query, transaction } from "./pool";
import { getAppConfig } from "../config/app-config";
import { elapsedMs, ragLog, textPreview } from "../search/debug";
import { normalizeSearchText, vectorLiteral } from "../search/normalize";
import type { SearchChunkDraft } from "../search/types";
import type { SearchEvidence, SearchFilters, UtteranceSegment } from "../types/models";

export async function getActiveEmbeddingModel() {
  const config = getAppConfig().search;
  const rows = await query<Record<string, any>>(
    `insert into embedding_models (provider, model_name, dimensions, distance_metric, is_active)
     values ($1, $2, $3, 'cosine', true)
     on conflict (provider, model_name, dimensions)
     do update set is_active = true
     returning *`,
    [config.embeddingProvider, config.embeddingModel, config.embeddingDimensions]
  );
  return rows[0] as { id: string; provider: string; model_name: string; dimensions: number };
}

export async function listUtterancesForIndex(recordingId: string): Promise<UtteranceSegment[]> {
  const rows = await query<Record<string, any>>("select * from utterance_segments where recording_id = $1 order by utterance_index", [recordingId]);
  return rows.map((row) => ({
    id: row.id,
    recordingId: row.recording_id,
    utteranceIndex: row.utterance_index,
    startMs: row.start_ms,
    endMs: row.end_ms,
    text: row.text,
    speakerLabel: row.speaker_label,
    speakerClusterId: row.speaker_cluster_id,
    sourceTranscriptionSegmentIds: row.source_transcription_segment_ids ?? [],
    isTargetPerson: row.is_target_person,
    targetPersonConfidence: row.target_person_confidence === null ? null : Number(row.target_person_confidence),
    matchedSpeakerProfileId: row.matched_speaker_profile_id,
    mergeReason: row.merge_reason,
    createdAt: row.created_at.toISOString()
  }));
}

export async function replaceRecordingSearchChunks(recordingId: string, chunks: SearchChunkDraft[], embeddings: number[][]) {
  const model = await getActiveEmbeddingModel();
  if (embeddings.some((embedding) => embedding.length !== Number(model.dimensions))) {
    throw new Error(`Embedding dimensions mismatch. Expected ${model.dimensions}.`);
  }
  if (chunks.length !== embeddings.length) {
    throw new Error("Chunk and embedding counts do not match");
  }

  await transaction(async (client) => {
    await client.query("delete from recording_search_chunks where recording_id = $1 and embedding_model_id = $2", [recordingId, model.id]);
    for (const [index, chunk] of chunks.entries()) {
      await client.query(
        `insert into recording_search_chunks (
          recording_id, embedding_model_id, chunk_index, text, normalized_text, start_ms, end_ms,
          speaker_labels, speaker_cluster_ids, source_utterance_segment_ids, source_transcription_segment_ids,
          is_target_person, matched_speaker_profile_ids, metadata, embedding
        ) values (
          $1, $2, $3, $4, $5, $6, $7,
          $8::text[], $9::text[], $10::uuid[], $11::uuid[],
          $12, $13::uuid[], $14::jsonb, $15::vector
        )`,
        [
          recordingId,
          model.id,
          chunk.chunkIndex,
          chunk.text,
          chunk.normalizedText,
          chunk.startMs,
          chunk.endMs,
          chunk.speakerLabels,
          chunk.speakerClusterIds,
          chunk.sourceUtteranceSegmentIds,
          chunk.sourceTranscriptionSegmentIds,
          chunk.isTargetPerson,
          chunk.matchedSpeakerProfileIds,
          JSON.stringify(chunk.metadata),
          vectorLiteral(embeddings[index])
        ]
      );
    }
  });
}

export async function enqueueEmbeddingIndexing(recordingId: string, options: { force?: boolean } = {}) {
  return transaction(async (client) => {
    if (options.force) {
      await client.query("delete from recording_search_chunks where recording_id = $1", [recordingId]);
    }
    const existing = await client.query(
      `select * from processing_jobs
       where recording_id = $1
         and job_type = 'embedding_indexing'
         and status in ('pending', 'running')
       order by created_at desc
       limit 1`,
      [recordingId]
    );
    if (existing.rows[0]) return existing.rows[0];
    const rows = await client.query(
      `insert into processing_jobs (recording_id, job_type, status)
       values ($1, 'embedding_indexing', 'pending')
       returning *`,
      [recordingId]
    );
    await client.query("update recordings set status = 'processing', error_message = null, updated_at = now() where id = $1", [recordingId]);
    await client.query("select pg_notify('processing_jobs', $1)", [rows.rows[0].id]);
    return rows.rows[0];
  });
}

export async function listCompletedRecordingIds() {
  const rows = await query<{ id: string }>("select id from recordings where status = 'completed' order by uploaded_at desc");
  return rows.map((row) => row.id);
}

function pushFilter(values: unknown[], sql: string, value: unknown) {
  values.push(value);
  return sql.replace("?", `$${values.length}`);
}

export async function searchChunks(input: {
  queryText: string;
  embedding: number[];
  limit: number;
  filters?: SearchFilters;
  latencyStartedAt?: number;
}): Promise<{ queryId: string; evidence: SearchEvidence[] }> {
  const config = getAppConfig().search;
  const startedAt = Date.now();
  const normalizedQuery = normalizeSearchText(input.queryText);
  const values: unknown[] = [vectorLiteral(input.embedding), Math.max(input.limit, config.vectorTopK)];
  const where = ["r.status = 'completed'"];
  const filters = input.filters ?? {};

  if (filters.recordingIds?.length) where.push(pushFilter(values, "c.recording_id = any(?::uuid[])", filters.recordingIds));
  if (filters.speakerProfileIds?.length) where.push(pushFilter(values, "c.matched_speaker_profile_ids && ?::uuid[]", filters.speakerProfileIds));
  if (filters.targetPersonOnly) where.push("c.is_target_person = true");
  if (filters.uploadedFrom) where.push(pushFilter(values, "r.uploaded_at >= ?", filters.uploadedFrom));
  if (filters.uploadedTo) where.push(pushFilter(values, "r.uploaded_at <= ?", filters.uploadedTo));

  ragLog("vector_search.start", {
    queryPreview: textPreview(input.queryText),
    embeddingDimensions: input.embedding.length,
    sqlLimit: values[1],
    finalLimit: input.limit,
    where,
    filters
  });
  const rows = await query<Record<string, any>>(
    `select
       c.id as chunk_id,
       c.recording_id,
       c.text,
       c.start_ms,
       c.end_ms,
       c.speaker_labels,
       c.is_target_person,
       c.matched_speaker_profile_ids,
       r.title,
       r.file_name,
       r.duration_seconds,
       1 - (c.embedding <=> $1::vector) as score,
       coalesce(jsonb_agg(jsonb_build_object('id', sp.id, 'displayName', sp.display_name)) filter (where sp.id is not null), '[]'::jsonb) as matched_profiles
     from recording_search_chunks c
     join recordings r on r.id = c.recording_id
     left join speaker_profiles sp on sp.id = any(c.matched_speaker_profile_ids)
     where ${where.join(" and ")}
     group by c.id, r.id
     order by c.embedding <=> $1::vector
     limit $2`,
    values
  );
  ragLog("vector_search.rows", {
    rawCount: rows.length,
    durationMs: elapsedMs(startedAt),
    topRaw: rows.slice(0, 5).map((row) => ({
      chunkId: row.chunk_id,
      recordingId: row.recording_id,
      title: row.title,
      score: Number(Number(row.score).toFixed(4)),
      time: `${row.start_ms}-${row.end_ms}`,
      textPreview: textPreview(row.text, 80)
    }))
  });

  const kept: SearchEvidence[] = [];
  const perRecording = new Map<string, number>();
  for (const row of rows) {
    const score = Number(row.score);
    if (score < config.minScore) continue;
    const count = perRecording.get(row.recording_id) ?? 0;
    if (count >= 3) continue;
    kept.push({
      index: kept.length + 1,
      recording: {
        id: row.recording_id,
        title: row.title,
        fileName: row.file_name,
        durationSeconds: row.duration_seconds
      },
      chunk: {
        id: row.chunk_id,
        text: row.text,
        startMs: row.start_ms,
        endMs: row.end_ms,
        speakerLabels: row.speaker_labels ?? [],
        isTargetPerson: row.is_target_person,
        matchedSpeakerProfiles: row.matched_profiles ?? []
      },
      score,
      matchType: "vector",
      url: `/recordings/${row.recording_id}?t=${row.start_ms}&chunk=${row.chunk_id}`
    });
    perRecording.set(row.recording_id, count + 1);
    if (kept.length >= input.limit) break;
  }

  const latencyMs = input.latencyStartedAt ? Date.now() - input.latencyStartedAt : null;
  const queryRows = await query<{ id: string }>(
    `insert into search_queries (query_text, normalized_query, filters, result_count, latency_ms)
     values ($1, $2, $3::jsonb, $4, $5)
     returning id`,
    [input.queryText, normalizedQuery, JSON.stringify(filters), kept.length, latencyMs]
  );
  ragLog("vector_search.done", {
    queryId: queryRows[0].id,
    keptCount: kept.length,
    rejectedByMinScore: rows.filter((row) => Number(row.score) < config.minScore).length,
    minScore: config.minScore,
    latencyMs,
    kept: kept.slice(0, 5).map((item) => ({
      index: item.index,
      chunkId: item.chunk.id,
      recordingId: item.recording.id,
      score: Number(item.score.toFixed(4)),
      url: item.url,
      textPreview: textPreview(item.chunk.text, 80)
    }))
  });
  return { queryId: queryRows[0].id, evidence: kept.map((item, index) => ({ ...item, index: index + 1 })) };
}
