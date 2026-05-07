import { query, transaction } from "./pool";
import { getAppConfig } from "../config/app-config";
import { elapsedMs, ragLog, textPreview } from "../search/debug";
import { normalizeSearchText, vectorLiteral } from "../search/normalize";
import type { SearchChunkDraft } from "../search/types";
import type { SearchEvidence, SearchFilters, UtteranceSegment } from "../types/models";

const MAX_RECORDING_SUMMARY_RECORDINGS = 50;
const MAX_RECORDING_SUMMARY_UTTERANCES = 1000;
const MAX_RECORDING_SUMMARY_CHARS = 30000;

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
          $12, $13::uuid[], $14::jsonb, $15::halfvec
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

export async function listRecentCompletedRecordingIds(limit: number) {
  const rows = await query<{ id: string }>(
    `select id
     from recordings
     where status = 'completed'
     order by uploaded_at desc
     limit $1`,
    [Math.max(1, Math.min(10, limit))]
  );
  return rows.map((row) => row.id);
}

export async function listCompletedRecordingIdsByRank(rank: number) {
  const rows = await query<{ id: string }>(
    `select id
     from recordings
     where status = 'completed'
     order by created_at desc
     limit 1 offset $1`,
    [Math.max(0, Math.min(9, rank - 1))]
  );
  return rows.map((row) => row.id);
}

export async function getRecordingSummaryEvidence(input: {
  filters?: SearchFilters;
  recordingLimit?: number | null;
  recordingRank?: number | null;
  maxRecordings?: number;
}): Promise<SearchEvidence[]> {
  const filters = input.filters ?? {};
  const values: unknown[] = [];
  const where = ["status = 'completed'"];
  if (filters.recordingIds?.length) where.push(pushFilter(values, "id = any(?::uuid[])", filters.recordingIds));
  if (filters.locations?.length) {
    const patterns = textPatterns(filters.locations);
    if (patterns.length) where.push(pushFilter(values, "location ilike any(?::text[])", patterns));
  }
  const createdFrom = filters.createdFrom ?? filters.uploadedFrom;
  const createdTo = filters.createdTo ?? filters.uploadedTo;
  if (createdFrom) where.push(pushFilter(values, "created_at >= ?", createdFrom));
  if (createdTo) where.push(pushFilter(values, "created_at < ?", createdTo));

  const limit = input.recordingRank
    ? 1
    : input.recordingLimit
      ? Math.max(1, Math.min(MAX_RECORDING_SUMMARY_RECORDINGS, input.recordingLimit))
      : Math.max(1, Math.min(MAX_RECORDING_SUMMARY_RECORDINGS, input.maxRecordings ?? MAX_RECORDING_SUMMARY_RECORDINGS));
  const offset = input.recordingRank ? Math.max(0, Math.min(9, input.recordingRank - 1)) : 0;
  values.push(limit, offset);
  const rows = await query<Record<string, any>>(
    `select id, title, file_name, location, duration_seconds
     from recordings
     where ${where.join(" and ")}
     order by created_at desc
     limit $${values.length - 1} offset $${values.length}`,
    values
  );
  return getRecordingSummaryEvidenceByRows(rows);
}

async function getRecordingSummaryEvidenceByRows(rows: Array<Record<string, any>>): Promise<SearchEvidence[]> {
  const evidence: SearchEvidence[] = [];
  for (const row of rows) {
    const utterances = await query<Record<string, any>>(
      `select *
       from utterance_segments
       where recording_id = $1
       order by utterance_index
       limit $2`,
      [row.id, MAX_RECORDING_SUMMARY_UTTERANCES]
    );
    const omittedCountRows = await query<{ count: string }>(
      `select greatest(count(*) - $2, 0)::text as count
       from utterance_segments
       where recording_id = $1`,
      [row.id, MAX_RECORDING_SUMMARY_UTTERANCES]
    );
    const omittedCount = Number(omittedCountRows[0]?.count ?? 0);
    const text = utterances
      .map((utterance) => `${utterance.speaker_label || "Unknown Speaker"}: ${utterance.text}`)
      .join("\n")
      .slice(0, MAX_RECORDING_SUMMARY_CHARS);
    const suffix = omittedCount > 0 ? `\n\n[该录音还有 ${omittedCount} 条连续发言未放入本次上下文。]` : "";
    const startMs = Number(utterances[0]?.start_ms ?? 0);
    const endMs = Number(utterances[utterances.length - 1]?.end_ms ?? 0);
    evidence.push({
      index: evidence.length + 1,
      recording: {
        id: row.id,
        title: row.title,
        fileName: row.file_name,
        location: row.location ?? null,
        durationSeconds: row.duration_seconds
      },
      chunk: {
        id: row.id,
        text: text ? `${text}${suffix}` : "该录音暂无连续发言文本。",
        startMs,
        endMs,
        speakerLabels: Array.from(new Set(utterances.map((utterance) => utterance.speaker_label).filter(Boolean))),
        isTargetPerson: utterances.some((utterance) => utterance.is_target_person),
        matchedSpeakerProfiles: []
      },
      score: 1,
      matchType: "hybrid",
      url: `/recordings/${row.id}?t=${startMs}`
    });
  }
  return evidence;
}

export async function getRecentRecordingSummaryEvidence(limit: number): Promise<SearchEvidence[]> {
  const rows = await query<Record<string, any>>(
    `select id, title, file_name, location, duration_seconds
     from recordings
     where status = 'completed'
     order by uploaded_at desc
     limit $1`,
    [Math.max(1, Math.min(5, limit))]
  );
  return getRecordingSummaryEvidenceByRows(rows);
}

export async function getCompletedRecordingSummaryEvidenceByRank(rank: number): Promise<SearchEvidence[]> {
  const rows = await query<Record<string, any>>(
    `select id, title, file_name, location, duration_seconds
     from recordings
     where status = 'completed'
     order by created_at desc
     limit 1 offset $1`,
    [Math.max(0, Math.min(9, rank - 1))]
  );
  return getRecordingSummaryEvidenceByRows(rows);
}

export async function getDateRangeSummaryEvidence(input: { from: string; to: string; limit?: number }): Promise<SearchEvidence[]> {
  const limit = Math.max(1, Math.min(MAX_RECORDING_SUMMARY_RECORDINGS, input.limit ?? MAX_RECORDING_SUMMARY_RECORDINGS));
  const rows = await query<Record<string, any>>(
    `select id, title, file_name, location, duration_seconds
     from recordings
     where status = 'completed'
      and created_at >= $1
      and created_at < $2
     order by created_at desc
     limit $3`,
    [input.from, input.to, limit]
  );
  return getRecordingSummaryEvidenceByRows(rows);
}

function pushFilter(values: unknown[], sql: string, value: unknown) {
  values.push(value);
  return sql.replace("?", `$${values.length}`);
}

function personNamePatterns(names: string[]) {
  return names
    .map((name) => name.trim())
    .filter(Boolean)
    .map((name) => `%${name.replaceAll("%", "\\%").replaceAll("_", "\\_")}%`);
}

function textPatterns(values: string[]) {
  return values
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => `%${value.replaceAll("%", "\\%").replaceAll("_", "\\_")}%`);
}

type VectorSearchRow = {
  chunk_id: string;
  recording_id: string;
  text: string;
  start_ms: number;
  end_ms: number;
  speaker_labels: string[] | null;
  is_target_person: boolean;
  matched_speaker_profile_ids: string[] | null;
  title: string;
  file_name: string;
  location: string | null;
  duration_seconds: number | null;
  score: number | string;
  matched_profiles: Array<{ id: string; displayName: string }> | null;
};

type EvidenceWindow = {
  recordingId: string;
  windowStartMs: number;
  windowEndMs: number;
  primary: VectorSearchRow;
  rows: VectorSearchRow[];
  score: number;
};

function formatEvidenceTime(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60) % 60;
  const hours = Math.floor(totalSeconds / 3600);
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function scoreOf(row: VectorSearchRow) {
  return Number(row.score);
}

function mergeVectorHitWindows(rows: VectorSearchRow[], config: ReturnType<typeof getAppConfig>["search"]): EvidenceWindow[] {
  const windows = rows
    .map((row) => ({
      recordingId: row.recording_id,
      windowStartMs: Math.max(0, Number(row.start_ms) - config.evidenceContextBeforeMs),
      windowEndMs: Number(row.end_ms) + config.evidenceContextAfterMs,
      primary: row,
      rows: [row],
      score: scoreOf(row)
    }))
    .sort((a, b) => (a.recordingId === b.recordingId ? a.windowStartMs - b.windowStartMs : a.recordingId.localeCompare(b.recordingId)));

  const merged: EvidenceWindow[] = [];
  for (const window of windows) {
    const previous = merged[merged.length - 1];
    if (previous && previous.recordingId === window.recordingId && window.windowStartMs <= previous.windowEndMs + config.evidenceMergeGapMs) {
      previous.windowEndMs = Math.max(previous.windowEndMs, window.windowEndMs);
      previous.rows.push(...window.rows);
      if (window.score > previous.score) {
        previous.score = window.score;
        previous.primary = window.primary;
      }
      continue;
    }
    merged.push({ ...window });
  }
  return merged.sort((a, b) => b.score - a.score);
}

function normalizeMatchedProfiles(profiles: unknown): Array<{ id: string; displayName: string }> {
  if (!Array.isArray(profiles)) return [];
  return profiles
    .map((profile) => {
      if (!profile || typeof profile !== "object") return null;
      const item = profile as Record<string, unknown>;
      return typeof item.id === "string" && typeof item.displayName === "string"
        ? { id: item.id, displayName: item.displayName }
        : null;
    })
    .filter((profile): profile is { id: string; displayName: string } => Boolean(profile));
}

function appendEvidenceLine(lines: string[], line: string, maxChars: number) {
  const currentLength = lines.reduce((sum, item) => sum + item.length + 1, 0);
  if (currentLength >= maxChars) return false;
  if (currentLength + line.length <= maxChars) {
    lines.push(line);
    return true;
  }
  const remaining = Math.max(0, maxChars - currentLength - 8);
  if (remaining > 0) lines.push(`${line.slice(0, remaining)}...`);
  return false;
}

async function buildExpandedEvidence(window: EvidenceWindow, index: number, maxChars: number): Promise<SearchEvidence> {
  const utterances = await query<Record<string, any>>(
    `select u.*, sp.id as matched_profile_id, sp.display_name as matched_profile_display_name
     from utterance_segments u
     left join speaker_profiles sp on sp.id = u.matched_speaker_profile_id
     where u.recording_id = $1
       and u.end_ms >= $2
       and u.start_ms <= $3
     order by u.utterance_index`,
    [window.recordingId, window.windowStartMs, window.windowEndMs]
  );

  const lines: string[] = [];
  const speakerLabels = new Set<string>();
  const matchedProfiles = new Map<string, { id: string; displayName: string }>();
  let isTargetPerson = false;
  let startMs = window.primary.start_ms;
  let endMs = window.primary.end_ms;

  for (const utterance of utterances) {
    const speaker = utterance.speaker_label || "Unknown Speaker";
    const line = `[${formatEvidenceTime(utterance.start_ms)}-${formatEvidenceTime(utterance.end_ms)}] ${speaker}: ${utterance.text}`;
    const added = appendEvidenceLine(lines, line, maxChars);
    speakerLabels.add(speaker);
    isTargetPerson ||= Boolean(utterance.is_target_person);
    if (utterance.matched_profile_id && utterance.matched_profile_display_name) {
      matchedProfiles.set(utterance.matched_profile_id, {
        id: utterance.matched_profile_id,
        displayName: utterance.matched_profile_display_name
      });
    }
    if (lines.length === 1) startMs = Number(utterance.start_ms);
    endMs = Number(utterance.end_ms);
    if (!added) break;
  }

  if (lines.length === 0) {
    for (const row of window.rows) {
      const line = `[${formatEvidenceTime(row.start_ms)}-${formatEvidenceTime(row.end_ms)}] ${(row.speaker_labels ?? []).join(", ") || "Unknown Speaker"}: ${row.text}`;
      if (!appendEvidenceLine(lines, line, maxChars)) break;
      for (const speaker of row.speaker_labels ?? []) speakerLabels.add(speaker);
      for (const profile of normalizeMatchedProfiles(row.matched_profiles)) matchedProfiles.set(profile.id, profile);
      isTargetPerson ||= Boolean(row.is_target_person);
      startMs = Math.min(startMs, Number(row.start_ms));
      endMs = Math.max(endMs, Number(row.end_ms));
    }
  }

  return {
    index,
    recording: {
      id: window.primary.recording_id,
      title: window.primary.title,
      fileName: window.primary.file_name,
      location: window.primary.location ?? null,
      durationSeconds: window.primary.duration_seconds
    },
    chunk: {
      id: window.primary.chunk_id,
      text: lines.join("\n"),
      startMs,
      endMs,
      speakerLabels: Array.from(speakerLabels),
      isTargetPerson,
      matchedSpeakerProfiles: Array.from(matchedProfiles.values())
    },
    score: window.score,
    matchType: "vector",
    url: `/recordings/${window.primary.recording_id}?t=${startMs}&chunk=${window.primary.chunk_id}`
  };
}

export async function searchChunks(input: {
  queryText: string;
  embedding: number[];
  limit: number;
  filters?: SearchFilters;
  latencyStartedAt?: number;
}): Promise<{ queryId: string; evidence: SearchEvidence[] }> {
  const config = getAppConfig().search;
  const model = await getActiveEmbeddingModel();
  const startedAt = Date.now();
  const normalizedQuery = normalizeSearchText(input.queryText);
  const values: unknown[] = [vectorLiteral(input.embedding), Math.max(input.limit, config.vectorTopK), model.id];
  const where = ["r.status = 'completed'", "c.embedding_model_id = $3"];
  const filters = input.filters ?? {};

  if (filters.recordingIds?.length) where.push(pushFilter(values, "c.recording_id = any(?::uuid[])", filters.recordingIds));
  if (filters.speakerProfileIds?.length) where.push(pushFilter(values, "c.matched_speaker_profile_ids && ?::uuid[]", filters.speakerProfileIds));
  if (filters.personNames?.length) {
    const patterns = personNamePatterns(filters.personNames);
    if (patterns.length) {
      where.push(
        pushFilter(
          values,
          `(exists (
             select 1
             from unnest(c.speaker_labels) as labels(speaker_label)
             where speaker_label ilike any(?::text[])
           )
           or exists (
             select 1
             from speaker_profiles person_profile
             where person_profile.id = any(c.matched_speaker_profile_ids)
               and person_profile.display_name ilike any(?::text[])
           ))`,
          patterns
        )
      );
      values.push(patterns);
      where[where.length - 1] = where[where.length - 1].replace("?", `$${values.length}`);
    }
  }
  if (filters.locations?.length) {
    const patterns = textPatterns(filters.locations);
    if (patterns.length) where.push(pushFilter(values, "r.location ilike any(?::text[])", patterns));
  }
  if (filters.targetPersonOnly) where.push("c.is_target_person = true");
  const createdFrom = filters.createdFrom ?? filters.uploadedFrom;
  const createdTo = filters.createdTo ?? filters.uploadedTo;
  if (createdFrom) where.push(pushFilter(values, "r.created_at >= ?", createdFrom));
  if (createdTo) where.push(pushFilter(values, "r.created_at < ?", createdTo));

  ragLog("vector_search.start", {
    queryPreview: textPreview(input.queryText),
    embeddingDimensions: input.embedding.length,
    sqlLimit: values[1],
    finalLimit: input.limit,
    where,
    filters
  });
  const rows = await query<VectorSearchRow>(
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
       r.location,
       r.duration_seconds,
       1 - (c.embedding <=> $1::halfvec) as score,
       coalesce(jsonb_agg(jsonb_build_object('id', sp.id, 'displayName', sp.display_name)) filter (where sp.id is not null), '[]'::jsonb) as matched_profiles
     from recording_search_chunks c
     join recordings r on r.id = c.recording_id
     left join speaker_profiles sp on sp.id = any(c.matched_speaker_profile_ids)
     where ${where.join(" and ")}
     group by c.id, r.id
     order by c.embedding <=> $1::halfvec
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

  const candidateRows: VectorSearchRow[] = rows.filter((row) => scoreOf(row) >= config.minScore);
  const windows = mergeVectorHitWindows(candidateRows, config);
  const selectedWindows: EvidenceWindow[] = [];
  const perRecording = new Map<string, number>();
  for (const window of windows) {
    const count = perRecording.get(window.recordingId) ?? 0;
    if (count >= 3) continue;
    selectedWindows.push(window);
    perRecording.set(window.recordingId, count + 1);
    if (selectedWindows.length >= input.limit) break;
  }
  const kept = await Promise.all(
    selectedWindows.map((window, index) => buildExpandedEvidence(window, index + 1, config.evidenceContextMaxChars))
  );

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
    rejectedByMinScore: rows.length - candidateRows.length,
    expandedWindowCount: windows.length,
    evidenceContext: {
      beforeMs: config.evidenceContextBeforeMs,
      afterMs: config.evidenceContextAfterMs,
      maxChars: config.evidenceContextMaxChars,
      mergeGapMs: config.evidenceMergeGapMs
    },
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
