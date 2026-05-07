import { query, transaction } from "./pool";
import { getAppConfig } from "../config/app-config";
import { redactSecrets } from "../security/redact";
import type {
  DiarizationOutput,
  CreatedRecording,
  ProcessingJob,
  Recording,
  RecordingDetail,
  RecordingSummary,
  SpeakerDiarizationSegment,
  SpeakerIdentificationMatch,
  Transcription,
  TranscriptionOutput,
  TranscriptionSegment,
  UtteranceSegment
} from "../types/models";

function rowRecording(row: Record<string, any>): Recording {
  return {
    id: row.id,
    title: row.title,
    fileName: row.file_name,
    storagePath: row.storage_path,
    location: row.location ?? null,
    mimeType: row.mime_type,
    fileSizeBytes: Number(row.file_size_bytes),
    durationSeconds: row.duration_seconds,
    processingDurationMs: row.processing_duration_ms === null || row.processing_duration_ms === undefined ? null : Number(row.processing_duration_ms),
    status: row.status,
    errorMessage: row.error_message ? redactSecrets(row.error_message) : null,
    uploadedAt: row.uploaded_at.toISOString(),
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString()
  };
}

function rowTranscription(row: Record<string, any>): Transcription {
  return {
    id: row.id,
    recordingId: row.recording_id,
    language: row.language,
    modelName: row.model_name,
    fullText: row.full_text,
    segmentCount: row.segment_count,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString()
  };
}

function rowRecordingSummary(row: Record<string, any>): RecordingSummary {
  return {
    id: row.id,
    recordingId: row.recording_id,
    provider: row.provider,
    modelName: row.model_name,
    summaryText: row.summary_text,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString()
  };
}

function rowTranscriptionSegment(row: Record<string, any>): TranscriptionSegment {
  return {
    id: row.id,
    recordingId: row.recording_id,
    transcriptionId: row.transcription_id,
    segmentIndex: row.segment_index,
    startMs: row.start_ms,
    endMs: row.end_ms,
    text: row.text,
    speakerLabel: row.speaker_label,
    speakerClusterId: row.speaker_cluster_id,
    speakerConfidence: row.speaker_confidence === null ? null : Number(row.speaker_confidence),
    isTargetPerson: row.is_target_person,
    targetPersonConfidence: row.target_person_confidence === null ? null : Number(row.target_person_confidence),
    diarizationSegmentId: row.diarization_segment_id,
    matchedSpeakerProfileId: row.matched_speaker_profile_id,
    createdAt: row.created_at.toISOString()
  };
}

function rowDiarizationSegment(row: Record<string, any>): SpeakerDiarizationSegment {
  return {
    id: row.id,
    recordingId: row.recording_id,
    speakerClusterId: row.speaker_cluster_id,
    speakerLabel: row.speaker_label,
    startMs: row.start_ms,
    endMs: row.end_ms,
    confidence: row.confidence === null ? null : Number(row.confidence),
    isTargetPerson: row.is_target_person,
    targetPersonConfidence: row.target_person_confidence === null ? null : Number(row.target_person_confidence),
    matchedSpeakerProfileId: row.matched_speaker_profile_id,
    createdAt: row.created_at.toISOString()
  };
}

function rowUtteranceSegment(row: Record<string, any>): UtteranceSegment {
  return {
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
  };
}

function rowJob(row: Record<string, any>): ProcessingJob {
  return {
    id: row.id,
    recordingId: row.recording_id,
    jobType: row.job_type,
    status: row.status,
    attemptCount: row.attempt_count,
    errorMessage: row.error_message ? redactSecrets(row.error_message) : null,
    startedAt: row.started_at?.toISOString() ?? null,
    finishedAt: row.finished_at?.toISOString() ?? null,
    processingDurationMs: row.processing_duration_ms === null || row.processing_duration_ms === undefined ? null : Number(row.processing_duration_ms),
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString()
  };
}

async function ensureRecordingLocationColumn(client: { query: (sql: string, values?: unknown[]) => Promise<any> }) {
  await client.query("alter table recordings add column if not exists location text");
}

export async function createRecording(input: {
  title: string;
  fileName: string;
  storagePath: string;
  mimeType: string;
  fileSizeBytes: number;
  durationSeconds?: number | null;
}): Promise<CreatedRecording> {
  console.log("[db] creating recording", {
    title: input.title,
    fileName: input.fileName,
    storagePath: input.storagePath,
    fileSizeBytes: input.fileSizeBytes
  });
  return transaction(async (client) => {
    const recordingRows = await client.query(
      `insert into recordings (
        title, file_name, storage_path, mime_type, file_size_bytes, duration_seconds, status
      ) values ($1, $2, $3, $4, $5, $6, 'uploaded')
      returning *`,
      [input.title, input.fileName, input.storagePath, input.mimeType, input.fileSizeBytes, input.durationSeconds ?? null]
    );
    const recording = rowRecording(recordingRows.rows[0]);
    const jobRows = await client.query(
      `insert into processing_jobs (recording_id, job_type, status)
       values ($1, 'transcription', 'pending')
       returning *`,
      [recording.id]
    );
    const job = rowJob(jobRows.rows[0]);
    await client.query("select pg_notify('processing_jobs', $1)", [job.id]);
    console.log("[db] recording and initial job committed", {
      recordingId: recording.id,
      jobId: job.id,
      jobType: job.jobType
    });
    return { recording, job };
  });
}

export async function listRecordings(params: {
  status?: string | null;
  page?: number;
  pageSize?: number;
}): Promise<{ items: Recording[]; total: number; stats: Record<string, number>; page: number; pageSize: number }> {
  const page = Math.max(1, params.page ?? 1);
  const pageSize = Math.min(50, Math.max(1, params.pageSize ?? 10));
  const offset = (page - 1) * pageSize;
  const values: unknown[] = [];
  const where = params.status && params.status !== "all" ? `where status = $${values.push(params.status)}` : "";
  const countRows = await query<{ count: string }>(`select count(*) from recordings ${where}`, values);
  const rows = await query<Record<string, any>>(
    `select recordings.*,
            coalesce(sum(processing_jobs.processing_duration_ms), 0)::integer as processing_duration_ms
       from recordings
       left join processing_jobs on processing_jobs.recording_id = recordings.id
       ${where}
       group by recordings.id
       order by recordings.created_at desc
       limit $${values.length + 1} offset $${values.length + 2}`,
    [...values, pageSize, offset]
  );
  const statsRows = await query<{ status: string; count: string }>("select status, count(*) from recordings group by status");
  const stats: Record<string, number> = { uploaded: 0, processing: 0, completed: 0, failed: 0 };
  for (const row of statsRows) {
    stats[row.status] = Number(row.count);
  }
  return {
    items: rows.map(rowRecording),
    total: Number(countRows[0]?.count ?? 0),
    page,
    pageSize,
    stats
  };
}

export async function getRecordingDetail(id: string): Promise<RecordingDetail | null> {
  const recordingRows = await query<Record<string, any>>("select * from recordings where id = $1", [id]);
  if (recordingRows.length === 0) return null;

  const [summaryRows, transcriptionRows, segmentRows, diarizationRows, utteranceRows, jobRows, profileRows] = await Promise.all([
    query<Record<string, any>>("select * from recording_summaries where recording_id = $1", [id]),
    query<Record<string, any>>("select * from transcriptions where recording_id = $1", [id]),
    query<Record<string, any>>("select * from transcription_segments where recording_id = $1 order by segment_index", [id]),
    query<Record<string, any>>("select * from speaker_diarization_segments where recording_id = $1 order by start_ms", [id]),
    query<Record<string, any>>("select * from utterance_segments where recording_id = $1 order by utterance_index", [id]),
    query<Record<string, any>>("select * from processing_jobs where recording_id = $1 order by created_at", [id]),
    query<Record<string, any>>("select * from speaker_profiles order by created_at desc")
  ]);

  return {
    recording: rowRecording(recordingRows[0]),
    summary: summaryRows[0] ? rowRecordingSummary(summaryRows[0]) : null,
    transcription: transcriptionRows[0] ? rowTranscription(transcriptionRows[0]) : null,
    transcriptionSegments: segmentRows.map(rowTranscriptionSegment),
    speakerDiarizationSegments: diarizationRows.map(rowDiarizationSegment),
    utteranceSegments: utteranceRows.map(rowUtteranceSegment),
    jobs: jobRows.map(rowJob),
    speakerProfiles: profileRows.map((row) => ({
      id: row.id,
      displayName: row.display_name,
      status: row.status,
      notes: row.notes,
      createdAt: row.created_at.toISOString(),
      updatedAt: row.updated_at.toISOString()
    }))
  };
}

export async function deleteRecording(id: string): Promise<Recording | null> {
  return transaction(async (client) => {
    const rows = await client.query<Record<string, any>>("select * from recordings where id = $1 for update", [id]);
    if (rows.rowCount === 0) return null;

    const recording = rowRecording(rows.rows[0]);
    await client.query("delete from recordings where id = $1", [id]);
    console.log("[db] recording deleted", {
      recordingId: recording.id,
      storagePath: recording.storagePath
    });
    return recording;
  });
}

export async function getNextPendingJob(excludeRecordingIds: string[] = []): Promise<ProcessingJob | null> {
  return transaction(async (client) => {
    const rows = await client.query(
      `select pending_jobs.* from processing_jobs pending_jobs
       where pending_jobs.status = 'pending'
         and not (pending_jobs.recording_id = any($1::uuid[]))
         and not exists (
           select 1 from processing_jobs running_jobs
           where running_jobs.recording_id = pending_jobs.recording_id
             and running_jobs.status = 'running'
         )
         and pg_try_advisory_xact_lock(hashtextextended(pending_jobs.recording_id::text, 0))
       order by pending_jobs.created_at
       for update skip locked
       limit 1`,
      [excludeRecordingIds]
    );
    if (rows.rowCount === 0) return null;
    const job = rows.rows[0];
    const updatedRows = await client.query(
      `update processing_jobs
       set status = 'running', attempt_count = attempt_count + 1, started_at = now(), updated_at = now()
       where id = $1 returning *`,
      [job.id]
    );
    await client.query("update recordings set status = 'processing', updated_at = now() where id = $1", [job.recording_id]);
    const claimedJob = rowJob(updatedRows.rows[0]);
    console.log("[jobs] claimed pending job", {
      jobId: claimedJob.id,
      recordingId: claimedJob.recordingId,
      jobType: claimedJob.jobType,
      attemptCount: claimedJob.attemptCount
    });
    return claimedJob;
  });
}

export async function completeJob(jobId: string, options: { nextJobType?: string; recordingStatus?: string } = {}): Promise<void> {
  await transaction(async (client) => {
    const rows = await client.query(
      `update processing_jobs
       set status = 'completed',
           error_message = null,
           finished_at = now(),
           processing_duration_ms = case
             when started_at is null then null
             else greatest(0, floor(extract(epoch from (now() - started_at)) * 1000)::integer)
           end,
           updated_at = now()
       where id = $1
       returning *`,
      [jobId]
    );
    const job = rows.rows[0];
    if (!job) throw new Error("Job not found");
    console.log("[jobs] completed job", {
      jobId,
      recordingId: job.recording_id,
      nextJobType: options.nextJobType,
      recordingStatus: options.recordingStatus
    });
    if (options.nextJobType) {
      const existingNextJobRows = await client.query(
        `select * from processing_jobs
         where recording_id = $1
           and job_type = $2
         order by created_at desc
         limit 1`,
        [job.recording_id, options.nextJobType]
      );
      let nextJob = existingNextJobRows.rows[0];
      if (nextJob?.status === "pending" || nextJob?.status === "running") {
        await client.query("select pg_notify('processing_jobs', $1)", [nextJob.id]);
      } else if (nextJob) {
        const resetRows = await client.query(
          `update processing_jobs
           set status = 'pending',
               error_message = null,
               started_at = null,
               finished_at = null,
               processing_duration_ms = null,
               updated_at = now()
           where id = $1
           returning *`,
          [nextJob.id]
        );
        nextJob = resetRows.rows[0];
        await client.query("select pg_notify('processing_jobs', $1)", [nextJob.id]);
      } else {
        const nextJobRows = await client.query("insert into processing_jobs (recording_id, job_type, status) values ($1, $2, 'pending') returning *", [job.recording_id, options.nextJobType]);
        nextJob = nextJobRows.rows[0];
        await client.query("select pg_notify('processing_jobs', $1)", [nextJob.id]);
      }
      console.log("[jobs] enqueued next job", {
        previousJobId: jobId,
        nextJobId: nextJob.id,
        recordingId: job.recording_id,
        nextJobType: options.nextJobType,
        reused: Boolean(existingNextJobRows.rows[0])
      });
    }
    if (options.recordingStatus) {
      await client.query("update recordings set status = $2, error_message = null, updated_at = now() where id = $1", [job.recording_id, options.recordingStatus]);
    }
  });
}

export async function failJob(jobId: string, error: unknown): Promise<void> {
  await transaction(async (client) => {
    const message = redactSecrets(error instanceof Error ? error.message : String(error));
    console.error("[jobs] failed job", {
      jobId,
      error: message
    });
    const rows = await client.query(
      `update processing_jobs
       set status = 'failed',
           error_message = $2,
           finished_at = now(),
           processing_duration_ms = case
             when started_at is null then null
             else greatest(0, floor(extract(epoch from (now() - started_at)) * 1000)::integer)
           end,
           updated_at = now()
       where id = $1
       returning *`,
      [jobId, message]
    );
    const job = rows.rows[0];
    if (job) {
      await client.query("update recordings set status = 'failed', error_message = $2, updated_at = now() where id = $1", [job.recording_id, message]);
    }
  });
}

export async function retryFailedJob(jobId: string): Promise<ProcessingJob> {
  return transaction(async (client) => {
    const rows = await client.query(
      `update processing_jobs
       set status = 'pending',
           error_message = null,
           started_at = null,
           finished_at = null,
           processing_duration_ms = null,
           updated_at = now()
       where id = $1
         and (
           status = 'failed'
           or (job_type = 'text_correction' and status in ('completed', 'failed'))
           or (job_type = 'embedding_indexing' and status in ('completed', 'failed'))
           or (job_type = 'summary' and status in ('completed', 'failed'))
         )
       returning *`,
      [jobId]
    );
    if (rows.rowCount === 0) {
      throw new Error("Job not found or not retryable");
    }
    const job = rowJob(rows.rows[0]);
    await client.query("update recordings set status = 'processing', error_message = null, updated_at = now() where id = $1", [job.recordingId]);
    await client.query("select pg_notify('processing_jobs', $1)", [job.id]);
    console.log("[jobs] retry queued", {
      jobId: job.id,
      recordingId: job.recordingId,
      jobType: job.jobType,
      attemptCount: job.attemptCount
    });
    return job;
  });
}

export async function updateRecordingSpeakerLabels(recordingId: string, mappings: Array<{ from: string; to: string }>): Promise<void> {
  const cleaned = mappings
    .map((mapping) => ({ from: mapping.from.trim(), to: mapping.to.trim() }))
    .filter((mapping) => mapping.from && mapping.to && mapping.from !== mapping.to);
  if (cleaned.length === 0) return;

  await transaction(async (client) => {
    const recordingRows = await client.query("select id from recordings where id = $1 for update", [recordingId]);
    if (recordingRows.rowCount === 0) {
      throw new Error("Recording not found");
    }

    for (const mapping of cleaned) {
      await client.query(
        `update speaker_diarization_segments
         set speaker_label = $3
         where recording_id = $1 and speaker_label = $2`,
        [recordingId, mapping.from, mapping.to]
      );
      await client.query(
        `update transcription_segments
         set speaker_label = $3
         where recording_id = $1 and speaker_label = $2`,
        [recordingId, mapping.from, mapping.to]
      );
      await client.query(
        `update utterance_segments
         set speaker_label = $3
         where recording_id = $1 and speaker_label = $2`,
        [recordingId, mapping.from, mapping.to]
      );
    }

    await client.query("delete from recording_search_chunks where recording_id = $1", [recordingId]);
    if (getAppConfig().search.embeddingEnabled) {
      const existing = await client.query(
        `select id, status from processing_jobs
         where recording_id = $1
           and job_type = 'embedding_indexing'
           and status in ('pending', 'running', 'completed', 'failed')
         order by created_at desc
         limit 1`,
        [recordingId]
      );
      if (existing.rows[0]?.status === "pending" || existing.rows[0]?.status === "running") {
        await client.query("select pg_notify('processing_jobs', $1)", [existing.rows[0].id]);
      } else if (existing.rows[0]) {
        await client.query(
          `update processing_jobs
           set status = 'pending',
               error_message = null,
               started_at = null,
               finished_at = null,
               processing_duration_ms = null,
               updated_at = now()
           where id = $1`,
          [existing.rows[0].id]
        );
        await client.query("select pg_notify('processing_jobs', $1)", [existing.rows[0].id]);
      } else {
        const jobRows = await client.query(
          `insert into processing_jobs (recording_id, job_type, status)
           values ($1, 'embedding_indexing', 'pending')
           returning id`,
          [recordingId]
        );
        await client.query("select pg_notify('processing_jobs', $1)", [jobRows.rows[0].id]);
      }
      await client.query("update recordings set status = 'processing', error_message = null, updated_at = now() where id = $1", [recordingId]);
    } else {
      await client.query("update recordings set updated_at = now() where id = $1", [recordingId]);
    }
  });
  console.log("[recordings] speaker labels updated", {
    recordingId,
    mappings: cleaned
  });
}

export async function updateRecordingLocation(recordingId: string, location: string | null): Promise<void> {
  const value = location?.trim() || null;
  await transaction(async (client) => {
    await ensureRecordingLocationColumn(client);
    const rows = await client.query(
      `update recordings
       set location = $2,
           updated_at = now()
       where id = $1
       returning id`,
      [recordingId, value]
    );
    if (rows.rowCount === 0) {
      throw new Error("Recording not found");
    }
  });
  console.log("[recordings] location updated", {
    recordingId,
    location: value
  });
}

export async function updateRecordingTitle(recordingId: string, title: string): Promise<void> {
  const value = title.trim();
  if (!value) throw new Error("Title is required");
  await transaction(async (client) => {
    const rows = await client.query(
      `update recordings
       set title = $2,
           updated_at = now()
       where id = $1
       returning id`,
      [recordingId, value]
    );
    if (rows.rowCount === 0) {
      throw new Error("Recording not found");
    }
  });
  console.log("[recordings] title updated", {
    recordingId,
    title: value
  });
}

export async function saveRecordingSummary(input: {
  recordingId: string;
  provider: "local_llm" | "deepseek_api";
  modelName: string;
  summaryText: string;
}): Promise<void> {
  const summaryText = input.summaryText.trim();
  if (!summaryText) throw new Error("Summary text is empty");
  await query(
    `insert into recording_summaries (recording_id, provider, model_name, summary_text)
     values ($1, $2, $3, $4)
     on conflict (recording_id)
     do update set
       provider = excluded.provider,
       model_name = excluded.model_name,
       summary_text = excluded.summary_text,
       updated_at = now()`,
    [input.recordingId, input.provider, input.modelName, summaryText]
  );
  console.log("[recordings] summary saved", {
    recordingId: input.recordingId,
    provider: input.provider,
    modelName: input.modelName,
    summaryLength: summaryText.length
  });
}

export async function saveTranscription(recordingId: string, output: TranscriptionOutput): Promise<void> {
  console.log("[transcription] saving output", {
    recordingId,
    modelName: output.modelName,
    language: output.language,
    segmentCount: output.segments.length,
    textLength: output.fullText.length
  });
  await transaction(async (client) => {
    await client.query("delete from recording_search_chunks where recording_id = $1", [recordingId]);
    await client.query("delete from utterance_segments where recording_id = $1", [recordingId]);
    await client.query("delete from transcriptions where recording_id = $1", [recordingId]);
    const transcriptionRows = await client.query(
      `insert into transcriptions (recording_id, language, model_name, full_text, segment_count)
       values ($1, $2, $3, $4, $5)
       returning *`,
      [recordingId, output.language, output.modelName, output.fullText, output.segments.length]
    );
    const transcriptionId = transcriptionRows.rows[0].id;
    for (const [index, segment] of output.segments.entries()) {
      await client.query(
        `insert into transcription_segments (
          recording_id, transcription_id, segment_index, start_ms, end_ms, text,
          speaker_label, speaker_cluster_id, speaker_confidence
        ) values ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
        [
          recordingId,
          transcriptionId,
          index,
          segment.startMs,
          segment.endMs,
          segment.text,
          segment.speakerLabel ?? null,
          segment.speakerClusterId ?? null,
          segment.speakerConfidence ?? null
        ]
      );
    }
  });
}

export async function saveDiarization(recordingId: string, output: DiarizationOutput): Promise<void> {
  console.log("[diarization] saving output", {
    recordingId,
    modelName: output.modelName,
    segmentCount: output.segments.length
  });
  await transaction(async (client) => {
    await client.query("delete from recording_search_chunks where recording_id = $1", [recordingId]);
    await client.query("delete from utterance_segments where recording_id = $1", [recordingId]);
    await client.query("delete from speaker_diarization_segments where recording_id = $1", [recordingId]);
    for (const segment of output.segments) {
      await client.query(
        `insert into speaker_diarization_segments (
          recording_id, speaker_cluster_id, speaker_label, start_ms, end_ms, confidence
        ) values ($1, $2, $3, $4, $5, $6)`,
        [recordingId, segment.speakerClusterId, segment.speakerLabel, segment.startMs, segment.endMs, segment.confidence]
      );
    }
  });
}

export async function alignTranscriptionSegments(recordingId: string): Promise<void> {
  console.log("[diarization] aligning transcription segments", { recordingId });
  let alignedCount = 0;
  await transaction(async (client) => {
    const textRows = await client.query("select * from transcription_segments where recording_id = $1 order by segment_index", [recordingId]);
    const speakerRows = await client.query("select * from speaker_diarization_segments where recording_id = $1 order by start_ms", [recordingId]);
    for (const textSegment of textRows.rows) {
      const match = speakerRows.rows
        .map((speakerSegment) => ({
          speakerSegment,
          overlap: Math.max(0, Math.min(textSegment.end_ms, speakerSegment.end_ms) - Math.max(textSegment.start_ms, speakerSegment.start_ms))
        }))
        .sort((a, b) => b.overlap - a.overlap)[0];
      if (match?.overlap > 0) {
        alignedCount += 1;
        await client.query(
          `update transcription_segments
           set speaker_label = $2,
               speaker_cluster_id = $3,
               speaker_confidence = $4,
               diarization_segment_id = $5
           where id = $1`,
          [textSegment.id, match.speakerSegment.speaker_label, match.speakerSegment.speaker_cluster_id, match.speakerSegment.confidence, match.speakerSegment.id]
        );
      }
    }
  });
  console.log("[diarization] alignment complete", { recordingId, alignedCount });
}

function speakerKey(segment: Record<string, any>): string {
  return segment.speaker_cluster_id || segment.speaker_label || "unknown";
}

function utteranceSpeakerKey(utterance: { speakerClusterId: string | null; speakerLabel: string | null }): string {
  return utterance.speakerClusterId || utterance.speakerLabel || "unknown";
}

function joinUtteranceText(parts: string[]) {
  const sentenceEndPattern = /[。！？!?；;：:]$/;
  return parts
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part, index, items) => {
      if (index === items.length - 1 || sentenceEndPattern.test(part)) return part;
      return `${part}。`;
    })
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

function estimateJoinedUtteranceTextLength(parts: string[]): number {
  return joinUtteranceText(parts).length;
}

function canAppendToUtterance(
  utterance: { startMs: number; textParts: string[]; speakerClusterId: string | null; speakerLabel: string | null },
  segment: Record<string, any>,
  limits: { maxDurationMs: number; maxTextChars: number }
): boolean {
  if (utteranceSpeakerKey(utterance) !== speakerKey(segment)) return false;
  const nextDurationMs = Number(segment.end_ms) - utterance.startMs;
  if (limits.maxDurationMs > 0 && nextDurationMs > limits.maxDurationMs) return false;
  if (limits.maxTextChars > 0 && estimateJoinedUtteranceTextLength([...utterance.textParts, segment.text]) > limits.maxTextChars) return false;
  return true;
}

function normalizeConfidence(value: number | null): number | null {
  if (value === null || !Number.isFinite(value)) return null;
  return Math.max(0, Math.min(1, value));
}

interface UtteranceDraft {
  startMs: number;
  endMs: number;
  text: string;
  speakerLabel: string | null;
  speakerClusterId: string | null;
  sourceIds: string[];
  isTargetPerson: boolean;
  targetPersonConfidence: number | null;
  matchedSpeakerProfileId: string | null;
  mergeReason: string;
}

interface UtteranceMergeCandidate {
  groupId: string;
  speakerLabel: string | null;
  segments: Array<{
    id: string;
    startMs: number;
    endMs: number;
    text: string;
  }>;
}

interface UtteranceMergeResult {
  groupId: string;
  groups: Array<{
    sourceIds: string[];
    text: string;
  }>;
}

function canAppendToLlmMergeGroup(current: UtteranceDraft[], next: UtteranceDraft, limits: { maxGapMs: number; maxDurationMs: number; maxTextChars: number }) {
  const first = current[0];
  const previous = current[current.length - 1];
  if (!first || !previous) return false;
  if (utteranceSpeakerKey(first) !== utteranceSpeakerKey(next)) return false;
  if (first.isTargetPerson !== next.isTargetPerson) return false;
  if ((first.matchedSpeakerProfileId ?? null) !== (next.matchedSpeakerProfileId ?? null)) return false;
  if (next.startMs - previous.endMs > limits.maxGapMs) return false;
  if (next.endMs - first.startMs > limits.maxDurationMs) return false;
  const textLength = [...current, next].reduce((sum, utterance) => sum + utterance.text.length, 0);
  if (textLength > limits.maxTextChars) return false;
  return true;
}

function buildLlmMergeCandidates(utterances: UtteranceDraft[], limits: { maxGapMs: number; maxDurationMs: number; maxTextChars: number }) {
  const candidates: UtteranceMergeCandidate[] = [];
  let group: UtteranceDraft[] = [];

  const flush = () => {
    if (group.length > 1) {
      candidates.push({
        groupId: `group-${candidates.length}`,
        speakerLabel: group[0].speakerLabel,
        segments: group.map((utterance, index) => ({
          id: `${candidates.length}:${index}`,
          startMs: utterance.startMs,
          endMs: utterance.endMs,
          text: utterance.text
        }))
      });
    }
    group = [];
  };

  for (const utterance of utterances) {
    if (group.length === 0) {
      group = [utterance];
      continue;
    }
    if (canAppendToLlmMergeGroup(group, utterance, limits)) {
      group.push(utterance);
    } else {
      flush();
      group = [utterance];
    }
  }
  flush();
  return candidates;
}

function applyLlmMergeResults(utterances: UtteranceDraft[], candidates: UtteranceMergeCandidate[], results: UtteranceMergeResult[]): UtteranceDraft[] {
  if (candidates.length === 0 || results.length === 0) return utterances;

  const candidateByGroupId = new Map(candidates.map((candidate) => [candidate.groupId, candidate]));
  const resultByGroupId = new Map(results.map((result) => [result.groupId, result]));
  const candidateStartByGroupId = new Map<string, number>();
  let searchStart = 0;
  for (const candidate of candidates) {
    const first = candidate.segments[0];
    const index = utterances.findIndex((utterance, utteranceIndex) => utteranceIndex >= searchStart && utterance.startMs === first.startMs && utterance.endMs === first.endMs && utterance.text === first.text);
    if (index >= 0) {
      candidateStartByGroupId.set(candidate.groupId, index);
      searchStart = index + candidate.segments.length;
    }
  }

  const output: UtteranceDraft[] = [];
  let index = 0;
  while (index < utterances.length) {
    const candidate = candidates.find((item) => candidateStartByGroupId.get(item.groupId) === index);
    if (!candidate) {
      output.push(utterances[index]);
      index += 1;
      continue;
    }

    const result = resultByGroupId.get(candidate.groupId);
    if (!result || !candidateByGroupId.has(candidate.groupId)) {
      output.push(...utterances.slice(index, index + candidate.segments.length));
      index += candidate.segments.length;
      continue;
    }

    const groupUtterances = utterances.slice(index, index + candidate.segments.length);
    const sourceIdToPosition = new Map(candidate.segments.map((segment, position) => [segment.id, position]));
    let coveredPositions: number[] = [];
    const merged: UtteranceDraft[] = [];
    let valid = true;

    for (const group of result.groups) {
      const positions = group.sourceIds.map((sourceId) => sourceIdToPosition.get(sourceId));
      if (positions.some((position) => position === undefined) || !group.text.trim()) {
        valid = false;
        break;
      }
      const numericPositions = positions as number[];
      const expected = Array.from({ length: numericPositions[numericPositions.length - 1] - numericPositions[0] + 1 }, (_, offset) => numericPositions[0] + offset);
      if (numericPositions.join(",") !== expected.join(",")) {
        valid = false;
        break;
      }
      coveredPositions = [...coveredPositions, ...numericPositions];
      const parts = numericPositions.map((position) => groupUtterances[position]);
      const first = parts[0];
      const last = parts[parts.length - 1];
      merged.push({
        startMs: first.startMs,
        endMs: last.endMs,
        text: group.text.trim(),
        speakerLabel: first.speakerLabel,
        speakerClusterId: first.speakerClusterId,
        sourceIds: parts.flatMap((part) => part.sourceIds),
        isTargetPerson: parts.some((part) => part.isTargetPerson),
        targetPersonConfidence: normalizeConfidence(Math.max(...parts.map((part) => part.targetPersonConfidence ?? 0))) || null,
        matchedSpeakerProfileId: first.matchedSpeakerProfileId,
        mergeReason: parts.length > 1 ? "llm_same_speaker_semantic_merge" : first.mergeReason
      });
    }

    const expectedCoverage = Array.from({ length: groupUtterances.length }, (_, position) => position).join(",");
    if (!valid || coveredPositions.join(",") !== expectedCoverage) {
      output.push(...groupUtterances);
    } else {
      output.push(...merged);
    }
    index += candidate.segments.length;
  }
  return output;
}

export async function generateUtteranceSegments(
  recordingId: string,
  options: {
    correctTexts?: (texts: string[]) => Promise<string[]>;
    mergeUtterances?: (candidates: UtteranceMergeCandidate[]) => Promise<UtteranceMergeResult[]>;
  } = {}
): Promise<void> {
  const config = getAppConfig();
  const limits = {
    maxDurationMs: config.audio.utteranceMaxDurationMs,
    maxTextChars: config.audio.utteranceMaxTextChars
  };
  console.log("[utterance] generating merged segments", { recordingId, ...limits });
  const rows = await query<Record<string, any>>(
    "select * from transcription_segments where recording_id = $1 order by segment_index",
    [recordingId]
  );

  const utterances: Array<{
    startMs: number;
    endMs: number;
    textParts: string[];
    speakerLabel: string | null;
    speakerClusterId: string | null;
    sourceIds: string[];
    isTargetPerson: boolean;
    targetPersonConfidence: number | null;
    matchedSpeakerProfileId: string | null;
  }> = [];

  for (const segment of rows) {
    const current = utterances[utterances.length - 1];
    if (current && canAppendToUtterance(current, segment, limits)) {
      current.endMs = segment.end_ms;
      current.textParts.push(segment.text);
      current.sourceIds.push(segment.id);
      current.isTargetPerson = current.isTargetPerson || segment.is_target_person;
      current.targetPersonConfidence = Math.max(current.targetPersonConfidence ?? 0, Number(segment.target_person_confidence ?? 0)) || null;
      current.matchedSpeakerProfileId ??= segment.matched_speaker_profile_id;
      continue;
    }

    utterances.push({
      startMs: segment.start_ms,
      endMs: segment.end_ms,
      textParts: [segment.text],
      speakerLabel: segment.speaker_label,
      speakerClusterId: segment.speaker_cluster_id,
      sourceIds: [segment.id],
      isTargetPerson: segment.is_target_person,
      targetPersonConfidence: segment.target_person_confidence === null ? null : Number(segment.target_person_confidence),
      matchedSpeakerProfileId: segment.matched_speaker_profile_id
    });
  }

  const rawTexts = utterances.map((utterance) => joinUtteranceText(utterance.textParts));
  const texts = options.correctTexts ? await options.correctTexts(rawTexts) : rawTexts;
  const correctedUtterances: UtteranceDraft[] = utterances.map((utterance, index) => ({
    startMs: utterance.startMs,
    endMs: utterance.endMs,
    text: texts[index] ?? rawTexts[index] ?? "",
    speakerLabel: utterance.speakerLabel,
    speakerClusterId: utterance.speakerClusterId,
    sourceIds: utterance.sourceIds,
    isTargetPerson: utterance.isTargetPerson,
    targetPersonConfidence: utterance.targetPersonConfidence,
    matchedSpeakerProfileId: utterance.matchedSpeakerProfileId,
    mergeReason: "same_speaker"
  }));
  const mergeCandidates = options.mergeUtterances
    ? buildLlmMergeCandidates(correctedUtterances, {
        maxGapMs: config.audio.utteranceLlmMergeMaxGapMs,
        maxDurationMs: config.audio.utteranceLlmMergeMaxDurationMs,
        maxTextChars: config.audio.utteranceLlmMergeMaxTextChars
      })
    : [];
  const mergeResults = options.mergeUtterances && mergeCandidates.length > 0 ? await options.mergeUtterances(mergeCandidates) : [];
  const finalUtterances = applyLlmMergeResults(correctedUtterances, mergeCandidates, mergeResults);

  await transaction(async (client) => {
    await client.query("delete from recording_search_chunks where recording_id = $1", [recordingId]);
    await client.query("delete from utterance_segments where recording_id = $1", [recordingId]);
    for (const [index, utterance] of finalUtterances.entries()) {
      await client.query(
        `insert into utterance_segments (
          recording_id, utterance_index, start_ms, end_ms, text, speaker_label, speaker_cluster_id,
          source_transcription_segment_ids, is_target_person, target_person_confidence, matched_speaker_profile_id, merge_reason
        ) values ($1, $2, $3, $4, $5, $6, $7, $8::uuid[], $9, $10, $11, $12)`,
        [
          recordingId,
          index,
          utterance.startMs,
          utterance.endMs,
          utterance.text,
          utterance.speakerLabel,
          utterance.speakerClusterId,
          utterance.sourceIds,
          utterance.isTargetPerson,
          utterance.targetPersonConfidence,
          utterance.matchedSpeakerProfileId,
          utterance.mergeReason
        ]
      );
    }
    console.log("[utterance] merged segments saved", {
      recordingId,
      sourceCount: rows.length,
      utteranceCount: finalUtterances.length,
      llmMergeCandidateCount: mergeCandidates.length
    });
  });
}

export async function getRecordingForWorker(recordingId: string): Promise<{ recording: Recording; transcriptionSegments: TranscriptionSegment[]; diarizationSegments: SpeakerDiarizationSegment[]; utteranceSegments: UtteranceSegment[] }> {
  const detail = await getRecordingDetail(recordingId);
  if (!detail) throw new Error("Recording not found");
  return {
    recording: detail.recording,
    transcriptionSegments: detail.transcriptionSegments,
    diarizationSegments: detail.speakerDiarizationSegments,
    utteranceSegments: detail.utteranceSegments
  };
}

export async function applySpeakerIdentification(recordingId: string, matches: SpeakerIdentificationMatch[]): Promise<void> {
  console.log("[speaker-id] applying matches", {
    recordingId,
    matchCount: matches.length,
    positiveCount: matches.filter((match) => match.isTargetPerson).length
  });
  await transaction(async (client) => {
    await client.query(
      `update speaker_diarization_segments
       set is_target_person = false,
           target_person_confidence = null,
           matched_speaker_profile_id = null
       where recording_id = $1`,
      [recordingId]
    );
    await client.query(
      `update transcription_segments
       set is_target_person = false,
           target_person_confidence = null,
           matched_speaker_profile_id = null
       where recording_id = $1`,
      [recordingId]
    );
    await client.query(
      `update utterance_segments
       set is_target_person = false,
           target_person_confidence = null,
           matched_speaker_profile_id = null
       where recording_id = $1`,
      [recordingId]
    );

    for (const match of matches) {
      const confidence = normalizeConfidence(match.confidence);
      await client.query(
        `update speaker_diarization_segments
         set is_target_person = $2,
             target_person_confidence = $3,
             matched_speaker_profile_id = $4
         where id = $1`,
        [match.diarizationSegmentId, match.isTargetPerson, confidence, match.speakerProfileId]
      );
      await client.query(
        `update transcription_segments
         set is_target_person = $2,
             target_person_confidence = $3,
             matched_speaker_profile_id = $4
         where recording_id = $5 and diarization_segment_id = $1`,
        [match.diarizationSegmentId, match.isTargetPerson, confidence, match.speakerProfileId, recordingId]
      );
    }
  });
}
