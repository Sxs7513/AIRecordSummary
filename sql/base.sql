-- Phase 1 PostgreSQL schema
-- Scope:
-- - recordings
-- - transcription
-- - speaker diarization
-- - target speaker profile and samples
-- - background processing jobs

create extension if not exists "pgcrypto";
create extension if not exists pg_trgm;
do $$
begin
    create extension if not exists vector;
exception
    when undefined_file then
        raise exception 'pgvector extension is required for Phase 2 search. Install pgvector for this PostgreSQL instance, or set EMBEDDING_ENABLED=false only after applying a schema variant without vector columns.';
end $$;

-- recordings
-- 录音主表。
-- 每上传一条音频，就会在这张表里生成一条记录。
-- 这张表主要负责保存文件级元数据，以及整条录音当前的总处理状态。
-- 后续转写、speaker diarization、目标人物识别等结果，都会通过 recording_id 关联回来。
create table if not exists recordings (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    file_name text not null,
    storage_path text not null,
    mime_type text not null,
    file_size_bytes bigint not null check (file_size_bytes >= 0),
    duration_seconds integer check (duration_seconds is null or duration_seconds >= 0),
    status text not null check (status in ('uploaded', 'processing', 'completed', 'failed')),
    error_message text,
    uploaded_at timestamptz not null default now(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table recordings is
    '录音主表，保存每条上传音频的基础信息、文件信息和整体处理状态。';
comment on column recordings.id is
    '录音主键，使用 UUID 生成，供业务层和关联表统一引用。';
comment on column recordings.title is
    '录音标题。Phase 1 可直接使用文件名作为默认标题，后续可支持手动编辑。';
comment on column recordings.file_name is
    '用户上传时的原始文件名，用于后台展示和排查问题。';
comment on column recordings.storage_path is
    '音频文件在本地文件系统或对象存储中的相对路径或 key。';
comment on column recordings.mime_type is
    '音频文件 MIME 类型，例如 audio/mpeg、audio/wav。';
comment on column recordings.file_size_bytes is
    '音频文件大小，单位字节，用于展示和容量控制。';
comment on column recordings.duration_seconds is
    '音频总时长，单位秒。允许为空，表示上传后尚未完成元数据提取。';
comment on column recordings.status is
    '录音整体处理状态。uploaded 表示已上传，processing 表示处理中，completed 表示已完成，failed 表示处理失败。';
comment on column recordings.error_message is
    '录音级错误信息，记录最近一次处理失败原因，便于后台排查。';
comment on column recordings.uploaded_at is
    '录音上传完成时间，反映文件进入系统的业务时间。';
comment on column recordings.created_at is
    '数据库记录创建时间。';
comment on column recordings.updated_at is
    '数据库记录最后更新时间，后续可结合触发器或应用层统一维护。';

create index if not exists recordings_status_idx on recordings (status);
create index if not exists recordings_uploaded_at_idx on recordings (uploaded_at desc);

-- transcriptions
-- 录音整条转写结果主表。
-- 一条录音在 Phase 1 只保留一份当前生效的完整转写结果。
-- full_text 保存整条录音合并后的完整文本，segment_count 对应下游 transcription_segments 的数量。
create table if not exists transcriptions (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    language text,
    model_name text not null,
    full_text text not null default '',
    segment_count integer not null default 0 check (segment_count >= 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (recording_id)
);

comment on table transcriptions is
    '整条录音的转写结果主表，保存 Whisper 生成的完整文本和模型信息。';
comment on column transcriptions.id is
    '转写结果主键。';
comment on column transcriptions.recording_id is
    '关联 recordings.id，表示这份转写结果属于哪条录音。';
comment on column transcriptions.language is
    'Whisper 识别出的语言标识，例如 zh、en。允许为空，表示模型未返回或暂未解析。';
comment on column transcriptions.model_name is
    '实际执行转写时使用的模型名称，例如 whisper-large-v3。';
comment on column transcriptions.full_text is
    '整条录音合并后的完整转写文本，便于详情页直接展示和后续检索使用。';
comment on column transcriptions.segment_count is
    '该录音被拆分出的转写片段数量，对应 transcription_segments 的条数。';
comment on column transcriptions.created_at is
    '转写结果创建时间。';
comment on column transcriptions.updated_at is
    '转写结果最后更新时间。';

create index if not exists transcriptions_recording_id_idx on transcriptions (recording_id);

-- speaker_profiles
-- 目标人物主表。
-- 这里保存系统需要重点识别的已知人物，例如你特别关注的某个核心发言人。
-- 该表不保存样本音频本身，只保存人物主体信息。
create table if not exists speaker_profiles (
    id uuid primary key default gen_random_uuid(),
    display_name text not null,
    status text not null check (status in ('active', 'inactive')),
    notes text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table speaker_profiles is
    '目标人物主表，用于保存需要重点识别的已知说话人。';
comment on column speaker_profiles.id is
    '目标人物主键。';
comment on column speaker_profiles.display_name is
    '目标人物展示名称，用于后台页面和识别结果展示。';
comment on column speaker_profiles.status is
    '目标人物状态。active 表示参与识别，inactive 表示暂不参与识别。';
comment on column speaker_profiles.notes is
    '备注信息，可用于记录人物背景、样本说明或人工校验说明。';
comment on column speaker_profiles.created_at is
    '目标人物记录创建时间。';
comment on column speaker_profiles.updated_at is
    '目标人物记录最后更新时间。';

create index if not exists speaker_profiles_status_idx on speaker_profiles (status);

-- speaker_profile_samples
-- 目标人物参考样本表。
-- 这张表保存用于声纹比对的参考音频样本，一条 speaker_profile 可以关联多条样本。
-- 后续目标人物识别时，会基于这些样本和录音中的说话片段做相似度比对。
create table if not exists speaker_profile_samples (
    id uuid primary key default gen_random_uuid(),
    speaker_profile_id uuid not null references speaker_profiles(id) on delete cascade,
    file_name text not null,
    storage_path text not null,
    mime_type text not null,
    file_size_bytes bigint not null check (file_size_bytes >= 0),
    duration_seconds integer check (duration_seconds is null or duration_seconds >= 0),
    status text not null check (status in ('uploaded', 'processing', 'completed', 'failed')),
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table speaker_profile_samples is
    '目标人物参考音频样本表，保存用于声纹比对的样本音频。';
comment on column speaker_profile_samples.id is
    '目标人物样本主键。';
comment on column speaker_profile_samples.speaker_profile_id is
    '关联 speaker_profiles.id，表示该样本属于哪个目标人物。';
comment on column speaker_profile_samples.file_name is
    '上传样本时的原始文件名。';
comment on column speaker_profile_samples.storage_path is
    '样本音频在文件系统或对象存储中的路径或 key。';
comment on column speaker_profile_samples.mime_type is
    '样本音频 MIME 类型。';
comment on column speaker_profile_samples.file_size_bytes is
    '样本音频文件大小，单位字节。';
comment on column speaker_profile_samples.duration_seconds is
    '样本音频时长，单位秒。';
comment on column speaker_profile_samples.status is
    '样本处理状态。uploaded 表示已上传，processing 表示处理中，completed 表示可参与识别，failed 表示处理失败。';
comment on column speaker_profile_samples.error_message is
    '样本处理失败时的错误信息。';
comment on column speaker_profile_samples.created_at is
    '样本记录创建时间。';
comment on column speaker_profile_samples.updated_at is
    '样本记录最后更新时间。';

create index if not exists speaker_profile_samples_profile_id_idx on speaker_profile_samples (speaker_profile_id);
create index if not exists speaker_profile_samples_status_idx on speaker_profile_samples (status);

-- speaker_diarization_segments
-- 说话人分离结果表。
-- 这张表记录单条录音里“谁在什么时间说话”的原始说话片段结果。
-- speaker_cluster_id 用于标识同一条录音中的同一说话人聚类，speaker_label 用于页面展示，例如 Speaker A。
-- 目标人物识别结果也会先挂在这张表上，再同步回转写片段表。
create table if not exists speaker_diarization_segments (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    speaker_cluster_id text not null,
    speaker_label text not null,
    start_ms integer not null check (start_ms >= 0),
    end_ms integer not null check (end_ms >= start_ms),
    confidence numeric(5,4) check (confidence is null or (confidence >= 0 and confidence <= 1)),
    is_target_person boolean not null default false,
    target_person_confidence numeric(5,4) check (
        target_person_confidence is null
        or (target_person_confidence >= 0 and target_person_confidence <= 1)
    ),
    matched_speaker_profile_id uuid references speaker_profiles(id) on delete set null,
    created_at timestamptz not null default now()
);

comment on table speaker_diarization_segments is
    'Speaker diarization 输出表，记录单条录音中“谁在什么时间说话”的原始说话片段。';
comment on column speaker_diarization_segments.id is
    '说话片段主键。';
comment on column speaker_diarization_segments.recording_id is
    '关联 recordings.id，表示该说话片段属于哪条录音。';
comment on column speaker_diarization_segments.speaker_cluster_id is
    '同一条录音内的说话人聚类标识，由 diarization 流程生成，用于内部关联同一说话人。';
comment on column speaker_diarization_segments.speaker_label is
    '面向页面展示的匿名说话人标签，例如 Speaker A、Speaker B。';
comment on column speaker_diarization_segments.start_ms is
    '说话片段开始时间，单位毫秒。';
comment on column speaker_diarization_segments.end_ms is
    '说话片段结束时间，单位毫秒，必须大于或等于开始时间。';
comment on column speaker_diarization_segments.confidence is
    'diarization 结果置信度，范围 0 到 1。';
comment on column speaker_diarization_segments.is_target_person is
    '该说话片段是否被识别为目标人物。false 表示未命中或尚未命中。';
comment on column speaker_diarization_segments.target_person_confidence is
    '目标人物识别置信度，范围 0 到 1。';
comment on column speaker_diarization_segments.matched_speaker_profile_id is
    '若命中目标人物，则关联 speaker_profiles.id；未命中时为空。';
comment on column speaker_diarization_segments.created_at is
    '说话片段记录创建时间。';

create index if not exists speaker_diarization_segments_recording_id_idx
    on speaker_diarization_segments (recording_id);
create index if not exists speaker_diarization_segments_recording_speaker_idx
    on speaker_diarization_segments (recording_id, speaker_cluster_id);
create index if not exists speaker_diarization_segments_recording_time_idx
    on speaker_diarization_segments (recording_id, start_ms, end_ms);

-- transcription_segments
-- 转写片段表。
-- 这张表保存 Whisper 输出的每一个时间片段文本。
-- 在 diarization 结果和转写结果完成时间对齐后，会把 speaker_label、speaker_cluster_id、
-- speaker_confidence 以及目标人物识别结果回写到这里，方便详情页直接查询和展示。
create table if not exists transcription_segments (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    transcription_id uuid not null references transcriptions(id) on delete cascade,
    segment_index integer not null check (segment_index >= 0),
    start_ms integer not null check (start_ms >= 0),
    end_ms integer not null check (end_ms >= start_ms),
    text text not null,
    speaker_label text,
    speaker_cluster_id text,
    speaker_confidence numeric(5,4) check (
        speaker_confidence is null or (speaker_confidence >= 0 and speaker_confidence <= 1)
    ),
    is_target_person boolean not null default false,
    target_person_confidence numeric(5,4) check (
        target_person_confidence is null
        or (target_person_confidence >= 0 and target_person_confidence <= 1)
    ),
    diarization_segment_id uuid references speaker_diarization_segments(id) on delete set null,
    matched_speaker_profile_id uuid references speaker_profiles(id) on delete set null,
    created_at timestamptz not null default now(),
    unique (transcription_id, segment_index)
);

comment on table transcription_segments is
    '转写片段表，保存 Whisper 输出的时间片段文本，并回写对应说话人和目标人物识别结果。';
comment on column transcription_segments.id is
    '转写片段主键。';
comment on column transcription_segments.recording_id is
    '关联 recordings.id，表示该片段属于哪条录音。';
comment on column transcription_segments.transcription_id is
    '关联 transcriptions.id，表示该片段属于哪份转写结果。';
comment on column transcription_segments.segment_index is
    '片段在整条转写中的顺序编号，从 0 开始。';
comment on column transcription_segments.start_ms is
    '转写片段开始时间，单位毫秒。';
comment on column transcription_segments.end_ms is
    '转写片段结束时间，单位毫秒。';
comment on column transcription_segments.text is
    '该时间片段对应的转写文本内容。';
comment on column transcription_segments.speaker_label is
    '对齐后得到的匿名说话人标签，例如 Speaker A。';
comment on column transcription_segments.speaker_cluster_id is
    '对齐后得到的内部说话人聚类标识。';
comment on column transcription_segments.speaker_confidence is
    '从 diarization 流程回写到转写片段的说话人置信度。';
comment on column transcription_segments.is_target_person is
    '该转写片段是否命中目标人物。';
comment on column transcription_segments.target_person_confidence is
    '目标人物识别置信度。';
comment on column transcription_segments.diarization_segment_id is
    '关联 speaker_diarization_segments.id，表示该转写片段主要匹配到的说话片段。';
comment on column transcription_segments.matched_speaker_profile_id is
    '若命中目标人物，则关联目标人物 ID。';
comment on column transcription_segments.created_at is
    '转写片段记录创建时间。';

create index if not exists transcription_segments_recording_id_idx on transcription_segments (recording_id);
create index if not exists transcription_segments_transcription_id_idx on transcription_segments (transcription_id);
create index if not exists transcription_segments_recording_time_idx
    on transcription_segments (recording_id, start_ms, end_ms);
create index if not exists transcription_segments_recording_speaker_idx
    on transcription_segments (recording_id, speaker_cluster_id);
create index if not exists transcription_segments_target_person_idx
    on transcription_segments (recording_id, is_target_person);

-- utterance_segments
-- 连续发言展示层。
-- 这张表不是模型原始输出，而是基于 transcription_segments 和 speaker diarization 对齐结果生成的业务展示段。
-- Phase 1 先按连续相同 speaker 合并 Whisper 切片，避免停顿导致同一人一句话被拆得过碎。
create table if not exists utterance_segments (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    utterance_index integer not null check (utterance_index >= 0),
    start_ms integer not null check (start_ms >= 0),
    end_ms integer not null check (end_ms >= start_ms),
    text text not null,
    speaker_label text,
    speaker_cluster_id text,
    source_transcription_segment_ids uuid[] not null default '{}',
    is_target_person boolean not null default false,
    target_person_confidence numeric(5,4) check (
        target_person_confidence is null
        or (target_person_confidence >= 0 and target_person_confidence <= 1)
    ),
    matched_speaker_profile_id uuid references speaker_profiles(id) on delete set null,
    merge_reason text not null default 'same_speaker',
    created_at timestamptz not null default now(),
    unique (recording_id, utterance_index)
);

comment on table utterance_segments is
    '连续发言展示层，基于转写片段和说话人对齐结果合并生成，供详情页优先展示。';
comment on column utterance_segments.recording_id is
    '关联 recordings.id，表示该连续发言属于哪条录音。';
comment on column utterance_segments.utterance_index is
    '连续发言在整条录音中的展示顺序，从 0 开始。';
comment on column utterance_segments.start_ms is
    '连续发言开始时间，单位毫秒。';
comment on column utterance_segments.end_ms is
    '连续发言结束时间，单位毫秒。';
comment on column utterance_segments.text is
    '合并后的连续发言文本。';
comment on column utterance_segments.speaker_label is
    '该连续发言对应的匿名说话人标签。';
comment on column utterance_segments.speaker_cluster_id is
    '该连续发言对应的说话人聚类标识。';
comment on column utterance_segments.source_transcription_segment_ids is
    '组成该连续发言的原始转写片段 ID 列表。';
comment on column utterance_segments.merge_reason is
    '生成该连续发言的合并原因。Phase 1 默认为 same_speaker。';

create index if not exists utterance_segments_recording_id_idx on utterance_segments (recording_id);
create index if not exists utterance_segments_recording_time_idx
    on utterance_segments (recording_id, start_ms, end_ms);
create index if not exists utterance_segments_recording_speaker_idx
    on utterance_segments (recording_id, speaker_cluster_id);
create index if not exists utterance_segments_target_person_idx
    on utterance_segments (recording_id, is_target_person);

-- processing_jobs
-- 后台处理任务表。
-- 这张表记录每条录音在异步链路中的处理进度，包括转写、speaker diarization 和目标人物识别。
-- Worker 会基于这张表拉取任务、更新状态、记录重试次数和失败原因。
create table if not exists processing_jobs (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    job_type text not null check (
        job_type in ('transcription', 'speaker_diarization', 'speaker_identification', 'text_correction', 'embedding_indexing')
    ),
    status text not null check (status in ('pending', 'running', 'completed', 'failed')),
    attempt_count integer not null default 0 check (attempt_count >= 0),
    error_message text,
    started_at timestamptz,
    finished_at timestamptz,
    processing_duration_ms integer check (processing_duration_ms is null or processing_duration_ms >= 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table processing_jobs
    add column if not exists processing_duration_ms integer check (
        processing_duration_ms is null or processing_duration_ms >= 0
    );

comment on table processing_jobs is
    '后台任务表，记录转写、speaker diarization、目标人物识别等异步任务的执行情况。';
comment on column processing_jobs.id is
    '后台任务主键。';
comment on column processing_jobs.recording_id is
    '关联 recordings.id，表示该任务属于哪条录音。';
comment on column processing_jobs.job_type is
    '任务类型。transcription 表示转写，speaker_diarization 表示说话人分离，speaker_identification 表示目标人物识别，text_correction 表示文本校正。';
comment on column processing_jobs.status is
    '任务状态。pending 表示待执行，running 表示执行中，completed 表示完成，failed 表示失败。';
comment on column processing_jobs.attempt_count is
    '任务重试次数统计，从 0 开始累计。';
comment on column processing_jobs.error_message is
    '任务失败原因或最近一次错误摘要。';
comment on column processing_jobs.started_at is
    '任务实际开始执行时间。';
comment on column processing_jobs.finished_at is
    '任务执行结束时间，无论成功或失败都应在结束时写入。';
comment on column processing_jobs.processing_duration_ms is
    '任务实际处理耗时，单位毫秒。任务成功或失败结束时由应用层根据 started_at 和 finished_at 计算写入。';
comment on column processing_jobs.created_at is
    '任务记录创建时间。';
comment on column processing_jobs.updated_at is
    '任务记录最后更新时间。';

create index if not exists processing_jobs_recording_id_idx on processing_jobs (recording_id);
create index if not exists processing_jobs_status_idx on processing_jobs (status);
create index if not exists processing_jobs_type_status_idx on processing_jobs (job_type, status);

update processing_jobs
set processing_duration_ms = greatest(0, floor(extract(epoch from (finished_at - started_at)) * 1000)::integer)
where processing_duration_ms is null
  and started_at is not null
  and finished_at is not null;

do $$
begin
    alter table processing_jobs drop constraint if exists processing_jobs_job_type_check;
    update processing_jobs set job_type = 'text_correction' where job_type = 'text_polishing';
    alter table processing_jobs add constraint processing_jobs_job_type_check
        check (job_type in ('transcription', 'speaker_diarization', 'speaker_identification', 'text_correction', 'embedding_indexing'));
end $$;

-- Phase 2: semantic search and RAG
create table if not exists embedding_models (
    id uuid primary key default gen_random_uuid(),
    provider text not null,
    model_name text not null,
    dimensions integer not null check (dimensions > 0),
    distance_metric text not null default 'cosine',
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    unique (provider, model_name, dimensions)
);

create table if not exists recording_search_chunks (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    embedding_model_id uuid not null references embedding_models(id) on delete restrict,
    chunk_index integer not null check (chunk_index >= 0),
    text text not null,
    normalized_text text not null,
    start_ms integer not null check (start_ms >= 0),
    end_ms integer not null check (end_ms >= start_ms),
    speaker_labels text[] not null default '{}',
    speaker_cluster_ids text[] not null default '{}',
    source_utterance_segment_ids uuid[] not null default '{}',
    source_transcription_segment_ids uuid[] not null default '{}',
    is_target_person boolean not null default false,
    matched_speaker_profile_ids uuid[] not null default '{}',
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1024) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (recording_id, embedding_model_id, chunk_index)
);

create index if not exists recording_search_chunks_recording_id_idx
    on recording_search_chunks (recording_id);
create index if not exists recording_search_chunks_time_idx
    on recording_search_chunks (recording_id, start_ms, end_ms);
create index if not exists recording_search_chunks_target_person_idx
    on recording_search_chunks (is_target_person);
create index if not exists recording_search_chunks_text_trgm_idx
    on recording_search_chunks using gin (normalized_text gin_trgm_ops);
create index if not exists recording_search_chunks_embedding_hnsw_idx
    on recording_search_chunks using hnsw (embedding vector_cosine_ops);

create table if not exists search_queries (
    id uuid primary key default gen_random_uuid(),
    query_text text not null,
    normalized_query text not null,
    filters jsonb not null default '{}'::jsonb,
    result_count integer not null default 0,
    latency_ms integer check (latency_ms is null or latency_ms >= 0),
    created_at timestamptz not null default now()
);

create table if not exists search_result_clicks (
    id uuid primary key default gen_random_uuid(),
    search_query_id uuid references search_queries(id) on delete set null,
    recording_id uuid not null references recordings(id) on delete cascade,
    search_chunk_id uuid references recording_search_chunks(id) on delete set null,
    target_ms integer check (target_ms is null or target_ms >= 0),
    created_at timestamptz not null default now()
);

-- Optional helper trigger target for app layer:
-- update updated_at on row modifications in application code or via triggers later.
