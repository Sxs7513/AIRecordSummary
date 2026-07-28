-- Phase 1 PostgreSQL schema
-- Scope:
-- - recordings
-- - transcription
-- - speaker diarization
-- - target speaker profile and samples

create extension if not exists "pgcrypto";
create extension if not exists pg_trgm;
do $$
begin
    create extension if not exists vector;
exception
    when undefined_file then
        raise exception 'pgvector extension is required for Phase 2 search. Install pgvector for this PostgreSQL instance, or set EMBEDDING_ENABLED=false only after applying a schema variant without vector columns.';
end $$;

-- users
-- 本地账号主表。密码只保存应用层生成的强哈希，不保存明文。
create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    email text not null unique,
    display_name text not null,
    password_hash text not null,
    status text not null default 'active' check (status in ('active', 'disabled')),
    current_workspace_id uuid,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table users is '本地登录账号。current_workspace_id 表示用户当前默认工作区，不替代多对多成员关系。';
comment on column users.id is '用户主键。';
comment on column users.email is '唯一登录邮箱，统一按小写保存。';
comment on column users.display_name is '界面展示名称。';
comment on column users.password_hash is '使用 scrypt 生成的密码哈希，绝不存储明文密码。';
comment on column users.status is '账号状态：active 可登录，disabled 被禁用。';
comment on column users.current_workspace_id is '当前默认工作区，必须是该用户的一条有效成员关系。';
comment on column users.created_at is '账号创建时间。';
comment on column users.updated_at is '账号最后更新时间。';

-- workspaces
-- 团队协作和录音默认授权边界。
create table if not exists workspaces (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table workspaces is '工作区，录音和聊天等业务数据的默认授权边界。';
comment on column workspaces.id is '工作区主键。';
comment on column workspaces.name is '工作区展示名称。';
comment on column workspaces.created_at is '工作区创建时间。';
comment on column workspaces.updated_at is '工作区最后更新时间。';

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'users_current_workspace_id_fkey'
          and conrelid = 'users'::regclass
    ) then
        alter table users
            add constraint users_current_workspace_id_fkey
            foreign key (current_workspace_id) references workspaces(id) on delete set null;
    end if;
end $$;

create table if not exists workspace_memberships (
    workspace_id uuid not null references workspaces(id) on delete cascade,
    user_id uuid not null references users(id) on delete cascade,
    role text not null check (role in ('owner', 'admin', 'member')),
    created_at timestamptz not null default now(),
    primary key (workspace_id, user_id)
);

comment on table workspace_memberships is '用户与工作区的多对多成员关系；同一用户可属于多个工作区。';
comment on column workspace_memberships.workspace_id is '所属工作区。';
comment on column workspace_memberships.user_id is '成员用户。';
comment on column workspace_memberships.role is '工作区角色：owner、admin 或 member。';
comment on column workspace_memberships.created_at is '加入工作区的时间。';

create table if not exists user_sessions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references users(id) on delete cascade,
    token_hash text not null unique,
    expires_at timestamptz not null,
    revoked_at timestamptz,
    last_seen_at timestamptz not null default now(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table user_sessions is '服务端 Session。浏览器 Cookie 的随机值只以哈希形式保存，支持立即撤销。';
comment on column user_sessions.id is '会话主键。';
comment on column user_sessions.user_id is '会话所属用户。';
comment on column user_sessions.token_hash is 'Cookie 随机 token 的 SHA-256 哈希。';
comment on column user_sessions.expires_at is '会话过期时间。';
comment on column user_sessions.revoked_at is '撤销时间；非空表示已登出或失效。';
comment on column user_sessions.last_seen_at is '最近一次通过该会话验证的时间。';
comment on column user_sessions.created_at is '会话创建时间。';
comment on column user_sessions.updated_at is '会话最后更新时间。';

-- recordings
-- 录音主表。
-- 每上传一条音频，就会在这张表里生成一条记录。
-- 这张表主要负责保存文件级元数据，以及整条录音当前的总处理状态。
-- 后续转写、speaker diarization、目标人物识别等结果，都会通过 recording_id 关联回来。
create table if not exists recordings (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete restrict,
    owner_user_id uuid not null references users(id) on delete restrict,
    title text not null,
    file_name text not null,
    storage_path text not null,
    location text,
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
comment on column recordings.workspace_id is
    '录音所属工作区，是默认访问授权边界。';
comment on column recordings.owner_user_id is
    '创建该录音的用户，用于审计和所有者级操作校验。';
comment on column recordings.title is
    '录音标题。Phase 1 可直接使用文件名作为默认标题，后续可支持手动编辑。';
comment on column recordings.file_name is
    '用户上传时的原始文件名，用于后台展示和排查问题。';
comment on column recordings.storage_path is
    '音频文件在本地文件系统或对象存储中的相对路径或 key。';
comment on column recordings.location is
    '录音发生地点，由用户在录音详情页手动配置，用于录音级筛选和联合检索。';
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
create index if not exists recordings_workspace_id_idx on recordings (workspace_id);
create index if not exists recordings_owner_user_id_idx on recordings (owner_user_id);
create index if not exists recordings_uploaded_at_idx on recordings (uploaded_at desc);
create index if not exists recordings_created_at_idx on recordings (created_at desc);
create index if not exists recordings_location_trgm_idx on recordings using gin (location gin_trgm_ops);

-- 例外录音分享；不用于表达同一工作区成员的默认访问权。
create table if not exists recording_memberships (
    recording_id uuid not null references recordings(id) on delete cascade,
    user_id uuid not null references users(id) on delete cascade,
    role text not null check (role in ('viewer', 'editor')),
    granted_by_user_id uuid not null references users(id) on delete restrict,
    created_at timestamptz not null default now(),
    primary key (recording_id, user_id)
);

comment on table recording_memberships is '单条录音的例外分享授权，供非所属工作区成员访问指定录音。';
comment on column recording_memberships.recording_id is '被分享的录音。';
comment on column recording_memberships.user_id is '被授予访问权限的用户。';
comment on column recording_memberships.role is '例外授权角色：viewer 只读，editor 可编辑。';
comment on column recording_memberships.granted_by_user_id is '执行授权的用户。';
comment on column recording_memberships.created_at is '授权创建时间。';

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

-- Manual, recording-local identity for one diarization cluster. This is the
-- canonical display mapping; pipeline outputs keep their stable anonymous label.
create table if not exists recording_speaker_mappings (
    recording_id uuid not null references recordings(id) on delete cascade,
    speaker_cluster_id text not null,
    display_name text not null check (btrim(display_name) <> ''),
    speaker_profile_id uuid references speaker_profiles(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (recording_id, speaker_cluster_id)
);

create index if not exists recording_speaker_mappings_profile_id_idx
    on recording_speaker_mappings (speaker_profile_id);

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

create table if not exists transcription_tokens (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    transcription_id uuid not null references transcriptions(id) on delete cascade,
    transcription_segment_id uuid references transcription_segments(id) on delete set null,
    token_index integer not null check (token_index >= 0),
    source_window_index integer not null check (source_window_index >= 0),
    text text not null,
    start_ms integer not null check (start_ms >= 0),
    end_ms integer not null check (end_ms >= start_ms),
    speaker_cluster_id text,
    speaker_label text,
    attribution_status text not null check (attribution_status in ('matched', 'ambiguous', 'unmatched')),
    unique (transcription_id, token_index)
);
create index if not exists transcription_tokens_recording_time_idx on transcription_tokens (recording_id, start_ms, end_ms);

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

-- Preserve labels edited through the former fan-out update endpoint when the
-- mapping table is first introduced into an existing database.
insert into recording_speaker_mappings (recording_id, speaker_cluster_id, display_name)
select recording_id, speaker_cluster_id, min(speaker_label)
from speaker_diarization_segments
where btrim(speaker_cluster_id) <> '' and btrim(speaker_label) <> ''
group by recording_id, speaker_cluster_id
on conflict (recording_id, speaker_cluster_id) do nothing;

insert into recording_speaker_mappings (recording_id, speaker_cluster_id, display_name)
select recording_id, speaker_cluster_id, min(speaker_label)
from transcription_segments
where speaker_cluster_id is not null
  and speaker_label is not null
  and btrim(speaker_cluster_id) <> ''
  and btrim(speaker_label) <> ''
group by recording_id, speaker_cluster_id
on conflict (recording_id, speaker_cluster_id) do nothing;

insert into recording_speaker_mappings (recording_id, speaker_cluster_id, display_name)
select recording_id, speaker_cluster_id, min(speaker_label)
from utterance_segments
where speaker_cluster_id is not null
  and speaker_label is not null
  and btrim(speaker_cluster_id) <> ''
  and btrim(speaker_label) <> ''
group by recording_id, speaker_cluster_id
on conflict (recording_id, speaker_cluster_id) do nothing;

create table if not exists recording_summaries (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references recordings(id) on delete cascade,
    provider text not null check (provider in ('local_llm', 'deepseek_api')),
    model_name text not null,
    summary_text text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (recording_id)
);

comment on table recording_summaries is
    '录音总结表，保存润色后文本生成的整条录音摘要。';
comment on column recording_summaries.recording_id is
    '关联 recordings.id，表示这份总结属于哪条录音。';
comment on column recording_summaries.provider is
    '总结使用的模型提供方，例如 local_llm 或 deepseek_api。';
comment on column recording_summaries.model_name is
    '总结使用的模型名称或文件名。';
comment on column recording_summaries.summary_text is
    '面向用户展示的录音总结正文。';

create index if not exists recording_summaries_recording_id_idx on recording_summaries (recording_id);

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
    embedding halfvec(2560) not null,
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
create index if not exists recording_search_chunks_text_trgm_gist_idx
    on recording_search_chunks using gist (normalized_text gist_trgm_ops(siglen=64));
create index if not exists recording_search_chunks_embedding_hnsw_idx
    on recording_search_chunks using hnsw (embedding halfvec_cosine_ops);

-- Pipeline runtime
-- Pipeline runs and stage runs are persisted so separate workers can safely
-- advance work, retry failures, and pass artifact references between stages.
create table if not exists pipeline_runs (
    id uuid primary key default gen_random_uuid(),
    subject_type text not null,
    subject_id uuid not null,
    pipeline_name text not null,
    pipeline_version text not null,
    status text not null check (status in ('queued', 'running', 'succeeded', 'partial_failed', 'failed', 'cancelled')),
    started_at timestamptz,
    finished_at timestamptz,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table pipeline_runs is
    '流水线运行主表，记录任意业务对象一次完整处理工作流的名称、版本、总状态和失败信息。';
comment on column pipeline_runs.id is
    '流水线运行主键。';
comment on column pipeline_runs.subject_type is
    '业务对象类型，例如 recording；与 subject_id 共同定位流水线所属对象。';
comment on column pipeline_runs.subject_id is
    '业务对象主键；不建立跨领域外键，保持 pipeline runtime 通用。';
comment on column pipeline_runs.pipeline_name is
    '流水线定义名称，例如 recording_processing。';
comment on column pipeline_runs.pipeline_version is
    '创建本次运行时使用的流水线定义版本，用于结果可追溯。';
comment on column pipeline_runs.status is
    '流水线整体状态：queued、running、succeeded、partial_failed、failed 或 cancelled。';
comment on column pipeline_runs.started_at is
    '首个节点实际开始执行的时间。';
comment on column pipeline_runs.finished_at is
    '流水线进入最终状态的时间。';
comment on column pipeline_runs.error_message is
    '流水线级错误摘要，通常记录必需节点最终失败的原因。';
comment on column pipeline_runs.created_at is
    '流水线运行记录创建时间。';
comment on column pipeline_runs.updated_at is
    '流水线运行记录最近更新时间。';

create table if not exists stage_runs (
    id uuid primary key default gen_random_uuid(),
    pipeline_run_id uuid not null references pipeline_runs(id) on delete cascade,
    subject_type text not null,
    subject_id uuid not null,
    node_name text not null,
    stage_name text not null,
    stage_version text not null,
    required boolean not null default true,
    resource_queue text not null check (resource_queue in ('cpu', 'gpu_normal', 'gpu_high')),
    status text not null check (status in ('pending', 'running', 'succeeded', 'retry_waiting', 'failed', 'cancelled', 'skipped')),
    attempt_count integer not null default 0 check (attempt_count >= 0),
    max_attempts integer check (max_attempts is null or max_attempts > 0),
    progress_percent integer check (progress_percent is null or (progress_percent >= 0 and progress_percent <= 100)),
    progress_message text,
    progress_updated_at timestamptz,
    input_fingerprint text not null,
    input_payload jsonb not null default '{}'::jsonb,
    input_artifacts jsonb not null default '[]'::jsonb,
    output_payload jsonb,
    available_at timestamptz not null default now(),
    started_at timestamptz,
    finished_at timestamptz,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (pipeline_run_id, node_name)
);

comment on table stage_runs is
    '流水线节点运行表，记录每个 stage 的队列、依赖输入、重试次数、输出与执行状态，由 pipeline runtime 推进。';
comment on column stage_runs.id is
    '节点运行主键。';
comment on column stage_runs.pipeline_run_id is
    '关联 pipeline_runs.id，表示节点所属的整次流水线运行。';
comment on column stage_runs.subject_type is
    '业务对象类型，从所属 pipeline_run 冗余保存，供节点执行快速读取。';
comment on column stage_runs.subject_id is
    '业务对象主键，从所属 pipeline_run 冗余保存，供节点执行和 artifact 分区使用。';
comment on column stage_runs.node_name is
    '流水线定义中的节点名称；同一流水线运行内唯一。';
comment on column stage_runs.stage_name is
    '实际执行的 stage 插件名称。';
comment on column stage_runs.stage_version is
    '实际执行的 stage 插件版本。';
comment on column stage_runs.required is
    '是否为流水线成功所必需的节点；非必需节点失败可形成 partial_failed。';
comment on column stage_runs.resource_queue is
    '资源队列：cpu、gpu_normal 或 gpu_high，决定 ResourceScheduler 的资源准入顺序。';
comment on column stage_runs.status is
    '节点状态：pending、running、succeeded、retry_waiting、failed、cancelled 或 skipped。';
comment on column stage_runs.attempt_count is
    '已领取并执行的次数。';
comment on column stage_runs.max_attempts is
    '节点允许的最大执行次数。为空表示不限次数，失败后按退避策略持续重试。';
comment on column stage_runs.progress_percent is
    '节点当前执行进度，范围 0 到 100。为空表示尚未开始或该 stage 不提供细粒度进度。';
comment on column stage_runs.progress_message is
    '节点当前进度说明，例如模型加载、分段转写的当前片段或 diarization 子阶段。';
comment on column stage_runs.progress_updated_at is
    '节点进度最后一次由 worker 写入的时间，用于前端判断进度是否仍在刷新。';
comment on column stage_runs.input_fingerprint is
    '节点输入与版本计算出的稳定指纹，用于幂等和缓存判定。';
comment on column stage_runs.input_payload is
    '节点声明时写入的非 artifact JSON 输入。';
comment on column stage_runs.input_artifacts is
    '节点所需 artifact 的声明性绑定，运行时由上游产物引用填充。';
comment on column stage_runs.output_payload is
    '节点成功后保存的结构化 JSON 输出摘要。';
comment on column stage_runs.available_at is
    '节点最早可被领取的时间；失败重试通过该字段实现退避等待。';
comment on column stage_runs.started_at is
    '节点首次开始执行的时间。';
comment on column stage_runs.finished_at is
    '节点最终成功、失败、取消或跳过的时间。';
comment on column stage_runs.error_message is
    '最近一次执行失败的错误信息，长度由应用层限制。';
comment on column stage_runs.created_at is
    '节点运行记录创建时间。';
comment on column stage_runs.updated_at is
    '节点运行记录最近更新时间。';

create table if not exists stage_run_dependencies (
    stage_run_id uuid not null references stage_runs(id) on delete cascade,
    depends_on_stage_run_id uuid not null references stage_runs(id) on delete cascade,
    primary key (stage_run_id, depends_on_stage_run_id),
    check (stage_run_id <> depends_on_stage_run_id)
);

comment on table stage_run_dependencies is
    '流水线节点依赖关系表，表示一个 stage_run 必须在其上游 stage_run 成功后才能开始执行。';
comment on column stage_run_dependencies.stage_run_id is
    '下游节点运行 ID；该节点等待 depends_on_stage_run_id 成功。';
comment on column stage_run_dependencies.depends_on_stage_run_id is
    '上游依赖节点运行 ID。';

create table if not exists artifacts (
    id uuid primary key default gen_random_uuid(),
    subject_type text not null,
    subject_id uuid not null,
    pipeline_run_id uuid not null references pipeline_runs(id) on delete cascade,
    stage_run_id uuid references stage_runs(id) on delete set null,
    artifact_type text not null,
    artifact_version text not null,
    uri text not null,
    checksum text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (stage_run_id, artifact_type, artifact_version)
);

comment on table artifacts is
    '流水线产物表，保存任意业务流水线 stage 输出的存储引用与元数据。';
comment on column artifacts.id is
    '产物记录主键。';
comment on column artifacts.subject_type is
    '业务对象类型；与 subject_id 共同定位产物所属对象。';
comment on column artifacts.subject_id is
    '业务对象主键；不建立跨领域外键，保持 artifact runtime 通用。';
comment on column artifacts.pipeline_run_id is
    '关联 pipeline_runs.id，表示产物所属的一次处理运行。';
comment on column artifacts.stage_run_id is
    '关联生成该产物的 stage_runs.id；原始输入产物可为空。';
comment on column artifacts.artifact_type is
    '产物逻辑类型，例如 audio.normalized、diarization.pyannote 或 transcript.qwen_asr。';
comment on column artifacts.artifact_version is
    '产物数据格式版本。';
comment on column artifacts.uri is
    '相对于 artifact 存储根目录的路径或对象存储键。';
comment on column artifacts.checksum is
    '产物内容校验和；可为空。';
comment on column artifacts.metadata is
    '产物附加元数据，例如时长、分段数量或存储属性。';
comment on column artifacts.created_at is
    '产物记录创建时间。';

create table if not exists pipeline_events (
    id bigserial primary key,
    pipeline_run_id uuid not null references pipeline_runs(id) on delete cascade,
    stage_run_id uuid references stage_runs(id) on delete cascade,
    event_type text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

comment on table pipeline_events is
    '流水线领域事件表，按时间记录 run 与 stage 的排队、领取、开始、成功、失败和重试等状态变化。';
comment on column pipeline_events.id is
    '按插入顺序递增的事件主键，供事件流稳定排序。';
comment on column pipeline_events.pipeline_run_id is
    '关联发生事件的 pipeline_runs.id。';
comment on column pipeline_events.stage_run_id is
    '关联发生事件的 stage_runs.id；流水线级事件可为空。';
comment on column pipeline_events.event_type is
    '领域事件类型，例如 pipeline.queued、stage.running 或 stage.succeeded。';
comment on column pipeline_events.payload is
    '事件附带的结构化 JSON 数据。';
comment on column pipeline_events.created_at is
    '事件写入时间。';

create table if not exists outbox_events (
    id uuid primary key default gen_random_uuid(),
    topic text not null,
    aggregate_type text not null,
    aggregate_id uuid not null,
    payload jsonb not null,
    available_at timestamptz not null default now(),
    published_at timestamptz,
    created_at timestamptz not null default now()
);

comment on table outbox_events is
    '事务外盒事件表，用于在数据库事务提交后可靠投递 pipeline run 创建等跨进程消息。';
comment on column outbox_events.id is
    '外盒事件主键。';
comment on column outbox_events.topic is
    '消息主题，例如 pipeline.run.created。';
comment on column outbox_events.aggregate_type is
    '事件所属聚合类型，例如 pipeline_run。';
comment on column outbox_events.aggregate_id is
    '事件所属聚合的主键。';
comment on column outbox_events.payload is
    '等待投递的结构化 JSON 消息体。';
comment on column outbox_events.available_at is
    '事件最早允许投递的时间，用于延迟投递和重试退避。';
comment on column outbox_events.published_at is
    '成功投递的时间；为空表示仍待投递。';
comment on column outbox_events.created_at is
    '外盒事件创建时间。';


-- Generation runtime
-- 独立于流水线的通用生成任务基座。录音总结、录音问答等长文本生成都通过它保存可恢复快照和流式事件。
create table if not exists generation_runs (
    id uuid primary key default gen_random_uuid(),
    kind text not null check (kind in ('text')),
    priority text not null check (priority in ('interactive', 'background')),
    idempotency_key text not null unique,
    parent_type text,
    parent_id text,
    owner_user_id uuid references users(id) on delete set null,
    subject_type text,
    subject_id uuid,
    status text not null default 'queued' check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    input_payload jsonb not null default '{}'::jsonb,
    phase jsonb,
    progress_percent integer check (progress_percent is null or (progress_percent >= 0 and progress_percent <= 100)),
    output_blocks jsonb not null default '[]'::jsonb,
    output_payload jsonb,
    last_sequence bigint not null default 0 check (last_sequence >= 0),
    first_token_at timestamptz,
    cancel_requested boolean not null default false,
    error_code text,
    error_message text,
    created_at timestamptz not null default now(),
    started_at timestamptz,
    finished_at timestamptz,
    updated_at timestamptz not null default now(),
    check ((subject_type is null) = (subject_id is null))
);

comment on table generation_runs is
    '通用生成任务主表，保存录音总结、问答等长文本生成的当前快照、状态和可恢复的输出内容。';
comment on column generation_runs.id is
    '生成任务主键，使用 UUID，供 API、SSE 连接和事件表关联。';
comment on column generation_runs.kind is
    '生成内容类型。当前统一为 text，业务模块可在通用运行时之上定义自己的生成用例。';
comment on column generation_runs.priority is
    '调度优先级。interactive 用于用户等待中的问答，background 用于总结等后台生成。';
comment on column generation_runs.idempotency_key is
    '创建命令的幂等键。同一个键只能对应一个生成任务，避免浏览器或 worker 重试造成重复生成。';
comment on column generation_runs.parent_type is
    '父业务对象类型，例如 stage_run、conversation 或 message。当前不建立外键，以允许不同业务聚合复用。';
comment on column generation_runs.parent_id is
    '父业务对象标识，与 parent_type 共同指向本次生成所属的业务对象。';
comment on column generation_runs.owner_user_id is
    '生成任务的直接所有者。用于用户范围的通用访问控制；删除用户时保留任务但清空该字段。';
comment on column generation_runs.subject_type is
    '受保护业务对象的类型，例如 recording；与 subject_id 共同交由 access 层完成授权校验。';
comment on column generation_runs.subject_id is
    '受保护业务对象主键；不建立跨领域外键，以保持 generation runtime 通用。';
comment on column generation_runs.status is
    '生成任务状态：queued、running、succeeded、failed 或 cancelled。';
comment on column generation_runs.input_payload is
    '创建时冻结的类型化输入 JSON，例如问答 query、过滤条件和 artifact 引用；不承担授权范围。';
comment on column generation_runs.phase is
    '当前用户可见阶段 JSON，包含稳定阶段名 name 和展示文案 label。';
comment on column generation_runs.progress_percent is
    '可选的任务进度百分比，范围为 0 到 100。';
comment on column generation_runs.output_blocks is
    '当前已持久化的完整内容块快照。第一版仅包含按顺序追加的 text block。';
comment on column generation_runs.output_payload is
    '生成完成后的结构化输出。问答的最终 sources 作为 JSON 数组保存在此字段，与最终 SSE 事件保持一致。';
comment on column generation_runs.last_sequence is
    '已成功写入快照和事件表的最大流事件序号；页面刷新和 SSE 续传的原子边界。';
comment on column generation_runs.first_token_at is
    '首个用户可见文本块被写入的时间，用于统计首 token 延迟。';
comment on column generation_runs.cancel_requested is
    '是否已请求协作取消。执行器应在 token 或 chunk 边界检查并结束任务。';
comment on column generation_runs.error_code is
    '失败的稳定错误码，便于前端判断是否可重试和聚合观测。';
comment on column generation_runs.error_message is
    '面向用户和排障的失败信息摘要。';
comment on column generation_runs.created_at is
    '生成任务创建时间。';
comment on column generation_runs.started_at is
    '执行器开始运行任务的时间。';
comment on column generation_runs.finished_at is
    '任务进入成功、失败或取消终态的时间。';
comment on column generation_runs.updated_at is
    '任务快照最后更新时间。';

create table if not exists generation_events (
    id bigserial primary key,
    generation_run_id uuid not null references generation_runs(id) on delete cascade,
    sequence bigint not null check (sequence > 0),
    event_type text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (generation_run_id, sequence)
);

comment on table generation_events is
    '生成任务的追加式流事件表。与 generation_runs 的快照共同支持 SSE 重连和页面刷新恢复。';
comment on column generation_events.id is
    '数据库事件主键，按插入顺序递增，主要用于运维排障。';
comment on column generation_events.generation_run_id is
    '关联 generation_runs.id，表示事件所属生成任务。';
comment on column generation_events.sequence is
    '同一生成任务内严格递增的协议序号，SSE id 和客户端去重依据。';
comment on column generation_events.event_type is
    '协议事件类型，例如 run.status、phase、content.delta、output.final、run.error 或 run.cancelled。';
comment on column generation_events.payload is
    '事件携带的 JSON 数据，不包含 envelope 的 run_id、sequence 和时间字段。';
comment on column generation_events.created_at is
    '事件持久化时间。';

create index if not exists generation_runs_parent_idx on generation_runs (parent_type, parent_id);
create index if not exists generation_runs_status_priority_idx on generation_runs (status, priority, created_at);
create index if not exists generation_runs_owner_idx on generation_runs (owner_user_id, created_at desc) where owner_user_id is not null;
create index if not exists generation_runs_subject_idx on generation_runs (subject_type, subject_id, created_at desc) where subject_id is not null;
create index if not exists generation_events_run_sequence_idx on generation_events (generation_run_id, sequence);

-- conversations
-- 多轮录音问答的长期会话。Generation 仅表示其中一条助手消息的一次流式执行。
create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete cascade,
    owner_user_id uuid not null references users(id) on delete restrict,
    title text not null default '新对话',
    next_message_sequence bigint not null default 0 check (next_message_sequence >= 0),
    archived_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table conversations is '多轮录音问答的长期会话，归属于一个工作区。';
comment on column conversations.id is '会话主键。';
comment on column conversations.workspace_id is '会话所属工作区，也是默认访问授权边界。';
comment on column conversations.owner_user_id is '创建会话的用户，用于审计与默认展示。';
comment on column conversations.title is '会话展示标题；首版可使用用户首条提问自动生成。';
comment on column conversations.next_message_sequence is '会话内下一次消息写入使用的原子计数器；每轮问答递增 2。';
comment on column conversations.archived_at is '归档时间；非空时默认不出现在活跃会话列表。';
comment on column conversations.created_at is '会话创建时间。';
comment on column conversations.updated_at is '会话最后更新时间。';

create table if not exists conversation_messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references conversations(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    sequence bigint not null check (sequence > 0),
    reply_to_message_id uuid references conversation_messages(id) on delete set null,
    content_blocks jsonb not null default '[]'::jsonb,
    sources jsonb not null default '[]'::jsonb,
    generation_run_id uuid unique references generation_runs(id) on delete set null,
    status text not null check (status in ('pending', 'streaming', 'completed', 'failed', 'cancelled')),
    client_message_id uuid,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (conversation_id, sequence),
    unique (conversation_id, client_message_id),
    check ((role = 'user' and reply_to_message_id is null and generation_run_id is null)
        or (role = 'assistant' and reply_to_message_id is not null))
);

comment on table conversation_messages is '会话内长期消息。每轮写入一条 user 消息和一条 assistant 占位消息。';
comment on column conversation_messages.id is '消息主键。';
comment on column conversation_messages.conversation_id is '消息所属会话。';
comment on column conversation_messages.role is '消息角色：user 表示上行提问，assistant 表示下行回答。';
comment on column conversation_messages.sequence is '会话内严格递增的展示与分页顺序；不依赖时间戳。';
comment on column conversation_messages.reply_to_message_id is 'assistant 所回复的 user 消息；user 消息为空。';
comment on column conversation_messages.content_blocks is '标准 Generation block 协议的最终正文。';
comment on column conversation_messages.sources is 'assistant 回答最终使用的录音资料范围。';
comment on column conversation_messages.generation_run_id is '生成该 assistant 消息的通用 Generation run；user 消息为空。';
comment on column conversation_messages.status is '消息状态：user 通常为 completed；assistant 为 pending、streaming 或终态。';
comment on column conversation_messages.client_message_id is '浏览器发送动作的幂等 UUID；仅 user 消息使用。';
comment on column conversation_messages.error_message is 'assistant 生成失败时的用户可见错误摘要。';
comment on column conversation_messages.created_at is '消息创建时间。';
comment on column conversation_messages.updated_at is '消息最后更新时间。';

create index if not exists conversations_workspace_updated_idx on conversations (workspace_id, updated_at desc) where archived_at is null;
create index if not exists conversation_messages_conversation_sequence_idx on conversation_messages (conversation_id, sequence desc);
create index if not exists conversation_messages_active_assistant_idx on conversation_messages (conversation_id)
    where role = 'assistant' and status in ('pending', 'streaming');

create index if not exists stage_runs_claim_idx on stage_runs (resource_queue, status, available_at, created_at);
create index if not exists stage_runs_pipeline_idx on stage_runs (pipeline_run_id, status);
create index if not exists stage_run_dependencies_upstream_idx on stage_run_dependencies (depends_on_stage_run_id);
create index if not exists pipeline_runs_subject_idx on pipeline_runs (subject_type, subject_id, created_at desc);
create index if not exists pipeline_events_run_idx on pipeline_events (pipeline_run_id, id);
create index if not exists outbox_events_pending_idx on outbox_events (available_at) where published_at is null;

-- Optional helper trigger target for app layer:
-- update updated_at on row modifications in application code or via triggers later.
