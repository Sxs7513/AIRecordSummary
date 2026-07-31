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
    provider text not null,
    model_name text not null,
    summary_text text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (recording_id)
);

-- 兼容旧数据库：provider 的可用值由应用层维护，避免新增模型提供方时写入失败。
alter table recording_summaries
    drop constraint if exists recording_summaries_provider_check;

comment on table recording_summaries is
    '录音总结表，保存润色后文本生成的整条录音摘要。';
comment on column recording_summaries.recording_id is
    '关联 recordings.id，表示这份总结属于哪条录音。';
comment on column recording_summaries.provider is
    '总结使用的模型提供方标识，例如 local、zhipu、gemini；具体可用值由应用层校验。';
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

-- Generation terminal query projection
-- 活跃状态和流式事件位于 Redis；该表只由 Kafka 终态投影器写入。
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
    status text not null check (status in ('succeeded', 'failed', 'cancelled')),
    input_payload jsonb not null default '{}'::jsonb,
    output_payload jsonb,
    error_code text,
    error_message text,
    created_at timestamptz not null default now(),
    started_at timestamptz,
    finished_at timestamptz,
    updated_at timestamptz not null default now(),
    check ((subject_type is null) = (subject_id is null))
);

comment on table generation_runs is
    '通用生成任务终态查询投影；不承担排队、运行时状态或流式事件存储。';
comment on column generation_runs.id is
    '生成任务关联 ID，由命令入口生成，并由 Kafka 终态投影器幂等写入。';
comment on column generation_runs.kind is
    '生成内容类型。当前统一为 text，业务模块可在通用运行时之上定义自己的生成用例。';
comment on column generation_runs.priority is
    '命令创建时的调度优先级元数据；实际排队与调度由 Kafka Consumer 完成。';
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
    'Kafka 终态投影：succeeded、failed 或 cancelled；活跃状态只存在于 Redis/Kafka。';
comment on column generation_runs.input_payload is
    '创建时冻结的类型化输入 JSON，例如问答 query、过滤条件和 artifact 引用；不承担授权范围。';
comment on column generation_runs.output_payload is
    '生成完成后的结构化输出。问答的最终 sources 作为 JSON 数组保存在此字段，与最终 SSE 事件保持一致。';
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

create index if not exists generation_runs_parent_idx on generation_runs (parent_type, parent_id);
create index if not exists generation_runs_status_priority_idx on generation_runs (status, priority, created_at);
create index if not exists generation_runs_owner_idx on generation_runs (owner_user_id, created_at desc) where owner_user_id is not null;
create index if not exists generation_runs_subject_idx on generation_runs (subject_type, subject_id, created_at desc) where subject_id is not null;

-- RAG observability
-- Caller-generated UUIDs make ingestion idempotent across retries. Mutable records
-- preserve the running state even when a process exits before publishing a terminal update.
create table if not exists rag_execution_spans (
    id uuid primary key,
    workspace_id uuid not null references workspaces(id) on delete restrict,
    generation_run_id uuid not null,
    parent_span_id uuid references rag_execution_spans(id) on delete set null,
    component text not null default 'rag',
    operation text not null,
    operation_version text not null default '1',
    attempt integer not null default 0 check (attempt >= 0),
    status text not null check (status in ('running', 'succeeded', 'failed', 'cancelled', 'abandoned')),
    started_at timestamptz not null,
    finished_at timestamptz,
    elapsed_ms numeric,
    error_type text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (length(btrim(component)) > 0),
    check (length(btrim(operation)) > 0),
    check (length(btrim(operation_version)) > 0),
    check (jsonb_typeof(metadata) = 'object'),
    check (
        (status = 'running' and finished_at is null and elapsed_ms is null)
        or
        (status <> 'running' and finished_at is not null and elapsed_ms is not null and elapsed_ms >= 0)
    )
);

create table if not exists model_invocations (
    id uuid primary key,
    workspace_id uuid not null references workspaces(id) on delete restrict,
    generation_run_id uuid not null,
    -- Deliberately no FK: best-effort delivery may lose or reorder a Span record.
    span_id uuid,
    component text not null default 'rag',
    operation text not null,
    operation_version text not null default '1',
    attempt integer not null default 0 check (attempt >= 0),
    usage_kind text not null default 'llm',
    provider text not null,
    model text,
    stream boolean not null default false,
    status text not null check (status in ('running', 'succeeded', 'failed', 'cancelled', 'abandoned')),
    prompt_tokens integer check (prompt_tokens is null or prompt_tokens >= 0),
    completion_tokens integer check (completion_tokens is null or completion_tokens >= 0),
    cached_input_tokens integer check (cached_input_tokens is null or cached_input_tokens >= 0),
    reasoning_tokens integer check (reasoning_tokens is null or reasoning_tokens >= 0),
    usage_source text not null check (usage_source in ('provider', 'local_tokenizer', 'estimated', 'unavailable')),
    finish_reason text,
    provider_request_id text,
    error_type text,
    estimated_cost_micros bigint check (estimated_cost_micros is null or estimated_cost_micros >= 0),
    currency text,
    started_at timestamptz not null,
    finished_at timestamptz,
    elapsed_ms numeric,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (length(btrim(component)) > 0),
    check (length(btrim(operation)) > 0),
    check (length(btrim(operation_version)) > 0),
    check (length(btrim(usage_kind)) > 0),
    check (length(btrim(provider)) > 0),
    check (jsonb_typeof(metadata) = 'object'),
    check ((estimated_cost_micros is null) = (currency is null)),
    check (
        (status = 'running' and finished_at is null and elapsed_ms is null)
        or
        (status <> 'running' and finished_at is not null and elapsed_ms is not null and elapsed_ms >= 0)
    )
);

comment on table rag_execution_spans is 'RAG 动态执行节点的状态、耗时和非敏感指标。';
comment on table model_invocations is '模型逐调用状态及 Token usage；开始和终态使用同一调用 ID 幂等更新。';

create index if not exists rag_execution_spans_run_started_idx on rag_execution_spans (generation_run_id, started_at, id);
create index if not exists rag_execution_spans_workspace_started_idx on rag_execution_spans (workspace_id, started_at desc);
create index if not exists rag_execution_spans_operation_started_idx on rag_execution_spans (workspace_id, operation, started_at desc);
create index if not exists model_invocations_run_started_idx on model_invocations (generation_run_id, started_at, id);
create index if not exists model_invocations_workspace_started_idx on model_invocations (workspace_id, started_at desc);
create index if not exists model_invocations_operation_started_idx on model_invocations (workspace_id, operation, started_at desc);
create index if not exists model_invocations_provider_model_started_idx on model_invocations (workspace_id, provider, model, started_at desc);

-- conversations
-- 多轮录音问答的长期会话。Generation 仅表示其中一条助手消息的一次流式执行。
create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete cascade,
    owner_user_id uuid references users(id) on delete set null,
    client_creation_id uuid,
    title text not null default '新对话',
    next_message_sequence bigint not null default 0 check (next_message_sequence >= 0),
    archived_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- 删除会话只解除用户关联并归档，历史消息、Generation 与观测数据继续保留。
alter table conversations alter column owner_user_id drop not null;
alter table conversations drop constraint if exists conversations_owner_user_id_fkey;
alter table conversations
    add constraint conversations_owner_user_id_fkey
    foreign key (owner_user_id) references users(id) on delete set null;

alter table conversations add column if not exists client_creation_id uuid;

comment on table conversations is '多轮录音问答的长期会话，归属于一个工作区。';
comment on column conversations.id is '会话主键。';
comment on column conversations.workspace_id is '会话所属工作区，也是默认访问授权边界。';
comment on column conversations.owner_user_id is '创建会话的用户，用于审计与默认展示。';
comment on column conversations.client_creation_id is '浏览器创建首轮会话时生成的幂等 UUID，用于 POST SSE 断线重连。';
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
    generation_run_id uuid unique,
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
create unique index if not exists conversations_owner_client_creation_idx
    on conversations (owner_user_id, client_creation_id) where client_creation_id is not null;
create index if not exists conversation_messages_conversation_sequence_idx on conversation_messages (conversation_id, sequence desc);
create index if not exists conversation_messages_active_assistant_idx on conversation_messages (conversation_id)
    where role = 'assistant' and status in ('pending', 'streaming');


-- Optional helper trigger target for app layer:
-- update updated_at on row modifications in application code or via triggers later.
