-- ASR Lab and generic evaluation schema.
--
-- This schema is separate from sql/base.sql, but it is installed into the
-- same PostgreSQL database and public schema. It requires the base schema's
-- users, workspaces, and recordings tables.
--
-- The file is additive and idempotent: no table is dropped and no row is
-- deleted.

create table if not exists evaluation_datasets (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete restrict,
    name text not null,
    description text,
    task_type text not null check (task_type in ('asr')),
    status text not null default 'active' check (status in ('active', 'archived')),
    created_by_user_id uuid not null references users(id) on delete restrict,
    archived_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (length(btrim(name)) > 0),
    check (
        (status = 'active' and archived_at is null)
        or (status = 'archived' and archived_at is not null)
    )
);

comment on table evaluation_datasets is
    '评测数据集主表。保存持续维护的标注集合；训练和评测必须引用其不可变版本。';
comment on column evaluation_datasets.task_type is
    '评测任务类型。第一版仅支持 asr，后续可扩展 RAG 和全链路任务。';
comment on column evaluation_datasets.status is
    '数据集状态。archived 仅隐藏数据集，不删除历史版本和实验结果。';

create index if not exists evaluation_datasets_workspace_updated_idx
    on evaluation_datasets (workspace_id, updated_at desc);
create unique index if not exists evaluation_datasets_workspace_active_name_uidx
    on evaluation_datasets (workspace_id, lower(name))
    where status = 'active';

-- Exactly one source is present:
-- - recording_id imports an existing production recording;
-- - artifact_uri points to a persisted ASR sample slice (or a legacy ASR Lab upload).
create table if not exists evaluation_source_assets (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete restrict,
    dataset_id uuid not null references evaluation_datasets(id) on delete restrict,
    recording_id uuid references recordings(id) on delete restrict,
    artifact_uri text,
    checksum text not null,
    file_name text not null,
    mime_type text not null,
    file_size_bytes bigint not null check (file_size_bytes >= 0),
    duration_ms bigint not null check (duration_ms > 0),
    metadata jsonb not null default '{}'::jsonb,
    created_by_user_id uuid not null references users(id) on delete restrict,
    archived_at timestamptz,
    created_at timestamptz not null default now(),
    check (num_nonnulls(recording_id, artifact_uri) = 1),
    check (artifact_uri is null or length(btrim(artifact_uri)) > 0),
    check (length(btrim(checksum)) > 0),
    check (length(btrim(file_name)) > 0),
    check (length(btrim(mime_type)) > 0),
    check (jsonb_typeof(metadata) = 'object')
);

comment on table evaluation_source_assets is
    'ASR Lab 持久化音频资产。新标注保存独立 FLAC 切片；历史数据可能仍引用完整录音。';
comment on column evaluation_source_assets.recording_id is
    '导入已有录音时使用；与 artifact_uri 必须且只能存在一个。';
comment on column evaluation_source_assets.artifact_uri is
    'ASR Lab 独立上传文件的稳定存储地址；与 recording_id 必须且只能存在一个。';
comment on column evaluation_source_assets.checksum is
    '完整源音频校验和。冻结版本、训练和评测执行时用于验证数据未发生变化。';

create index if not exists evaluation_source_assets_workspace_created_idx
    on evaluation_source_assets (workspace_id, created_at desc);
create index if not exists evaluation_source_assets_dataset_created_idx
    on evaluation_source_assets (dataset_id, created_at desc);
create index if not exists evaluation_source_assets_recording_idx
    on evaluation_source_assets (dataset_id, recording_id)
    where recording_id is not null;
create index if not exists evaluation_source_assets_checksum_idx
    on evaluation_source_assets (workspace_id, checksum);

-- Editing approved content must increment revision and return status to draft
-- in the same application transaction.
create table if not exists evaluation_annotations (
    id uuid primary key default gen_random_uuid(),
    dataset_id uuid not null references evaluation_datasets(id) on delete restrict,
    source_asset_id uuid not null references evaluation_source_assets(id) on delete restrict,
    start_ms bigint not null check (start_ms >= 0),
    end_ms bigint not null check (end_ms > 0),
    reference_text text not null,
    language text,
    status text not null default 'draft' check (status in ('draft', 'reviewed', 'approved')),
    train_allowed boolean not null default true,
    evaluation_allowed boolean not null default true,
    contains_sensitive_data boolean not null default false,
    group_key text not null,
    revision integer not null default 1 check (revision > 0),
    reviewed_by_user_id uuid references users(id) on delete restrict,
    reviewed_at timestamptz,
    approved_by_user_id uuid references users(id) on delete restrict,
    approved_at timestamptz,
    created_by_user_id uuid not null references users(id) on delete restrict,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (start_ms < end_ms),
    check (length(btrim(reference_text)) > 0),
    check (language is null or length(btrim(language)) > 0),
    check (length(btrim(group_key)) > 0),
    check ((reviewed_by_user_id is null) = (reviewed_at is null)),
    check ((approved_by_user_id is null) = (approved_at is null)),
    check (
        (status = 'draft'
            and reviewed_by_user_id is null
            and approved_by_user_id is null)
        or (status = 'reviewed'
            and reviewed_by_user_id is not null
            and approved_by_user_id is null)
        or (status = 'approved'
            and reviewed_by_user_id is not null
            and approved_by_user_id is not null)
    )
);

comment on table evaluation_annotations is
    '人工音频区间与参考文本的可编辑工作区。只有 approved 且用途允许的数据可以被冻结为训练或评测 case。';
comment on column evaluation_annotations.start_ms is
    '相对完整源音频起点的毫秒偏移，包含起点。';
comment on column evaluation_annotations.end_ms is
    '相对完整源音频起点的毫秒偏移，不包含终点。';
comment on column evaluation_annotations.group_key is
    '防数据泄漏分组键。同一 group_key 的 case 冻结时必须进入同一个 split。';
comment on column evaluation_annotations.revision is
    '乐观锁版本。更新必须携带当前 revision，成功后递增。';

create index if not exists evaluation_annotations_dataset_status_idx
    on evaluation_annotations (dataset_id, status, updated_at desc);
create index if not exists evaluation_annotations_asset_timeline_idx
    on evaluation_annotations (source_asset_id, start_ms, end_ms);
create index if not exists evaluation_annotations_dataset_group_idx
    on evaluation_annotations (dataset_id, group_key);

create table if not exists evaluation_dataset_versions (
    id uuid primary key default gen_random_uuid(),
    dataset_id uuid not null references evaluation_datasets(id) on delete restrict,
    version_number integer not null check (version_number > 0),
    status text not null default 'building' check (status in ('building', 'frozen')),
    normalization_name text not null,
    normalization_version text not null,
    split_strategy jsonb not null default '{}'::jsonb,
    case_count integer not null default 0 check (case_count >= 0),
    checksum text,
    created_by_user_id uuid not null references users(id) on delete restrict,
    frozen_at timestamptz,
    created_at timestamptz not null default now(),
    unique (dataset_id, version_number),
    check (length(btrim(normalization_name)) > 0),
    check (length(btrim(normalization_version)) > 0),
    check (jsonb_typeof(split_strategy) = 'object'),
    check (
        (status = 'building' and checksum is null and frozen_at is null)
        or (status = 'frozen'
            and checksum is not null
            and length(btrim(checksum)) > 0
            and frozen_at is not null)
    )
);

comment on table evaluation_dataset_versions is
    '不可变评测数据集版本。frozen 后 case、切分、标准化版本和 checksum 均不得修改。';
comment on column evaluation_dataset_versions.split_strategy is
    '冻结时使用的分组切分算法、比例和随机种子的完整快照。';
comment on column evaluation_dataset_versions.checksum is
    '覆盖 case、音频 checksum、区间、参考文本、split 和标准化版本的稳定摘要。';

create index if not exists evaluation_dataset_versions_dataset_created_idx
    on evaluation_dataset_versions (dataset_id, created_at desc);
create unique index if not exists evaluation_dataset_versions_dataset_checksum_uidx
    on evaluation_dataset_versions (dataset_id, checksum)
    where status = 'frozen';

-- Frozen snapshots intentionally duplicate annotation fields, so later edits
-- do not change historical training or evaluation runs.
create table if not exists evaluation_cases (
    id uuid primary key default gen_random_uuid(),
    dataset_version_id uuid not null references evaluation_dataset_versions(id) on delete restrict,
    source_annotation_id uuid not null references evaluation_annotations(id) on delete restrict,
    source_asset_id uuid not null references evaluation_source_assets(id) on delete restrict,
    start_ms bigint not null check (start_ms >= 0),
    end_ms bigint not null check (end_ms > 0),
    reference_text_raw text not null,
    reference_text_normalized text not null,
    language text,
    split text not null check (split in ('train', 'validation', 'test')),
    group_key text not null,
    train_allowed boolean not null,
    evaluation_allowed boolean not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (dataset_version_id, source_annotation_id),
    check (start_ms < end_ms),
    check (length(btrim(reference_text_raw)) > 0),
    check (length(btrim(reference_text_normalized)) > 0),
    check (language is null or length(btrim(language)) > 0),
    check (length(btrim(group_key)) > 0),
    check (split <> 'train' or train_allowed),
    check (split = 'train' or evaluation_allowed),
    check (jsonb_typeof(metadata) = 'object')
);

comment on table evaluation_cases is
    '冻结版本中的不可变样本快照。训练和评测读取本表，不直接读取持续变化的 annotation。';
comment on column evaluation_cases.reference_text_raw is
    '冻结时的人工确认原文。';
comment on column evaluation_cases.reference_text_normalized is
    '按数据集版本指定规则生成的参考文本，用于对应版本的指标计算。';
comment on column evaluation_cases.split is
    '样本用途：train、validation 或 test；同一 group_key 不得跨 split，由冻结服务校验。';

create index if not exists evaluation_cases_version_split_idx
    on evaluation_cases (dataset_version_id, split, id);
create index if not exists evaluation_cases_version_group_idx
    on evaluation_cases (dataset_version_id, group_key);
create index if not exists evaluation_cases_asset_timeline_idx
    on evaluation_cases (source_asset_id, start_ms, end_ms);

-- Database-level guard for the immutable snapshot contract. A version may
-- transition from building to frozen once; frozen version metadata and its
-- cases cannot then be inserted, updated, or deleted.
create or replace function reject_frozen_evaluation_version_mutation()
returns trigger
language plpgsql
as $$
begin
    if old.status = 'frozen' then
        raise exception 'evaluation dataset version % is frozen and immutable', old.id;
    end if;
    if tg_op = 'DELETE' then
        return old;
    end if;
    return new;
end;
$$;

create or replace function reject_frozen_evaluation_case_mutation()
returns trigger
language plpgsql
as $$
declare
    old_version_status text;
    new_version_status text;
begin
    if tg_op in ('UPDATE', 'DELETE') then
        select status
        into old_version_status
        from evaluation_dataset_versions
        where id = old.dataset_version_id;

        if old_version_status = 'frozen' then
            raise exception 'evaluation dataset version % is frozen and its cases are immutable', old.dataset_version_id;
        end if;
    end if;

    if tg_op in ('INSERT', 'UPDATE') then
        select status
        into new_version_status
        from evaluation_dataset_versions
        where id = new.dataset_version_id;

        if new_version_status = 'frozen' then
            raise exception 'evaluation dataset version % is frozen and its cases are immutable', new.dataset_version_id;
        end if;
    end if;

    if tg_op = 'DELETE' then
        return old;
    end if;
    return new;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_trigger
        where tgname = 'evaluation_dataset_versions_immutable_trigger'
          and tgrelid = 'evaluation_dataset_versions'::regclass
          and not tgisinternal
    ) then
        execute
            'create trigger evaluation_dataset_versions_immutable_trigger
             before update or delete on evaluation_dataset_versions
             for each row execute function reject_frozen_evaluation_version_mutation()';
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'evaluation_cases_immutable_trigger'
          and tgrelid = 'evaluation_cases'::regclass
          and not tgisinternal
    ) then
        execute
            'create trigger evaluation_cases_immutable_trigger
             before insert or update or delete on evaluation_cases
             for each row execute function reject_frozen_evaluation_case_mutation()';
    end if;
end $$;

create table if not exists model_versions (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete restrict,
    model_family text not null,
    name text not null,
    version text not null,
    base_model_name text not null,
    adapter_uri text,
    merged_model_uri text,
    training_run_id uuid,
    status text not null default 'candidate'
        check (status in ('candidate', 'validated', 'approved', 'deployed', 'retired')),
    runtime_config jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_by_user_id uuid not null references users(id) on delete restrict,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (workspace_id, model_family, name, version),
    check (length(btrim(model_family)) > 0),
    check (length(btrim(name)) > 0),
    check (length(btrim(version)) > 0),
    check (length(btrim(base_model_name)) > 0),
    check (adapter_uri is null or length(btrim(adapter_uri)) > 0),
    check (merged_model_uri is null or length(btrim(merged_model_uri)) > 0),
    check (jsonb_typeof(runtime_config) = 'object'),
    check (jsonb_typeof(metadata) = 'object')
);

comment on table model_versions is
    '通用模型版本注册表。基础模型、LoRA adapter 和合并模型都通过同一 ID 参与评测。';
comment on column model_versions.training_run_id is
    '产出该候选模型的训练任务。基础模型为空；外键在 training_runs 创建后补充。';
comment on column model_versions.status is
    '候选模型生命周期。第一版不因 approved 自动发布生产。';

create index if not exists model_versions_workspace_family_status_idx
    on model_versions (workspace_id, model_family, status, updated_at desc);
create unique index if not exists model_versions_training_run_uidx
    on model_versions (training_run_id)
    where training_run_id is not null;

create table if not exists training_runs (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete restrict,
    dataset_version_id uuid not null references evaluation_dataset_versions(id) on delete restrict,
    base_model_version_id uuid not null references model_versions(id) on delete restrict,
    status text not null default 'queued'
        check (status in ('queued', 'preparing', 'training', 'validating', 'succeeded', 'failed', 'cancelled')),
    training_method text not null check (training_method in ('lora')),
    preset_name text not null,
    candidate_model_name text not null,
    idempotency_key text not null,
    config_snapshot jsonb not null default '{}'::jsonb,
    code_commit text,
    environment_snapshot jsonb not null default '{}'::jsonb,
    output_uri text,
    progress_percent integer check (progress_percent is null or (progress_percent >= 0 and progress_percent <= 100)),
    progress_message text,
    cancel_requested boolean not null default false,
    started_at timestamptz,
    finished_at timestamptz,
    error_message text,
    created_by_user_id uuid not null references users(id) on delete restrict,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (workspace_id, idempotency_key),
    check (length(btrim(preset_name)) > 0),
    check (length(btrim(candidate_model_name)) > 0),
    check (length(btrim(idempotency_key)) > 0),
    check (jsonb_typeof(config_snapshot) = 'object'),
    check (jsonb_typeof(environment_snapshot) = 'object'),
    check (output_uri is null or length(btrim(output_uri)) > 0)
);

comment on table training_runs is
    '独立 GPU training worker 执行的模型训练任务，保存数据、配置、代码和环境快照。';
comment on column training_runs.idempotency_key is
    '工作区内唯一的客户端幂等键，防止重复点击创建多个训练任务。';
comment on column training_runs.config_snapshot is
    '基础模型 revision、LoRA 配置、超参数、随机种子和训练输入的不可变快照。';
comment on column training_runs.cancel_requested is
    '协作式取消标记；worker 在安全点终止并写入 cancelled 终态。';

create index if not exists training_runs_claim_idx
    on training_runs (status, created_at)
    where status = 'queued';
create index if not exists training_runs_workspace_created_idx
    on training_runs (workspace_id, created_at desc);
create index if not exists training_runs_dataset_idx
    on training_runs (dataset_version_id, created_at desc);

-- Add the circular provenance relationship after training_runs exists.
do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'model_versions_training_run_id_fkey'
          and conrelid = 'model_versions'::regclass
    ) then
        alter table model_versions
            add constraint model_versions_training_run_id_fkey
            foreign key (training_run_id) references training_runs(id) on delete set null;
    end if;
end $$;

create table if not exists evaluation_runs (
    id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references workspaces(id) on delete restrict,
    dataset_version_id uuid not null references evaluation_dataset_versions(id) on delete restrict,
    evaluator_type text not null,
    split text not null check (split in ('validation', 'test')),
    status text not null default 'queued'
        check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    idempotency_key text not null,
    config_snapshot jsonb not null default '{}'::jsonb,
    code_commit text,
    total_case_count integer not null default 0 check (total_case_count >= 0),
    completed_case_count integer not null default 0 check (completed_case_count >= 0),
    failed_case_count integer not null default 0 check (failed_case_count >= 0),
    cancel_requested boolean not null default false,
    started_at timestamptz,
    finished_at timestamptz,
    error_message text,
    created_by_user_id uuid not null references users(id) on delete restrict,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (workspace_id, idempotency_key),
    check (length(btrim(evaluator_type)) > 0),
    check (length(btrim(idempotency_key)) > 0),
    check (jsonb_typeof(config_snapshot) = 'object'),
    check (completed_case_count + failed_case_count <= total_case_count)
);

comment on table evaluation_runs is
    '一次可复现评测任务。数据集版本、split、标准化和推理配置在创建时冻结。';
comment on column evaluation_runs.split is
    '评测只允许 validation 或 test；train 不作为正式评测输入。';
comment on column evaluation_runs.config_snapshot is
    '标准化版本、推理参数、metric 版本等不可变配置。';

create index if not exists evaluation_runs_claim_idx
    on evaluation_runs (status, created_at)
    where status = 'queued';
create index if not exists evaluation_runs_workspace_created_idx
    on evaluation_runs (workspace_id, created_at desc);
create index if not exists evaluation_runs_dataset_idx
    on evaluation_runs (dataset_version_id, split, created_at desc);

-- Normalized relation gives model IDs real foreign keys, unlike uuid[].
create table if not exists evaluation_run_models (
    evaluation_run_id uuid not null references evaluation_runs(id) on delete cascade,
    model_version_id uuid not null references model_versions(id) on delete restrict,
    role text not null check (role in ('baseline', 'candidate', 'peer')),
    position integer not null check (position >= 0),
    created_at timestamptz not null default now(),
    primary key (evaluation_run_id, model_version_id),
    unique (evaluation_run_id, position)
);

comment on table evaluation_run_models is
    '评测任务的模型集合。role 和 position 决定前端基准/对比展示，不使用无法加外键的 uuid 数组。';

create unique index if not exists evaluation_run_models_one_baseline_uidx
    on evaluation_run_models (evaluation_run_id)
    where role = 'baseline';
create index if not exists evaluation_run_models_model_idx
    on evaluation_run_models (model_version_id, created_at desc);

-- Empty ASR text is a valid successful hypothesis and is stored as an empty
-- string, not NULL.
create table if not exists evaluation_case_results (
    id uuid primary key default gen_random_uuid(),
    evaluation_run_id uuid not null,
    model_version_id uuid not null,
    evaluation_case_id uuid not null references evaluation_cases(id) on delete restrict,
    hypothesis_text_raw text,
    hypothesis_text_normalized text,
    inference_duration_ms integer check (inference_duration_ms is null or inference_duration_ms >= 0),
    status text not null check (status in ('succeeded', 'failed')),
    error_message text,
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (evaluation_run_id, model_version_id, evaluation_case_id),
    foreign key (evaluation_run_id, model_version_id)
        references evaluation_run_models(evaluation_run_id, model_version_id)
        on delete cascade,
    check (jsonb_typeof(details) = 'object'),
    check (
        (status = 'succeeded'
            and hypothesis_text_raw is not null
            and hypothesis_text_normalized is not null
            and error_message is null)
        or (status = 'failed'
            and hypothesis_text_raw is null
            and hypothesis_text_normalized is null
            and error_message is not null)
    )
);

comment on table evaluation_case_results is
    '指定模型对一个冻结 case 的原始 ASR 输出、标准化输出、推理耗时和可渲染差异详情。';
comment on column evaluation_case_results.details is
    '字符或词级编辑操作等诊断信息；前端 diff 只渲染该结果，不重新计算指标。';

create index if not exists evaluation_case_results_run_model_status_idx
    on evaluation_case_results (evaluation_run_id, model_version_id, status, evaluation_case_id);
create index if not exists evaluation_case_results_case_idx
    on evaluation_case_results (evaluation_case_id, created_at desc);

-- Metric scopes:
-- - run: model_version_id and evaluation_case_id are both NULL;
-- - model: model_version_id is set and evaluation_case_id is NULL;
-- - case: model_version_id and evaluation_case_id are both set.
create table if not exists evaluation_metric_values (
    id uuid primary key default gen_random_uuid(),
    evaluation_run_id uuid not null references evaluation_runs(id) on delete cascade,
    model_version_id uuid references model_versions(id) on delete restrict,
    evaluation_case_id uuid references evaluation_cases(id) on delete restrict,
    metric_name text not null,
    metric_version text not null,
    value numeric not null,
    sample_count integer check (sample_count is null or sample_count >= 0),
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    foreign key (evaluation_run_id, model_version_id)
        references evaluation_run_models(evaluation_run_id, model_version_id)
        on delete cascade,
    foreign key (evaluation_run_id, model_version_id, evaluation_case_id)
        references evaluation_case_results(evaluation_run_id, model_version_id, evaluation_case_id)
        on delete cascade,
    check (evaluation_case_id is null or model_version_id is not null),
    check (length(btrim(metric_name)) > 0),
    check (length(btrim(metric_version)) > 0),
    check (jsonb_typeof(details) = 'object')
);

comment on table evaluation_metric_values is
    '版本化评测指标。支持 run、model 和 case 粒度；总体 CER/WER 由后端 micro aggregation 生成。';
comment on column evaluation_metric_values.details is
    '替换、删除、插入、参考 token 数、分桶等指标诊断明细。';

create unique index if not exists evaluation_metric_values_run_scope_uidx
    on evaluation_metric_values (evaluation_run_id, metric_name, metric_version)
    where model_version_id is null and evaluation_case_id is null;
create unique index if not exists evaluation_metric_values_model_scope_uidx
    on evaluation_metric_values (evaluation_run_id, model_version_id, metric_name, metric_version)
    where model_version_id is not null and evaluation_case_id is null;
create unique index if not exists evaluation_metric_values_case_scope_uidx
    on evaluation_metric_values (
        evaluation_run_id,
        model_version_id,
        evaluation_case_id,
        metric_name,
        metric_version
    )
    where model_version_id is not null and evaluation_case_id is not null;
create index if not exists evaluation_metric_values_case_idx
    on evaluation_metric_values (evaluation_case_id, metric_name)
    where evaluation_case_id is not null;
