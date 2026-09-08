# Qwen3-Embedding-0.6B 并行索引与召回评测方案

## 1. 背景与目标

当前项目默认使用 `Qwen/Qwen3-Embedding-4B`，向量维度为 2560。为了降低 Query Embedding 延迟、模型加载时间和显存占用，需要引入更小的 `Qwen/Qwen3-Embedding-0.6B`，在不破坏现有 4B 索引的前提下完成历史数据回填和同数据集召回评测。

Qwen 官方发布的同系列小模型是 `Qwen3-Embedding-0.6B`，不是 0.9B。0.6B 最大输出维度为 1024，4B 为 2560：

- [Qwen3-Embedding-0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3 Embedding 官方介绍](https://qwenlm.github.io/zh/blog/qwen3-embedding/)

本方案目标：

- 4B 和 0.6B 向量可以同时存在；
- 关键词索引和 SearchChunk 不因模型数量而重复；
- 可通过配置选择生产和评测使用的 embedding profile；
- 历史 Chunk 和录音画像可以断点续跑地批量回填；
- 安装阶段可以预下载 0.6B，运行时不临时联网；
- 4B 与 0.6B 使用同一批文本和评测 Case 公平对比；
- 切换失败时可以立即回到 4B，不需要重新生成历史向量。

## 2. 当前实现与问题

### 2.1 已有能力

当前项目已经具备部分模型版本意识：

- `embedding_models` 保存 provider、model name、dimensions 和 distance metric；
- 向量查询按 `model_name + dimensions` 过滤；
- 已有 `EMBEDDING_MODEL` 和 `EMBEDDING_DIMENSIONS` 配置；
- RAG Pipeline Version 会保存 embedding 模型快照；
- 安装脚本通过 `ensure_hf_snapshot.py` 下载并复用本地 Hugging Face snapshot；
- SearchChunk 保存了重建向量输入所需的文本及 topic、terms、search context 元数据。

### 2.2 当前阻碍

1. `recording_search_chunks.embedding` 固定为 `halfvec(2560)`，不能保存 1024 维向量；
2. `recording_retrieval_documents.embedding` 同样固定为 `halfvec(2560)`；
3. SearchChunk 重建时会删除该录音全部 chunk，旧模型向量无法并存；
4. 如果按模型复制整个 SearchChunk，关键词召回会返回重复文本；
5. Compute Worker 启动时只加载全局配置指定的一个 embedding 模型；
6. Embedding 请求只携带 texts，没有携带模型 profile，无法校验请求模型与 Worker 模型是否一致；
7. Corpus Snapshot 当前绑定 `embedding_model_id`，同一文本使用不同模型时可能被识别为不同语料快照；
8. installer 当前只确保 `EMBEDDING_MODEL` 对应的一个模型已经下载。

## 3. 核心决策

### 3.1 不采用“一模型一字段”

不建议在主表持续增加：

```text
embedding_4b
embedding_0_6b
embedding_8b
embedding_bge
```

该方式会把模型名称固化到数据库结构中。每增加一个模型都需要修改表、索引、查询、写入和回填逻辑。

### 3.2 Chunk 与 Embedding 分离

`recording_search_chunks` 只保存稳定的文本、时间范围和关键词检索字段。每个模型的向量保存到子表：

```text
SearchChunk A
├── Qwen3-Embedding-4B   / 2560 维
└── Qwen3-Embedding-0.6B / 1024 维
```

这样关键词召回始终只扫描一次 Chunk，不会因为存在多套向量而重复。

向量子表保留 `on delete cascade`。级联删除本身不是问题：当 Chunk 文本、切分结果或录音画像正文发生变化时，旧模型向量已经失效，应随父记录一起删除。为了让文本未变化时的多模型向量可以稳定并存，在线投影逻辑必须同步调整：

- 不再因为生成某个 profile 的向量就删除并重建父 Chunk 或父画像文档；
- 父记录文本和结构未变化时，只 upsert 当前 profile 对应的向量子表记录，保留其他 profile 的向量；
- 父记录文本、Chunk 边界或检索元数据发生变化时，更新或重建父记录，并通过级联删除使其全部旧向量失效；
- 父记录发生变化后，只能在目标 profile 的新向量生成成功后将其计入该 profile 的覆盖率。

### 3.3 模型与向量必须成对选择

Query Embedding 模型和数据库候选向量必须来自同一个 profile。禁止只切换查询字段或只切换 Query 模型。

运行时至少校验：

```text
provider
model_name
dimensions
distance_metric
```

任一字段不一致时直接失败，不允许继续执行跨模型向量比较。

## 4. Embedding Profile

新增稳定的 profile key，业务代码不直接判断 Hugging Face 模型字符串：

```python
EMBEDDING_PROFILES = {
    "qwen3-4b": {
        "provider": "sentence_transformers",
        "model": "Qwen/Qwen3-Embedding-4B",
        "dimensions": 2560,
        "distance_metric": "cosine",
    },
    "qwen3-0.6b": {
        "provider": "sentence_transformers",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": 1024,
        "distance_metric": "cosine",
    },
}
```

新增配置：

```env
# 实际用于新索引和在线 Query Embedding 的模型
EMBEDDING_PROFILE=qwen3-4b

# 安装阶段需要提前下载的模型；不会同时启用
# 安装脚本默认额外预下载 qwen3-0.6b；可用逗号分隔覆盖
EMBEDDING_PRELOAD_PROFILES=qwen3-0.6b

EMBEDDING_MODEL_CACHE_DIR=model-cache/embedding
EMBEDDING_INFERENCE_BATCH_SIZE=8
```

`EMBEDDING_PROFILE` 是唯一的运行时模型选择配置。模型仓库名和向量维度必须由 Profile 注册表派生，不提供 `EMBEDDING_MODEL` 或 `EMBEDDING_DIMENSIONS` 独立覆盖入口。

Pipeline Version 必须保存完整 profile：

```json
{
  "embedding": {
    "profile": "qwen3-0.6b",
    "provider": "sentence_transformers",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "dimensions": 1024,
    "distance_metric": "cosine"
  }
}
```

## 5. 数据库设计

### 5.1 Chunk 向量表

```sql
create table recording_search_chunk_embeddings (
    chunk_id uuid not null references recording_search_chunks(id) on delete cascade,
    embedding_model_id uuid not null references embedding_models(id) on delete restrict,
    model_key text not null,
    dimensions integer not null,
    content_hash text not null,
    embedding halfvec not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (chunk_id, embedding_model_id),
    check (vector_dims(embedding) = dimensions)
);
```

`content_hash` 对实际送入 embedding 模型的 `retrieval_text` 计算，用来判断文本是否变化以及支持幂等跳过。

### 5.2 录音画像向量表

```sql
create table recording_retrieval_document_embeddings (
    document_id uuid not null references recording_retrieval_documents(id) on delete cascade,
    embedding_model_id uuid not null references embedding_models(id) on delete restrict,
    model_key text not null,
    dimensions integer not null,
    content_hash text not null,
    embedding halfvec not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (document_id, embedding_model_id),
    check (vector_dims(embedding) = dimensions)
);
```

`recording_retrieval_documents` 保留画像的 `retrieval_text` 和 document type，不再因为模型不同复制文本。

### 5.3 主表与在线投影调整

向量迁入子表后，主表不再承担模型向量存储职责：

- `recording_search_chunks` 移除 `embedding_model_id`、`embedding` 以及旧向量索引，唯一键调整为 `(recording_id, chunk_index)`；
- `recording_retrieval_documents` 移除 `embedding_model_id`、`embedding` 以及旧向量索引，唯一键调整为 `(recording_id, document_index)`；
- SearchChunk 投影先判断现有 Chunk 的文本、时间边界和检索元数据是否变化；未变化时复用父记录，只写当前 profile 的向量；发生变化时替换父记录并使全部旧向量级联失效；
- Summary 投影采用相同规则：`retrieval_text` 未变化时复用画像文档，发生变化时更新父记录并清理全部旧模型向量；
- 禁止为了切换 embedding profile 而删除父记录。profile 切换只改变新向量写入的子表行。

向量子表继续使用 `on delete cascade`，避免父记录确实被删除或重建后遗留孤儿向量。

### 5.4 HNSW 索引

pgvector 允许无固定维度的 `halfvec` 列保存不同维度向量，但近似索引中的向量必须维度相同，因此为每个 profile 创建 partial expression index：

```sql
create index recording_chunk_embeddings_qwen3_4b_hnsw_idx
on recording_search_chunk_embeddings using hnsw (
    (embedding::halfvec(2560)) halfvec_cosine_ops
)
where model_key = 'qwen3-4b';

create index recording_chunk_embeddings_qwen3_0_6b_hnsw_idx
on recording_search_chunk_embeddings using hnsw (
    (embedding::halfvec(1024)) halfvec_cosine_ops
)
where model_key = 'qwen3-0.6b';
```

录音画像向量表建立对应的两个索引。查询必须包含与 partial index 完全一致的 `model_key` 条件和维度 cast。

参考：[pgvector 不同维度向量与 partial index 官方说明](https://github.com/pgvector/pgvector#can-i-store-vectors-with-different-dimensions-in-the-same-column)

### 5.5 迁移现有 4B 向量

现有 4B 向量可以直接复制到子表，不需要重新推理：

```text
recording_search_chunks.embedding
→ recording_search_chunk_embeddings / qwen3-4b

recording_retrieval_documents.embedding
→ recording_retrieval_document_embeddings / qwen3-4b
```

迁移必须在 Retriever 切换到子表之前完成。切换后，向量召回只读取子表；即使旧向量列仍然存在，也不参与运行时查询。

本次不实现以下兼容回退：

```text
目标子表没有数据
→ 自动切回 recording_search_chunks.embedding
```

如果目标 profile 的子表向量缺失，系统应在运行前覆盖率检查或查询时明确失败。不能静默读取旧字段，否则同一次运行可能混用新旧存储路径，评测结果也无法确认实际使用了哪套向量。

## 6. 安装脚本与模型下载

### 6.1 当前能力

`scripts/install_audio_dependencies.sh` 已经：

- 安装 `sentence-transformers` 和 `huggingface_hub`；
- 调用 `scripts/ensure_hf_snapshot.py`；
- 检查本地缓存，已存在时跳过；
- 下载中定期输出 heartbeat；
- 中断后复用 Hugging Face 缓存继续下载；
- 使用 `EMBEDDING_MODEL_CACHE_DIR` 保存模型。

因此不需要新增另一套下载器，只需要让 installer 遍历需要预加载的 embedding profiles。

### 6.2 建议逻辑

将当前单模型逻辑：

```text
ensure_huggingface_snapshot EMBEDDING_MODEL
```

改为：

```text
解析 EMBEDDING_PRELOAD_PROFILES
加入当前 EMBEDDING_MODEL，去重
逐个从 profile registry 解析 model repo
调用 ensure_huggingface_snapshot
```

当前 installer 在完整的运行时 profile 配置落地前继续使用 `EMBEDDING_MODEL` 作为 active model，下载逻辑等价于：

```bash
models=("${EMBEDDING_MODEL}")

for profile in $(parse_profiles "${EMBEDDING_PRELOAD_PROFILES}"); do
  model_repo="$(embedding_model_for_profile "${profile}")"
  append_unique models "${model_repo}"
done

for model_repo in "${models[@]}"; do
  ensure_huggingface_snapshot "${model_repo}" "${ROOT_DIR}/${EMBEDDING_MODEL_CACHE_DIR}" "embedding"
done
```

### 6.3 下载策略

- 默认安装保证当前 `EMBEDDING_MODEL` 和 `qwen3-0.6b` 均已下载；如果当前模型就是 0.6B，会自动去重；
- 可通过逗号分隔的 `EMBEDDING_PRELOAD_PROFILES` 覆盖额外预下载列表；当前 `EMBEDDING_MODEL` 始终会加入下载列表；
- 下载失败必须让安装失败，不能等到 Worker 处理请求时再联网；
- Worker 运行时只允许从本地 snapshot 加载；
- installer 重复运行必须幂等；
- 日志需要明确显示 profile、repo 和最终 snapshot 路径；
- 未知 profile 直接报错，不能回退到默认模型；
- 保留现有 `HF_HUB_DOWNLOAD_WORKERS`、heartbeat 和超时配置；
- 不自动删除旧模型缓存，模型对比期间必须保留 4B snapshot。

## 7. Compute Worker 模型一致性

### 7.1 第一版运行方式

第一版采用进程级 active profile：一个 Compute Worker 实例只加载一个 embedding 模型。切换 profile 后需要重启 Compute Worker。

这样可以避免 4B 和 0.6B 同时常驻显存，也避免频繁加载和卸载模型。

### 7.2 请求与响应校验

Embedding 请求增加期望 profile：

```json
{
  "texts": ["..."],
  "embedding_profile": "qwen3-0.6b"
}
```

Worker 收到请求时校验它与当前加载 profile 一致。不一致时明确失败：

```text
requested=qwen3-0.6b, worker_active=qwen3-4b
```

响应继续返回：

```json
{
  "provider": "sentence_transformers",
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "dimensions": 1024,
  "vectors": []
}
```

Retriever 在执行 SQL 前再次校验响应模型和 Pipeline Version，防止错误模型向量进入数据库查询。

## 8. 历史数据回填

### 8.1 脚本

新增：

```text
backend/scripts/backfill_search_embeddings.py
```

Chunk 与录音画像分别使用：

```bash
cd backend
PYTHONPATH=packages .venv/bin/python scripts/backfill_search_embeddings.py \
  --model-profile qwen3-0.6b --resume

EMBEDDING_PROFILE=qwen3-0.6b PYTHONPATH=packages .venv/bin/python \
  scripts/backfill_recording_summary_embeddings.py
```

执行回填前，Compute Worker 必须以 `EMBEDDING_PROFILE=qwen3-0.6b` 启动。Worker 与请求 profile 不一致时任务会明确失败，不会使用错误模型继续写入。

建议参数：

```text
--model-profile qwen3-0.6b
--batch-size 16
--workspace-id UUID
--recording-id UUID
--limit N
--resume
--force
--dry-run
```

### 8.2 Chunk 回填流程

```text
读取稳定 SearchChunk
→ 使用 text + metadata.topic/terms/search_context 重建 retrieval_text
→ 计算 content_hash
→ 跳过同模型且 hash 一致的向量
→ 分批调用 0.6B Worker
→ 校验 profile/model/dimensions
→ upsert 到 chunk embedding 子表
→ 提交进度
```

不需要重新执行 ASR、文本润色、Chunk 切分或 Summary LLM。

不能直接把 `original_text` 用作向量输入。当前向量召回使用润色后的 `retrieval_text()`；关键词召回继续使用原文归一化字段。

### 8.3 录音画像回填

直接读取 `recording_retrieval_documents.retrieval_text`，使用相同 profile 编码并写入画像 embedding 子表。

如果本次实验不希望录音画像影响结果，也可以在 4B 和 0.6B 两组评测中同时关闭 Recording Profile Retrieval，但两组配置必须一致。

### 8.4 幂等与恢复

- 主键使用 `entity_id + embedding_model_id`；
- 默认遇到相同 content hash 时跳过；
- 每个 batch 独立事务；
- 记录 scanned、skipped、succeeded、failed；
- 失败项记录 entity ID 和错误类型；
- 中断后可从未成功记录继续；
- `--force` 只重算指定 profile，不覆盖其他模型；
- `--dry-run` 只统计缺失数量和预计输入规模。

### 8.5 切换前覆盖率门禁

```text
chunk_coverage
= 具有目标模型向量的有效 Chunk 数 / 有效 Chunk 总数

profile_coverage
= 具有目标模型向量的画像文档数 / 有效画像文档总数
```

默认要求：

- Chunk coverage = 100%；
- Recording profile coverage = 100%，或者两组评测均关闭画像召回；
- 维度异常 = 0；
- content hash 过期 = 0；
- failed = 0。

门禁不通过时禁止将该 profile 设置为生产 active。

## 9. 检索改造

Retriever 根据 Pipeline Version 中的 profile：

1. 生成同 profile 的 Query Embedding；
2. 解析 `embedding_model_id` 和 dimensions；
3. Join 对应的 embedding 子表；
4. 使用对应维度 cast 和 HNSW partial index；
5. 使用余弦距离 `<=>` 排序；
6. 返回 `1 - cosine_distance` 作为 score。

0.6B 查询示意：

```sql
select chunks.id,
       1 - (vectors.embedding::halfvec(1024) <=> :query::halfvec(1024)) as score
from recording_search_chunks chunks
join recording_search_chunk_embeddings vectors on vectors.chunk_id = chunks.id
where vectors.model_key = 'qwen3-0.6b'
order by vectors.embedding::halfvec(1024) <=> :query::halfvec(1024)
limit :limit;
```

关键词召回不 Join 向量表，保持现有行为。

## 10. Corpus Snapshot 与公平评测

Corpus Snapshot 应表达文本语料，而不是某个模型生成的向量：

```text
Corpus Snapshot：Chunk 文本、时间范围、内容 checksum
Pipeline Version：Embedding profile、维度、检索参数
```

建议：

- Corpus checksum 不包含 embedding row ID 和 embedding model ID；
- 相同 Chunk 文本使用 4B 和 0.6B 时复用同一个 Corpus Snapshot；
- Evaluation Run 通过 Pipeline Version 选择 embedding profile；
- Run 开始前验证目标 snapshot 的每个 Chunk 都有目标 profile 向量；
- 4B 与 0.6B 除 embedding profile 外，其余 RRF、Lexical、Expand、Rerank 参数完全相同。

本次暂不改造 Retriever，使其直接从冻结的 Corpus Snapshot 检索。评测运行仍从实时 `recording_search_chunks` 检索，Corpus Snapshot 用于记录语料版本和结果映射。因此正式对比期间必须满足以下操作约束：

- 从 Baseline Run 开始到 Candidate Run 完成，不导入会进入评测范围的新录音；
- 不重新处理、删除或修改评测 Workspace 中的录音、SearchChunk 和录音画像；
- Baseline Run 完成后再切换 Worker profile 并启动 Candidate Run，不能让两组 Run 跨越语料变更；
- 如果无法保证上述约束，本轮结果不能视为严格的同语料对比，需要重新生成两组 Run。

这是当前评测方案的已知限制。本次迁移不实现 Snapshot 约束检索或 Snapshot 独立向量表。

评测指标：

| 分类 | 指标 |
|---|---|
| 召回质量 | Hit@5、Recall@10、MRR、nDCG@10 |
| 节点诊断 | Gold 首次丢失节点、Vector 节点 Gold 覆盖率 |
| Query 性能 | embedding latency P50/P95、吞吐量 |
| 数据库性能 | vector search P50/P95、HNSW index size |
| 资源 | 峰值显存、常驻内存、模型加载时间 |

正式对比步骤：

1. 固定 Dataset Version 和文本 Corpus Snapshot；
2. 使用 4B Pipeline 执行 Baseline Run；
3. 重启 Compute Worker，active profile 改为 0.6B；
4. 使用 0.6B Pipeline 执行 Candidate Run；
5. 对比聚合指标和逐 Gold 节点 Journey；
6. 记录准确率变化、延迟收益和失败 Case 分布。
