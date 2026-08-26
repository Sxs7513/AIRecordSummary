# Phase 2 实施计划：可溯源 RAG、智能总结与录音片段定位

## 1. 文档目的

Phase 1 已经完成录音上传、ASR 转写、说话人分离、目标人物识别、文本校正和 `utterance_segments` 展示层沉淀。

Phase 2 的目标是在 Phase 1 产物之上接入向量数据库和 RAG 能力，让用户可以输入自然语言问题，由系统先检索相关录音片段，再把这些片段作为证据交给大模型生成智能总结或回答。

最终输出不能只是模型的一段总结文本，还必须能追踪到具体是哪一条录音、哪个片段、哪个大致时间点支撑了这段总结。

本阶段优先做“检索得到、总结可信、证据可回放”。考虑到用户 query 不可控，Phase 2 不能只做单一 `query embedding -> vector topK -> answer` 链路，还需要引入 LLM router、证据充分性判断、必要时二次检索以及回答引用校验。

Phase 2 在线 RAG 编排使用 LangGraph 作为核心工作流引擎。LLM router、retrieval executor、evidence grader、answer generator、answer validator、query rewrite 和 answer rewrite 都作为 LangGraph node 实现，并通过 graph state 串联。

## 2. Phase 2 范围

### 2.1 必做能力

- 接入 PostgreSQL + pgvector 作为向量索引存储
- 基于 `utterance_segments` 生成检索 chunk
- 为 chunk 生成 embedding 并落库
- 支持已完成录音自动进入索引任务
- 支持历史录音批量补索引
- 提供 RAG 查询 API：`用户问题 -> LLM router -> 检索计划 -> 检索证据 -> 证据评估 -> 模型总结 -> 引用校验 -> 带引用返回`
- 支持 LLM router 识别用户问题类型，并选择不同 retrieval strategy
- 支持非向量类问题，例如“最近两个音频都说了什么”“总结昨天的录音”
- 支持证据不足时最多一次 query rewrite 和二次检索
- 支持回答引用不合法时最多一次 answer rewrite
- 使用 LangGraph 编排可重试 RAG 工作流、节点状态和条件分支
- 默认使用本地开源大模型做总结实验
- 预留 DeepSeek API provider，方便后续效果不够时切换
- 返回模型总结、引用证据和多个相关录音片段
- 每条引用证据必须包含：
  - 录音 ID、标题、文件名
  - chunk 文本
  - 起止时间
  - 相关度分数
  - speaker label
  - 目标人物命中信息
  - 可跳转到详情页并定位播放的 URL
- 提供聊天式 RAG 总结页面
- 聊天式页面支持真正流式回答：先返回 evidence，再持续返回 answer delta
- 模型 thinking 内容默认不直接展示，前端只显示“思考中”，结束后折叠展示
- 详情页支持从 URL query 中读取时间点并定位音频播放器

### 2.2 可选增强

- 混合检索：向量检索 + 关键词检索 + 元数据过滤
- 对检索结果做轻量重排
- 查询日志与点击日志
- 支持按说话人、目标人物、日期范围、录音状态过滤
- 支持“只检索不总结”的调试模式，调试入口可以先隐藏但 API 保留

### 2.3 暂不做

- 多轮长期记忆
- 复杂权限系统
- 独立 Elasticsearch / Milvus / Qdrant 集群
- 对音频波形做精确语义定位
- 让模型在没有证据时自由发挥

## 3. 设计原则

### 3.1 以最终展示文本为默认检索来源

Phase 1 中 `utterance_segments` 已经保存了合并、规则替换、pycorrector、本地 LLM 校正和语义合并后的文本。Phase 2 默认从 `utterance_segments` 建索引，而不是直接使用原始 `transcription_segments`。

这样用户搜索时命中的文本更接近页面展示内容，也更适合 embedding。

### 3.2 保留来源可追溯

每个检索 chunk 都必须保存来源：

- `recording_id`
- 来源 `utterance_segment_ids`
- 来源 `transcription_segment_ids`
- `start_ms`
- `end_ms`

检索结果不能只返回一段文本，必须能回链到录音和时间点。

### 3.3 先做离线索引，再做在线检索

embedding 生成属于重任务，应接入现有 `processing_jobs` 链路，不放在用户搜索请求中同步处理。

在线搜索请求只做：

1. query 标准化
2. query embedding
3. 数据库检索
4. 结果聚合
5. 返回证据

### 3.4 MVP 优先使用 PostgreSQL + pgvector

Phase 2 建议继续使用已有 PostgreSQL，安装 `pgvector` extension。这样业务数据、chunk 数据和向量索引在同一数据库里，开发和本地部署成本最低。

后续如果数据量明显增长，再把向量索引抽象层迁移到独立向量数据库。

### 3.5 模型只基于检索证据输出

RAG 链路中，大模型不是直接读取所有录音，也不是凭空总结。它只能看到本次 query 召回出来的 chunk 证据包。

模型输出必须遵守：

- 每个关键结论后带引用编号
- 引用编号必须对应实际召回 chunk
- 不允许引用不存在的录音或时间点
- 证据不足时明确说明“没有在录音中找到足够依据”
- 总结和回答中的事实必须能追踪到 `recording_search_chunks`

### 3.6 Provider 可替换

Phase 2 需要把“embedding 模型”和“总结模型”拆成两个 provider：

- embedding provider：把文本和用户问题转换成向量
- answer provider：基于检索证据生成总结或回答

MVP 默认：

- embedding：本地 `Qwen/Qwen3-Embedding-4B`
- answer：本地 `Qwen3.5-9B Q8_0 GGUF`

后续如果本地开源模型效果不够，再把 answer provider 切到 DeepSeek API。业务层不直接耦合具体模型。

### 3.7 Query 不等于向量检索文本

用户输入的自然语言问题不一定适合直接做向量检索。

例如：

- “最近两个音频都说了什么”
- “总结昨天的录音”
- “对比一下今天和昨天讨论重点”
- “上周跟张三相关的会议有什么风险”

这些问题首先需要确定录音范围、时间范围、人物范围或对比对象，再决定是否需要片段级向量检索。因此在线查询链路必须增加 LLM router，把原始 query 转换为结构化 retrieval plan。

人物检索先作为 retrieval plan 的一等约束预留。Router 可以识别用户 query 中的人物名，但第一版不直接把人名当成数据库过滤条件；后续会在 `routeQuery` 和 `retrieveEvidence` 之间增加 `resolvePeople` 节点，把 `personNames` 解析为已确认的 `speakerProfileIds`，再交给检索执行器。

### 3.8 先确定检索计划，再生成回答

RAG 链路必须拆成可观测、可校验的步骤：

```text
query
-> routeQuery
-> executeRetrievalPlan
-> gradeEvidence
-> maybeRewriteQueryAndRetrieveAgain
-> generateAnswer
-> validateAnswer
-> maybeRewriteAnswer
-> return answer + evidence
```

每个步骤都要输出结构化日志。模型可以建议检索计划、改写 query 或改写回答，但不能直接编造录音 ID、片段 ID 或引用来源。所有录音范围、片段范围和引用都必须由后端根据数据库结果生成。

### 3.9 LangGraph 工作流编排

Phase 2 将 LangGraph 作为 RAG 查询链路的编排层。业务逻辑仍以可测试的 node 函数组织，但运行时由 LangGraph 负责节点串联、条件分支、重试上限和 graph state 传递。

LangGraph 负责：

- 根据 router 结果进入不同 retrieval strategy
- 在 evidence 不足时进入 query rewrite 分支
- 在 answer validation 失败时进入 answer rewrite 分支
- 控制 `retrievalAttempt` 和 `answerRewriteCount`，避免无限循环
- 暴露每个 node 的结构化日志和中间状态，便于排查问题
- 后续需要多轮对话或状态恢复时接入 checkpoint

Graph node 边界：

```ts
routeQuery(state) -> state
resolvePeople(state) -> state
retrieveEvidence(state) -> state
gradeEvidence(state) -> state
rewriteQuery(state) -> state
generateAnswer(state) -> state
validateAnswer(state) -> state
rewriteAnswer(state) -> state
```

## 4. 技术选型

### 4.1 向量数据库

首选：

- PostgreSQL + pgvector

需要 extension：

```sql
create extension if not exists vector;
create extension if not exists pg_trgm;
```

`vector` 用于语义检索，`pg_trgm` 用于标题和文本的轻量模糊匹配。

### 4.2 Embedding 模型

MVP 推荐本地 embedding provider：

- 默认模型：`Qwen/Qwen3-Embedding-4B`
- 默认维度：`2560`
- Python 依赖：`sentence-transformers`

推荐原因：

- 2025 年发布，比 `BAAI/bge-m3` 更新
- 中文、英文和混合语义效果较稳
- 适合录音转写文本这种口语化内容
- 可本地运行，与 Phase 1 的本地模型策略一致
- 4B 版本召回质量更强，适合当前以本地高质量 RAG 为目标的配置

轻量备选：

- `Qwen/Qwen3-Embedding-0.6B`
- 维度：`1024`
- 适合资源成本更敏感的本地轻量场景
- `BAAI/bge-small-zh-v1.5`
- 维度：`512`
- 适合 CPU 资源紧张场景

注意：pgvector 的向量维度和模型输出维度强相关。Phase 2 默认使用 `halfvec(2560)`，便于 2560 维 embedding 继续使用 HNSW 索引。如果切换到不同维度模型，需要执行迁移或重建 chunk embedding 表。

### 4.3 关键词检索

中文全文检索如果只依赖 PostgreSQL 内置分词效果有限，因此 MVP 不强依赖 `tsvector` 中文分词。

建议策略：

- 标题、文件名：`pg_trgm` 相似度或 `ilike`
- chunk 文本：`ilike` 作为兜底
- 主要召回仍以向量检索为主

后续可以评估：

- PostgreSQL 中文分词 extension
- Meilisearch
- Elasticsearch / OpenSearch

### 4.4 总结模型

Phase 2 的 RAG 总结模型优先使用开源模型做本地实验。

推荐本地模型方向：

- 默认：`Qwen3.5-9B Q8_0 GGUF`
- 资源更省备选：`Qwen3-8B`
- 资源紧张备选：`Qwen3-4B`
- 量化 GGUF 版本优先，便于复用 Phase 1 本地 llama.cpp 运行方式

默认策略：

- 先用 `llama-cpp-python` 跑本地开源模型
- MVP 使用 `Qwen3.5-9B` 的 Q8_0 GGUF 量化版本
- prompt 中强制要求只基于证据回答
- 输出结构化 JSON，便于校验引用
- 如果本地效果不好，再新增 `deepseek_api` provider

DeepSeek API 作为后续备选：

- provider 名称：`deepseek_api`
- 使用环境变量配置 API key、base URL 和模型名
- 与本地模型共享同一套 answer provider 接口

新增配置建议：

```env
RAG_ANSWER_ENABLED=true
RAG_ONLINE_DEFAULT_MODEL=gemini-gemini-3.5-flash-lite
LOCAL_LLM_MODEL_REPO=DevQuasar/Qwen.Qwen3.5-9B-GGUF
LOCAL_LLM_MODEL_FILE=Qwen.Qwen3.5-9B.Q8_0.gguf
RAG_ANSWER_CONTEXT_SIZE=8192
RAG_ANSWER_TIMEOUT_MS=600000

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

注意：本地模型和 DeepSeek API 只负责“基于证据生成总结”，不负责检索。检索仍由 embedding 模型 + pgvector 完成。

## 5. 数据模型设计

### 5.1 新增任务类型

`processing_jobs.job_type` 增加：

- `embedding_indexing`

完整任务链路变为：

```text
transcription
-> speaker_diarization
-> speaker_identification
-> text_correction
-> embedding_indexing
-> recording completed
```

Phase 1 当前在 `text_correction` 完成后直接把录音置为 `completed`。Phase 2 需要调整为：

- `text_correction` 完成后创建 `embedding_indexing`
- `embedding_indexing` 完成后再把录音置为 `completed`

如果 embedding 功能关闭，可以保留 Phase 1 行为。

### 5.2 新增表：embedding_models

用于记录当前使用过的 embedding 模型，便于后续模型升级和重建索引。

```sql
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
```

### 5.3 新增表：recording_search_chunks

用于保存可检索文本、来源时间范围和 embedding。

MVP 默认 `embedding halfvec(2560)`：

```sql
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
```

推荐索引：

```sql
create index if not exists recording_search_chunks_recording_id_idx
    on recording_search_chunks (recording_id);

create index if not exists recording_search_chunks_time_idx
    on recording_search_chunks (recording_id, start_ms, end_ms);

create index if not exists recording_search_chunks_target_person_idx
    on recording_search_chunks (is_target_person);

create index if not exists recording_search_chunks_text_trgm_idx
    on recording_search_chunks using gin (normalized_text gin_trgm_ops);

create index if not exists recording_search_chunks_embedding_hnsw_idx
    on recording_search_chunks using hnsw (embedding halfvec_cosine_ops);
```

### 5.4 新增表：search_queries

用于记录用户查询和系统召回结果，便于排查和后续评估。

```sql
create table if not exists search_queries (
    id uuid primary key default gen_random_uuid(),
    query_text text not null,
    normalized_query text not null,
    filters jsonb not null default '{}'::jsonb,
    result_count integer not null default 0,
    latency_ms integer check (latency_ms is null or latency_ms >= 0),
    created_at timestamptz not null default now()
);
```

### 5.5 可选表：search_result_clicks

用于记录用户是否点击某条结果或播放某个时间点。

```sql
create table if not exists search_result_clicks (
    id uuid primary key default gen_random_uuid(),
    search_query_id uuid references search_queries(id) on delete set null,
    recording_id uuid not null references recordings(id) on delete cascade,
    search_chunk_id uuid references recording_search_chunks(id) on delete set null,
    target_ms integer check (target_ms is null or target_ms >= 0),
    created_at timestamptz not null default now()
);
```

## 6. Chunk 生成策略

### 6.1 输入来源

默认输入：

- `utterance_segments`
- 按 `recording_id, utterance_index` 顺序读取

每个 utterance 自带：

- `start_ms`
- `end_ms`
- `text`
- `speaker_label`
- `speaker_cluster_id`
- `source_transcription_segment_ids`
- `is_target_person`
- `matched_speaker_profile_id`

### 6.2 Chunk 粒度

用户希望“大概定位到具体时间点”，因此 chunk 不能太长。

MVP 推荐：

- 目标时长：`15s - 45s`
- 最大时长：`60s`
- 目标文本长度：`120 - 500` 中文字符
- 最大文本长度：`800` 中文字符
- 跨 speaker 默认不合并
- 同 speaker 相邻 utterance 可以合并
- 相邻 utterance 间隔超过 `3000ms` 不合并

### 6.3 短句处理

单条 utterance 很短时，直接 embedding 可能缺少上下文。建议：

- chunk 文本包含当前 utterance
- 对极短 utterance 可向前后各带 1 条同 speaker 或同主题邻近 utterance
- 但 `start_ms` 和 `end_ms` 仍覆盖 chunk 实际文本范围

### 6.4 来源记录

一个 chunk 可能由多个 utterance 组成，必须保存所有来源 ID：

- `source_utterance_segment_ids`
- `source_transcription_segment_ids`

这样用户点击结果时可以：

- 跳到 chunk 的 `start_ms`
- 在详情页高亮来源 utterance
- 后续必要时展开上下文

### 6.5 文本标准化

`normalized_text` 用于关键词兜底检索。

建议处理：

- trim
- 合并多余空白
- 英文统一小写
- 中文标点不强行删除
- 保留专业词大小写在 `text` 中

## 7. Embedding 生成设计

### 7.1 Provider 边界

新增模块建议：

```text
lib/audio-transcoding-analysis/embedding/
  index.ts
  provider.ts
  local-qwen.ts
  scripts/run_qwen_embedding.py
```

TypeScript 接口：

```ts
export interface EmbeddingProvider {
  embedTexts(texts: string[]): Promise<number[][]>;
  embedQuery(query: string): Promise<number[]>;
}
```

本地 provider 通过 Python 脚本调用 `sentence-transformers`。

### 7.2 批量策略

索引任务中按批量生成 embedding：

- 默认 batch size：`16`
- CPU 环境可降到 `4`
- 失败时整条录音索引任务失败，允许重试

### 7.3 模型缓存

沿用 Phase 1 的项目内缓存策略：

- 默认缓存目录：`model-cache/embedding`
- 通过环境变量配置

新增配置：

```env
EMBEDDING_ENABLED=true
EMBEDDING_PROVIDER=local_qwen3
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
EMBEDDING_DIMENSIONS=2560
EMBEDDING_PYTHON_BIN=.venv-audio/bin/python
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=16
EMBEDDING_MODEL_CACHE_DIR=model-cache/embedding
SEARCH_VECTOR_TOP_K=30
SEARCH_FINAL_TOP_K=10
SEARCH_MIN_SCORE=0.25
SEARCH_CHUNK_MAX_DURATION_MS=60000
SEARCH_CHUNK_MAX_TEXT_CHARS=800
```

### 7.4 Query embedding

搜索接口收到 query 后实时生成 query embedding。

为降低延迟，可以增加简单内存缓存：

- key：`normalized_query + model_name`
- value：embedding
- TTL：`10min`
- 最大条数：`100`

## 8. 索引任务设计

### 8.1 任务入口

新增 job：

- `embedding_indexing`

处理步骤：

1. 读取录音和 `utterance_segments`
2. 删除当前录音当前 active embedding model 的旧 chunk
3. 按策略生成 chunk
4. 批量生成 embedding
5. 写入 `recording_search_chunks`
6. 完成 job
7. 将录音状态置为 `completed`

### 8.2 与 text_correction 的关系

Phase 2 中 `text_correction` 完成后不直接完成录音，而是：

```ts
await completeJob(job.id, { nextJobType: "embedding_indexing" });
```

`embedding_indexing` 完成后：

```ts
await completeJob(job.id, { recordingStatus: "completed" });
```

如果 `EMBEDDING_ENABLED=false`：

```ts
await completeJob(job.id, { recordingStatus: "completed" });
```

### 8.3 重试与重建

需要支持三类重建：

- 单条录音索引重建：录音详情页按钮或 API
- 所有 completed 录音补索引：脚本
- embedding 模型升级后的全量重建：脚本

建议脚本：

```text
scripts/search/reindex-recordings.ts
```

支持参数：

```bash
npm run search:reindex
npm run search:reindex -- --recording-id <uuid>
npm run search:reindex -- --force
```

`--force` 表示删除旧 chunk 并重建。

## 9. RAG API 设计

### 9.0 LLM Router 与检索计划

单纯向量检索只适合具体主题类问题，例如“预算超支原因是什么”。对于“最近两个音频都说了什么”这类录音范围总结问题，系统必须先选择录音范围，再读取录音上下文，而不是直接从 chunk 向量库取 topK。

#### 9.0.1 Router 职责

Router 的职责是把用户 query 转成结构化 retrieval plan。Router 不负责回答问题，也不能编造录音 ID 或 chunk ID。

第一版检索计划只保留两类执行模式，其他条件都作为联合检索 scope：

```ts
type RetrievalStrategy =
  | "scope_summary"
  | "chunk_search";
```

含义：

- `scope_summary`：筛选录音集合后读取这些录音的上下文做整体总结，例如“这两天的录音分别说了什么”
- `chunk_search`：筛选录音集合后，在范围内按主题做片段级语义检索，例如“昨天关于预算讲了什么”

`recordingLimit`、`recordingRank`、`timeRange`、人物、地点、指定录音等都属于 scope filter，可以联合出现，不再拆成独立 strategy。

人物相关 query 第一版根据是否有具体主题进入 `chunk_search` 或 `scope_summary`。Router 只抽取人名到 `filters.personNames`，不能编造 `speakerProfileIds`。后续人物检索上线时，由独立 resolver 把人名、别名或用户选择映射到 `speaker_profiles.id`。

Router 输出 JSON：

```json
{
  "intent": "scope_summary",
  "strategy": "scope_summary",
  "topic": null,
  "recordingLimit": null,
  "recordingRank": null,
  "timeRange": {
    "text": "这两天",
    "type": "relative",
    "amount": 2,
    "unit": "day",
    "direction": "past",
    "from": null,
    "to": null
  },
  "dateRange": null,
  "filters": {
    "recordingIds": [],
    "speakerProfileIds": [],
    "personNames": [],
    "locations": [],
    "targetPersonOnly": false
  },
  "needsAnswer": true,
  "reason": "用户询问最近两天录音的整体内容，需要按创建时间筛选时间范围内的 completed 录音"
}
```

程序侧必须校验：

- JSON 必须能解析
- `strategy` 必须属于允许枚举
- `recordingLimit` 必须 clamp，例如 `1 <= limit <= 5`
- `timeRange` 只表达时间意图，程序侧归一化为 `dateRange.from/to`
- 相对时间 query 不能信任模型直接给出的 `dateRange`，必须优先由程序根据 `timeRange` 或原始 query 归一化
- `dateRange` 使用 Asia/Shanghai 的本地边界字符串，例如 `2026-05-07T00:00:00.000+08:00`
- SQL 时间范围使用半开区间：`created_at >= from and created_at < to`
- `dateRange` 如果由模型直接给出，必须是合法日期，且不能无限大
- `speakerProfileIds` 必须来自数据库已有目标人物
- `personNames` 只是 router 抽取的未解析人名，不能直接进入 SQL 过滤
- Router 失败时 fallback 到 `chunk_search`

#### 9.0.1.1 人物检索预留设计

人物检索会分为两层：

- 人名解析层：把 query 中的 `personNames` 解析为候选 `speaker_profiles.id`
- 检索过滤层：用 `speakerProfileIds` 或 `targetPersonOnly` 限定 `recording_search_chunks.matched_speaker_profile_ids`

后续新增节点：

```text
routeQuery
-> resolvePeople
-> executeRetrievalPlan
```

`resolvePeople` 的输出写回 `route.filters.speakerProfileIds`，并保留 `personNames` 便于日志和前端澄清。当人名无法唯一匹配时，第一版可以不做 SQL 人物过滤，只把人名并入 `topic` 做向量召回；之后可以扩展为让前端要求用户选择具体人物。

#### 9.0.2 Retrieval Plan Executor

两类执行模式共享同一套 scope filter：

```text
buildScopeFilters
-> recordingLimit / recordingRank / timeRange / recordingIds / personNames / locations

scope_summary
-> select completed recordings by scope
-> read utterance_segments by recording
-> evidence recording contexts

chunk_search
-> resolve recording/person/location/time scope
-> query embedding
-> pgvector search within scope
-> evidence chunks
```

录音级总结上下文不能无限塞给模型。第一版建议：

- 单条录音最多取 `SEARCH_RECORDING_CONTEXT_MAX_CHARS=12000`
- 多条录音总上下文最多 `RAG_ANSWER_CONTEXT_SIZE` 的安全预算
- 优先取 `utterance_segments`，必要时按时间顺序压缩
- evidence 中保留录音 ID、标题、文件名、时间范围和可跳转 URL

#### 9.0.3 gradeEvidence

`gradeEvidence` 判断当前 evidence 是否足够回答用户问题。它不生成最终回答。

输入：

```json
{
  "query": "最近两个音频都说了什么",
  "route": {
    "strategy": "scope_summary",
    "recordingLimit": 2
  },
  "evidence": [
    {
      "index": 1,
      "recordingId": "uuid",
      "chunkId": "uuid",
      "title": "会议录音",
      "score": 0.82,
      "text": "片段文本"
    }
  ]
}
```

输出：

```json
{
  "sufficient": false,
  "reason": "只召回到一条录音，但用户要求最近两个音频",
  "missingAspects": ["第二条录音内容"],
  "rewriteQuery": "最近两个录音 主要内容 总结",
  "confidence": 0.42
}
```

第一版规则优先：

- evidence 为空：不足
- top score 低于阈值：不足
- 用户要求 N 条录音，但 evidence 覆盖录音数小于 N：不足
- 用户要求对比，但 evidence 只覆盖一个对象：不足

规则无法判断时再调用 LLM grader。LLM grader 只输出 JSON，不回答用户问题。证据不足且 `retrievalAttempt < 1` 时，使用 `rewriteQuery` 进行二次检索。最多二次检索一次，避免循环失控。

#### 9.0.4 validateAnswer

`validateAnswer` 判断模型回答是否被 evidence 支撑。

程序校验必须先执行：

- citation index 是否存在
- citation chunkId 是否属于本次 evidence
- citation recordingId 是否匹配
- 回答文本中出现的 `[n]` 是否都存在于 evidence
- `notEnoughEvidence=true` 时不应包含大量事实结论

LLM 校验用于语义支撑判断：

```json
{
  "valid": false,
  "reason": "回答提到了下周三上线，但证据中没有上线日期",
  "unsupportedClaims": ["下周三上线"],
  "badCitations": [2],
  "rewriteInstruction": "删除未被证据支持的上线日期，只保留证据中提到的风险和动作"
}
```

当 `valid=false` 且 `answerRewriteCount < 1` 时，使用同一 evidence 要求 answer provider 重写一次。重写时不能新增 evidence。若仍不合法，降级为 extractive answer，并提示用户查看证据片段。

#### 9.0.5 推荐执行流

```text
routeQuery
-> retrieve
-> gradeEvidence
   -> insufficient && retrievalAttempt < 1: rewriteQuery -> retrieve
   -> insufficient && retrievalAttempt >= 1: answer with notEnoughEvidence
-> answer(streaming)
-> validateAnswer
   -> invalid && answerRewriteCount < 1: rewriteAnswer
   -> invalid && answerRewriteCount >= 1: extractive fallback
-> done
```

所有节点都要打 `[rag]` 结构化日志，至少包含：

- query preview
- route JSON
- retrieval strategy
- evidence count
- score 分布
- covered recording IDs
- grader output
- answer provider
- citation validation output
- retry count
- durationMs

### 9.1 POST /api/rag/query

请求：

```json
{
  "query": "总结一下上次讨论预算超支的原因和后续动作",
  "mode": "answer",
  "limit": 10,
  "filters": {
    "recordingIds": [],
    "speakerProfileIds": [],
    "personNames": [],
    "locations": [],
    "targetPersonOnly": false,
    "createdFrom": null,
    "createdTo": null
  }
}
```

响应：

```json
{
  "queryId": "uuid",
  "query": "总结一下上次讨论预算超支的原因和后续动作",
  "answer": {
    "text": "录音中提到预算超支主要来自外包成本增加和需求范围扩大。后续动作包括重新核算剩余预算，并在下次会议前给出压缩方案。[1][2]",
    "citations": [
      {
        "index": 1,
        "chunkId": "uuid",
        "recordingId": "uuid",
        "startMs": 128000,
        "endMs": 151000
      },
      {
        "index": 2,
        "chunkId": "uuid",
        "recordingId": "uuid",
        "startMs": 402000,
        "endMs": 431000
      }
    ],
    "notEnoughEvidence": false
  },
  "evidence": [
    {
      "index": 1,
      "recording": {
        "id": "uuid",
        "title": "会议录音",
        "fileName": "meeting.m4a",
        "durationSeconds": 3600
      },
      "chunk": {
        "id": "uuid",
        "text": "这里是命中的片段文本",
        "startMs": 128000,
        "endMs": 151000,
        "speakerLabels": ["Speaker A"],
        "isTargetPerson": true,
        "matchedSpeakerProfiles": [
          {
            "id": "uuid",
            "displayName": "张三"
          }
        ]
      },
      "score": 0.82,
      "matchType": "vector",
      "url": "/recordings/<id>?t=128000&chunk=<chunkId>"
    }
  ]
}
```

`mode` 支持：

- `answer`：检索证据并生成总结或回答
- `retrieve_only`：只返回证据，不调用总结模型，用于调试检索效果

第一版为了做 ChatGPT 风格的单轮体验，建议额外提供流式响应：

```text
POST /api/rag/query/stream
```

流式接口使用 Server-Sent Events 或 fetch readable stream 均可。事件建议拆成：

- `evidence`：先返回本次召回证据列表
- `thinking_start`：模型进入 thinking 阶段，前端展示“思考中”
- `thinking_done`：thinking 结束，前端将思考内容折叠展示，不直接混入回答
- `answer_delta`：持续返回模型生成的增量文本
- `answer_done`：返回最终 answer、citations 和校验结果
- `error`：返回检索或生成错误

第一版不需要保存多轮对话历史。每次请求都是独立 query。

### 9.2 POST /api/search

可以保留一个底层检索 API，供调试和页面开发使用。

职责：

- query embedding
- pgvector 检索
- 结果聚合与去重
- 不调用总结模型

最终用户页面优先调用 `/api/rag/query`。

### 9.3 GET /api/search/chunks/:id/context

用于点击结果后展开上下文。

返回：

- 命中 chunk
- 前后若干 utterance
- 录音信息
- 可播放 URL

MVP 可以先不做独立 API，直接让详情页按 chunk ID 高亮。

## 10. 检索排序策略

### 10.1 第一版排序

第一版可以只做向量检索：

```sql
select
    c.*,
    r.title,
    r.file_name,
    1 - (c.embedding <=> $1::halfvec) as vector_score
from recording_search_chunks c
join recordings r on r.id = c.recording_id
where r.status = 'completed'
order by c.embedding <=> $1::halfvec
limit $2;
```

### 10.2 混合排序

增强版可以合并：

- vector score
- keyword score
- recording title score
- target person boost
- recency boost

建议最终分数：

```text
final_score =
  vector_score * 0.75
  + keyword_score * 0.15
  + title_score * 0.05
  + target_person_boost * 0.03
  + recency_boost * 0.02
```

### 10.3 多结果去重

用户可能问到同一段内容，多个相邻 chunk 都命中。需要做简单去重：

- 同一 recording 内，如果两个 chunk 时间范围重叠超过 50%，只保留分数更高的
- 最终结果允许多个录音出现
- 同一录音最多返回 3 个 chunk，避免单条录音刷屏

## 11. 前端页面设计

### 11.1 RAG 总结入口

建议新增页面：

- `/chat`

并把首页从 `/recordings` 改为 `/chat` 或在顶部导航增加“问录音”入口。

### 11.2 页面形态

页面采用极简单轮 RAG 总结形态。第一版不要做完整多轮聊天系统。

主要区域：

- 首屏居中一个输入框
- 用户提交后展示一条用户问题
- 下方展示一条模型回答，回答文本按流式逐步出现
- 回答完成后展示引用证据列表
- 每条证据展示录音标题、时间点、speaker、目标人物和片段文本
- 点击证据新开标签页进入详情页并定位播放，避免当前单轮对话状态丢失

API 可以保留一个很轻的调试入口：

- `生成总结`
- `只看召回片段`

这样可以快速判断问题出在检索召回还是模型总结。前端第一版可以先隐藏该 tab，固定使用 `answer` 模式。

不做：

- 多轮上下文记忆
- 会话列表
- 历史消息持久化
- 复杂 sidebar
- prompt 模板编辑器

### 11.3 回答展示

回答区至少展示：

- 用户问题
- 模型回答
- 引用编号
- 证据不足提示
- 生成失败或超时时的错误提示

回答中的引用编号要能和证据列表对应，例如 `[1]`、`[2]`。

流式展示策略：

- 检索完成后可以先显示“找到 N 段相关录音”
- 模型生成时逐字或逐 chunk 更新回答文本
- 模型 thinking 阶段不直接展示正文，只展示“思考中”；结束后可折叠查看
- 引用证据可以先折叠在回答下方，生成完成后展开
- 如果生成失败但检索成功，仍展示证据列表

### 11.4 证据卡片

每条证据至少展示：

- 引用编号
- 录音标题
- 时间范围：`02:08 - 02:31`
- speaker label 或目标人物名称
- 片段文本
- 相关度
- “播放片段”按钮

点击“播放”进入：

```text
/recordings/<recordingId>?t=<startMs>&chunk=<chunkId>
```

前端应使用新标签页打开，避免从 `/chat` 跳走后丢失当前未持久化的问答状态。

### 11.5 详情页定位

`app/recordings/[id]/page.tsx` 需要支持 query：

- `t`: 毫秒时间点
- `chunk`: chunk ID

详情页行为：

- 音频加载后自动 seek 到 `t / 1000`
- 可选：不自动播放，避免浏览器策略限制
- 高亮命中的 utterance 或 chunk 来源
- 滚动到命中位置

由于当前详情页是 server component，建议增加一个 client component：

```text
components/recording-player.tsx
components/utterance-list.tsx
```

`recording-player` 负责读取 search params 后 seek。

## 12. 基于证据生成总结

RAG 总结是 Phase 2 主链路，不是可选增强。

### 12.1 原则

- 回答和总结只能基于检索到的 chunk
- 每个关键结论都要引用证据编号
- 没有足够证据时明确说没有找到
- 不编造录音中没有出现的信息
- 输出必须可以被程序校验引用是否合法

### 12.2 Provider

新增可替换 answer provider：

```text
lib/search/answering/
  provider.ts
  local-llm.ts
  deepseek-api.ts
```

TypeScript 接口：

```ts
export interface RagAnswerProvider {
  generateAnswer(input: RagAnswerInput): Promise<RagAnswerOutput>;
}
```

`RagAnswerInput` 至少包含：

- 用户问题
- 证据 chunk 列表
- 输出语言
- 引用格式要求

`RagAnswerOutput` 至少包含：

- `text`
- `citations`
- `notEnoughEvidence`

### 12.3 输出校验

模型生成后，业务层必须校验：

- citation index 是否存在
- citation chunkId 是否属于本次召回证据
- 回答为空时是否给出证据不足标记
- JSON 解析失败时是否可以重试一次或降级为纯文本回答

如果引用不合法，不能直接把模型原文返回给用户。可以降级为：

- 返回证据列表
- 提示“模型总结引用校验失败，请查看下方片段”

## 13. 代码模块规划

建议新增：

```text
lib/search/
  chunking.ts
  graph-state.ts
  graph.ts
  normalize.ts
  planner.ts
  rag.ts
  retrieval.ts
  router.ts
  search.ts
  scoring.ts
  types.ts
  grading/
    evidence-grader.ts
    answer-validator.ts
  answering/
    provider.ts
    local-llm.ts
    local-llm-stream.ts
    deepseek-api.ts
  router/
    provider.ts
    local-llm-router.ts
    route-schema.ts

lib/audio-transcoding-analysis/embedding/
  index.ts
  provider.ts
  local-qwen.ts
  scripts/run_qwen_embedding.py

lib/db/search.ts

app/chat/page.tsx
app/api/rag/query/route.ts
app/api/rag/query/stream/route.ts
app/api/search/route.ts

components/rag-chat.tsx
components/rag-answer.tsx
components/evidence-list.tsx
components/recording-player.tsx
```

需要修改：

```text
sql/base.sql
lib/types/models.ts
lib/config/app-config.ts
lib/audio-transcoding-analysis/jobs/process.ts
lib/audio-transcoding-analysis/jobs/progress.ts
lib/db/recordings.ts
app/recordings/[id]/page.tsx
package.json
```

## 14. 实施步骤

### Step 1：数据库 schema

- 安装并初始化 `vector`、`pg_trgm`
- 新增 `embedding_models`
- 新增 `recording_search_chunks`
- 新增 `search_queries`
- 可选新增 `search_result_clicks`
- 扩展 `processing_jobs.job_type` check 约束，加入 `embedding_indexing`

验收：

- `npm run db:init` 可重复执行
- 老数据不丢失
- 没安装 pgvector 时错误信息明确

### Step 2：配置与类型

- `AppConfig` 增加 `search` 配置段
- `JobType` 增加 `embedding_indexing`
- 增加 search request / response 类型

验收：

- `npm run typecheck` 通过

### Step 3：Embedding provider

- 增加本地 Qwen3 embedding provider
- Python 脚本支持批量输入输出 JSON
- 支持模型缓存目录
- 支持 batch size

验收：

- 给定 2 条文本返回 2 个 2560 维向量
- 空文本和异常输入有明确错误

### Step 4：Chunking

- 从 `utterance_segments` 生成 chunk
- 合并同 speaker 邻近 utterance
- 保存来源 utterance 和 transcription segment IDs
- 写单元级测试或脚本级验证

验收：

- 一条录音可生成稳定 chunk
- chunk 时间范围和来源 ID 正确
- chunk 不跨很长静音或明显 speaker 切换

### Step 5：索引任务

- 新增 `embedding_indexing` job 处理分支
- `text_correction` 完成后创建索引任务
- 索引任务完成后录音状态置为 `completed`
- 录音删除时 chunk 级联删除

验收：

- 新上传录音完成后能自动生成 search chunks
- 失败可重试
- 已完成录音可手动重建索引

### Step 6：检索 API

- 实现 `POST /api/search`
- query embedding
- pgvector 检索
- 结果聚合与去重
- 写入 `search_queries`

验收：

- 输入一句自然语言可以返回相关 chunk
- 返回结果包含录音、时间点、文本、score 和跳转 URL
- 未找到时返回空数组和友好文案

### Step 7：RAG 总结 API

- 实现 `POST /api/rag/query`
- 实现 `POST /api/rag/query/stream`
- 增加 LLM router，将 query 转成 retrieval plan
- 实现 `scope_summary`、`chunk_search` 两种执行模式，并支持 recording/time/person/location 联合 scope
- 调用 retrieval executor 获取 chunk evidence 或 recording context evidence
- 实现 `gradeEvidence`，证据不足时最多一次 query rewrite 和二次检索
- 将最终 evidence 组装成证据包
- 调用本地开源 answer provider
- 实现 `validateAnswer`，引用不合法时最多一次 answer rewrite
- 非流式接口返回回答和证据列表
- 流式接口先返回 evidence，再持续返回 answer delta，最后返回校验后的最终 answer
- 流式接口支持 thinking 事件，前端默认折叠 thinking 内容

验收：

- 输入自然语言问题可以返回模型总结
- “最近两个音频都说了什么”能走录音范围总结，而不是纯向量 topK
- 证据不足时会尝试一次改写检索
- 流式接口可以逐步返回回答文本
- 总结中的引用编号都能对应到 evidence
- 引用证据包含录音和时间点
- 检索为空时模型不编造回答

### Step 8：RAG 聊天页面

- 新增 `/chat`
- 首屏居中输入框
- 单轮问题提交
- ChatGPT 风格流式回答展示
- 证据列表
- 点击证据新开标签页跳转录音详情页
- thinking 期间展示“思考中”，thinking 内容折叠展示
- 暂时隐藏“生成总结 / 只看片段”tab，固定 answer 模式，API 保留 retrieve_only

验收：

- 用户可以在页面输入问题
- 页面展示基于录音证据生成的流式总结
- 页面展示多个相关录音片段
- 证据可点击新开标签页跳转
- 页面不需要支持多轮对话

### Step 9：详情页播放定位

- 支持 `?t=<startMs>&chunk=<chunkId>`
- 音频 seek 到对应时间
- 高亮来源片段

验收：

- 从搜索结果进入详情页后，播放器定位到目标时间附近
- 用户能看到对应文本片段

### Step 10：历史数据补索引

- 增加 `search:reindex` 脚本
- 支持全量和单条录音
- 支持 force 重建

验收：

- Phase 1 已完成录音可批量生成索引
- 重跑脚本不会生成重复 chunk

### Step 11：LangGraph RAG 工作流

- 引入 LangGraph 作为在线 RAG 编排层
- 定义 `RagGraphState`，包含 query、route、retrievalAttempt、answerRewriteCount、evidence、grader、answer、validator 等字段
- 新增 graph node：`routeQuery`、`retrieveEvidence`、`gradeEvidence`、`rewriteQuery`、`generateAnswer`、`validateAnswer`、`rewriteAnswer`、`fallbackAnswer`
- 定义条件边：证据不足进入 query rewrite，引用不合法进入 answer rewrite，达到重试上限进入 fallback
- 新增 router provider，输出结构化 retrieval plan
- 新增 route schema 校验和 fallback
- 新增 recording scope retrieval：最近 N 条、日期范围、范围内主题检索
- 新增 evidence grader
- 新增 query rewrite，最多一次二次检索
- 新增 answer validator
- 新增 answer rewrite，最多一次重写
- 所有节点增加 `[rag]` 结构化日志

验收：

- query router 输出可解析 JSON
- router 失败时 fallback 到 `chunk_search`
- “最近两个音频都说了什么”能覆盖两条最近 completed 录音
- evidence 不足时会产生 grader reason 和 rewriteQuery
- answer 引用不合法时会触发一次 rewrite 或 extractive fallback
- 不会出现无限检索或无限重写
- `RagGraphState` 可序列化
- 每个 node 都可以独立测试
- SSE 事件由 graph 执行过程映射出来，不直接耦合具体 answer provider

## 15. 风险与处理

### 15.1 pgvector 未安装

风险：本地 PostgreSQL 没有 `vector` extension，`db:init` 失败。

处理：

- 文档中说明安装方式
- 初始化失败时提示需要安装 pgvector
- embedding 功能可用 `EMBEDDING_ENABLED=false` 临时关闭

### 15.2 本地 embedding 速度慢

风险：CPU 环境下 Qwen3 embedding 索引速度慢。

处理：

- batch size 可配置
- 支持轻量模型
- 索引任务异步执行
- 页面展示 `embedding_indexing` 任务状态

### 15.3 口语化 ASR 文本导致误召回

风险：转写文本存在错字或断句问题，影响检索。

处理：

- 默认索引 `utterance_segments` 校正后文本
- 保留关键词兜底
- 后续可以加入重排模型

### 15.4 时间点不够精确

风险：chunk 合并后只能定位到一段时间范围，不一定精准到某个词。

处理：

- MVP 明确定位粒度是 chunk 起点
- chunk 时长控制在 60s 内
- UI 展示起止时间，而不是宣称精确词级定位

### 15.5 模型切换导致维度不一致

风险：embedding 模型切换后向量维度变化，旧表无法写入。

处理：

- 默认固定 `Qwen/Qwen3-Embedding-4B` 和 `2560`
- `embedding_models` 记录维度
- 切换模型必须执行迁移或重建索引

### 15.6 纯向量检索无法处理范围总结

风险：“最近两个音频都说了什么”“总结昨天录音”这类问题不应该直接走 chunk 向量 topK，否则会漏掉录音整体上下文。

处理：

- 引入 LLM router
- 增加 recording scope retrieval
- 对录音级总结读取 `utterance_segments` 上下文
- evidence 中区分 chunk evidence 和 recording context evidence

### 15.7 Router 输出不稳定

风险：LLM router 输出非法 JSON、策略不在枚举内或日期解析错误。

处理：

- 所有 router 输出必须经过 schema 校验
- 失败 fallback 到 `chunk_search`
- `recordingLimit`、dateRange、speakerProfileIds 都由程序校验
- router 只生成计划，不直接生成录音 ID 或 chunk ID

### 15.8 二次检索和重写失控

风险：证据不足或引用不合法时，如果无限 rewrite，会导致延迟和成本失控。

处理：

- `retrievalAttempt <= 1`
- `answerRewriteCount <= 1`
- 每次 retry 必须记录日志和 reason
- 达到上限后返回证据不足或 extractive fallback

### 15.9 Thinking 内容误展示

风险：Qwen 等模型可能输出 `<think>...</think>`，如果直接展示会污染最终回答。

处理：

- 流式 runner 将 thinking 和 answer delta 分离
- 前端 thinking 期间仅显示“思考中”
- thinking 完成后折叠展示，不混入正式回答文本

## 16. 验收标准

Phase 2 完成时，至少满足：

- 新上传录音完成 Phase 1 后会自动生成向量索引
- 历史 completed 录音可以通过脚本补索引
- 用户可在 `/chat` 输入自然语言问题
- 系统先通过 LLM router 选择 retrieval strategy
- 系统可以召回多个相关录音片段，也可以按最近录音或日期范围召回录音级上下文
- “最近两个音频都说了什么”可以覆盖最近两条 completed 录音
- 证据不足时最多进行一次 query rewrite 和二次检索
- 系统基于召回片段生成智能总结或回答
- 回答完成后会进行引用和事实支撑校验
- 引用不合法时最多进行一次 answer rewrite 或降级为 extractive answer
- 回答中的关键结论带引用编号
- 每个引用都能追踪到具体录音、chunk、起止时间和 speaker 信息
- 点击引用证据能进入录音详情页并定位到对应时间点
- 点击引用证据默认新开标签页，不破坏当前未持久化对话
- 流式输出中 thinking 不直接展示，前端显示“思考中”并折叠 thinking 内容
- 检索证据不足时，模型不会编造总结
- 本地开源 answer provider 可跑通
- DeepSeek API provider 有清晰接口预留，后续可替换
- 录音删除后关联 chunk 自动删除
- `npm run typecheck` 通过
- `npm run build` 通过

## 17. 推荐第一轮开发顺序

第一轮主线是把“检索证据 -> 模型总结 -> 引用回放”打通：

1. schema + config + types
2. embedding provider
3. chunking + indexing job
4. reindex script
5. search API
6. 本地开源 answer provider
7. RAG query API
8. `/chat` 页面
9. 详情页定位

实现时建议保留 `retrieve_only` 调试模式。这样当总结效果不好时，可以判断问题出在 ASR、chunk、embedding、排序还是本地大模型回答上。若本地开源模型效果确实不够，再接入 DeepSeek API provider。
