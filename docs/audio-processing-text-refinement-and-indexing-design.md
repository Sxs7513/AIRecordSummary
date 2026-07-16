# 录音文本校正、连续发言与检索索引重构方案

## 1. 背景与目标

当前录音处理流水线中的文本相关节点为：

```text
ASR 原始片段
  → build_utterances
  → correct_text
  → build_search_chunks
  → embedding_indexing
```

这套顺序解决了“先将过短的 ASR 片段合并，再交给模型润色”的上下文问题，但也混淆了两种不同的数据：

- 为文本校正临时拼接的上下文；
- 面向详情页、总结和检索的最终连续发言。

`build_utterances` 当前产出的内容实际上只是润色前的中间数据，却已经使用了业务含义较强的 `UtterancesOutput`；`correct_text` 再通过替换其中的 `text` 生成最终连续发言。这导致“构建连续发言”节点在流程图中出现在文本校正之前，也使节点和 artifact 的职责不够准确。

本次重构的目标是：

1. 最终连续发言必须基于已经校正、润色后的 ASR 文本生成。
2. 文本校正仍然拥有足够的上下文，不能退化为逐个短 ASR 片段独立润色。
3. 校正过程中使用的临时聚合不能成为独立业务节点，也不投影到业务表。
4. `build_utterances` 保持独立，不与向量索引合并。
5. 检索分块基于最终连续发言，支持连续主题边界和上下文扩展。
6. 每个持久化 artifact 都有明确的数据语义、来源关系和版本。
7. 本次只调整 `audio_processing` 业务流水线；通用 `pipeline` 运行时不感知校正、发言、主题或 embedding。

## 2. 目标流水线

调整后的文本处理主链为：

```text
normalize_audio
  → diarize_pyannote
  → transcribe_qwen_asr / transcribe_funasr_nano
  → correct_text
  → build_utterances
      ├→ build_search_chunks
      │    → embedding_indexing
      └→ generate_summary
```

其中：

- `transcribe_*` 产出按说话人分离片段进行识别的原始 ASR 结果。
- `correct_text` 在节点内部构造临时校正单元，依次执行 pycorrector、规则校正和 LLM 上下文润色。
- `build_utterances` 使用校正后的单元生成最终连续发言。
- `build_search_chunks` 只负责从最终连续发言生成可检索文本单元。
- `embedding_indexing` 只负责向量化和索引投影。
- `generate_summary` 直接读取最终连续发言，与检索索引形成并行分支。

这里的关键不是简单交换 `correct_text` 和 `build_utterances`，而是引入一种仅服务于校正过程的中间数据 `CorrectionUnit`：

```text
原始 ASR 片段
  → 节点内部临时聚合为 CorrectionUnit
  → 上下文校正
  → CorrectedTextOutput
  → 最终连续发言
```

## 3. 三种文本粒度

重构后需要明确区分三种文本粒度。

### 3.1 ASR 原始片段

ASR 原始片段与 pyannote 的说话人分段对应，主要用于：

- 保存模型最原始的识别结果；
- 保留精确的时间和说话人来源；
- 定位识别错误；
- 支持后续重新校正，而不必重新运行 ASR。

其特点是数量多、文本短、上下文不足，不适合直接逐段交给 LLM 润色，也不适合直接作为最终检索单元。

### 3.2 校正单元

校正单元是 `correct_text` 内部为提高校正质量而构造的短上下文窗口。它不是最终业务数据，也不单独投影数据库。

一个校正单元只能由时间连续、同一说话人的若干 ASR 原始片段组成。建议约束：

- 说话人必须相同；
- 相邻间隔默认不超过 800～1200 毫秒；
- 单元总时长建议不超过 15～30 秒；
- 单元文本建议不超过 200～300 个字符；
- 必须保留组成它的所有原始片段引用。

校正单元比单个 ASR 片段拥有更完整的句意，但比最终连续发言更短、更保守，适合进行文本纠错。

### 3.3 最终连续发言

最终连续发言是详情页、总结、检索和后续问答共同使用的业务数据，必须由校正后的文本构建。

它可以在同一说话人、时间连续的前提下继续合并多个校正单元。建议约束：

- 说话人必须相同；
- 相邻间隔默认不超过 1200 毫秒；
- 总时长默认不超过 60 秒；
- 文本默认不超过 500 个字符；
- 保留全部原始 ASR / diarization 来源；
- 不跨说话人合并；
- 不为了相同主题合并时间上不连续的内容。

主题检索需要更大的上下文时，由 `build_search_chunks` 在连续发言之上继续构建，不应反过来污染连续发言的数据语义。

## 4. 数据契约

### 4.1 原始 ASR 输出

现有 `TranscriptSegment` 和 `TranscriptOutput` 可以继续保留：

```python
class TranscriptSegment(BaseModel):
    source_diarization_segment_id: str
    start_ms: int
    end_ms: int
    text: str
    speaker_cluster_id: str
    speaker_label: str


class TranscriptOutput(BaseModel):
    provider: Literal["qwen_asr", "funasr_nano"]
    model_name: str
    language: str | None
    segments: list[TranscriptSegment]
```

`transcript.raw` 表示未经文本校正的模型原始输出。

### 4.2 校正单元

建议新增：

```python
class CorrectionUnit(BaseModel):
    unit_index: int
    start_ms: int
    end_ms: int
    text: str
    speaker_cluster_id: str
    speaker_label: str
    source_segment_indexes: list[int]
    source_diarization_segment_ids: list[str]


class CorrectedTextOutput(BaseModel):
    provider: Literal["pycorrector_llm", "pycorrector", "rules"]
    model_name: str | None
    units: list[CorrectionUnit]
```

`CorrectionUnit` 的来源必须可追溯。仅记录拼接后的字符串是不够的，否则后续无法稳定映射回 `transcription_segments`。

建议使用新的 artifact 类型：

```text
transcript.corrected
```

不再让 `correct_text` 输出语义不准确的 `utterances.final`。

### 4.3 最终连续发言

`Utterance` 和 `UtterancesOutput` 继续作为最终业务数据：

```python
class Utterance(BaseModel):
    utterance_index: int
    start_ms: int
    end_ms: int
    text: str
    speaker_cluster_id: str
    speaker_label: str
    source_segment_indexes: list[int]
    source_diarization_segment_ids: list[str]


class UtterancesOutput(BaseModel):
    segments: list[Utterance]
```

最终 artifact 统一使用：

```text
utterances.final
```

`source_segment_indexes` 用于直接对应 `TranscriptOutput.segments`，`source_diarization_segment_ids` 用于兼容现有投影和追踪说话人来源。后续数据库投影可以逐步改成主要依赖 segment index，而不是解析字符串形式的 diarization source id。

### 4.4 检索分块

建议为 `SearchChunk` 增加主题和构建元数据：

```python
class SearchChunk(BaseModel):
    chunk_index: int
    text: str
    start_ms: int
    end_ms: int
    speaker_labels: list[str]
    speaker_cluster_ids: list[str]
    source_utterance_indexes: list[int]
    source_diarization_segment_ids: list[str]
    topic: str | None
    topic_section_index: int | None
    build_method: Literal["topic_boundary", "deterministic_fallback"]
```

持久化到 `recording_search_chunks` 时，主题和构建方式写入现有 `metadata` 字段，不需要为本次改造新增数据库列。

## 5. `correct_text` 节点设计

### 5.1 节点职责

`CorrectTextStage` 负责：

1. 读取 `transcript.raw`。
2. 按确定性规则构造临时 `CorrectionUnit`。
3. 对每个单元执行清理和 pycorrector。
4. 以多个相邻单元为一组调用本地 LLM 润色。
5. 校验 LLM 没有丢失、合并或重排单元。
6. 输出 `transcript.corrected`。
7. 无论成功或失败都释放本地校正模型。

它不负责：

- 生成最终 `utterance_segments`；
- 生成搜索 chunk；
- 写入向量索引；
- 将临时校正单元投影到业务表。

### 5.2 校正单元构造

校正单元构造应是确定性的，不能依赖 LLM 判断是否合并。推荐算法：

```text
按 start_ms 排序 ASR 片段
  → 跳过空文本
  → 当前单元为空时创建单元
  → 同说话人且 gap、duration、chars 均未超限时追加
  → 否则结束当前单元并创建下一单元
```

拼接文本时需要使用可读分隔，避免当前直接使用 `previous.text + segment.text` 导致词句粘连。中文可以默认不加空格，但要根据前后字符补充必要的空格或标点；英文和数字边界需要保留空格。

### 5.3 pycorrector 与规则校正

顺序继续保持：

```text
文本清理
  → pycorrector
  → 术语及固定替换规则
  → 本地 LLM 润色
  → 再次应用保护规则
```

pycorrector 可以逐个校正单元执行，因为它主要处理字词级错误。LLM 润色不能逐个极短片段调用，应读取相邻多个校正单元作为上下文。

### 5.4 LLM 上下文校正

每次向 LLM 提交一个有界批次，例如：

- 默认包含 8～20 个连续校正单元；
- 总输入字符数受配置限制；
- 批次之间可以带少量只读上下文；
- 只读上下文用于理解，不允许再次修改或重复输出。

模型输入必须带稳定的 `unit_index`：

```json
{
  "context_before": [
    {"unit_index": 10, "speaker": "Speaker A", "text": "……"}
  ],
  "items": [
    {"unit_index": 11, "speaker": "Speaker A", "text": "……"},
    {"unit_index": 12, "speaker": "Speaker B", "text": "……"}
  ],
  "context_after": [
    {"unit_index": 13, "speaker": "Speaker B", "text": "……"}
  ]
}
```

模型必须返回结构化 JSON：

```json
{
  "items": [
    {"unit_index": 11, "text": "校正后的文本"},
    {"unit_index": 12, "text": "校正后的文本"}
  ]
}
```

服务端需要校验：

- 返回的 `unit_index` 集合与本批次完全一致；
- 数量一致；
- 顺序可以由服务端按 index 恢复，不能信任模型输出顺序；
- 不允许返回未知 index；
- 文本为空时回退到该单元进入 LLM 前的版本；
- 结构解析失败时整批回退，不影响其他批次。

LLM 可以修正标点、同音错字、术语和口语冗余，但不能：

- 改变说话人；
- 改变时间范围；
- 将多个单元合并为一个；
- 将一个单元拆成多个；
- 删除有实际语义的内容；
- 根据常识补充录音中不存在的信息。

### 5.5 失败策略

文本校正不是全有或全无：

- pycorrector 失败：保留清理后的文本并继续。
- 单个 LLM 批次失败：该批次回退到规则校正结果，其他批次继续。
- LLM 模型加载失败：全部使用 pycorrector + 规则结果。
- 输出数量或 ID 不一致：视为该批次无效并回退。
- artifact 写入失败：节点失败，由 pipeline runtime 按节点策略重试。

节点输出需要记录实际采用的 provider 和模型，以便区分完整润色结果与降级结果。

## 6. `build_utterances` 节点设计

### 6.1 节点职责

`BuildUtterancesStage` 调整为：

1. 读取 `transcript.corrected`。
2. 对校正单元按时间排序。
3. 依据同说话人、间隔、时长和文本长度进行最终合并。
4. 重新生成连续的 `utterance_index`。
5. 合并并去重全部来源引用。
6. 输出 `utterances.final`。

它是纯 CPU、确定性节点，不调用 LLM。

### 6.2 为什么不与 `embedding_indexing` 合并

最终连续发言是业务事实，向量索引是可重建的派生索引，生命周期不同：

- 详情页直接展示 `utterance_segments`。
- 总结读取最终连续发言。
- `scope_summary` 类 RAG 查询需要读取完整连续发言。
- 后续说话人修订、字幕导出也会依赖连续发言。
- embedding 模型切换或索引损坏时，只需重建向量索引，不应重新构建业务文本。
- embedding 是可选节点，失败不应使最终转写不可用。

因此必须保留：

```text
build_utterances → build_search_chunks → embedding_indexing
```

而不是：

```text
build_utterances_and_embedding
```

## 7. `build_search_chunks` 节点设计

### 7.1 当前问题

当前实现只按最大字符数和最大时间跨度顺序装箱：

```text
加入下一条 utterance
  → 超过 1200 字或 180 秒时切块
```

它能保证 chunk 有界，但不能识别话题边界，可能出现：

- 一个 chunk 同时包含前后两个话题；
- 一个完整话题被固定长度从中间切断；
- 检索命中局部内容后缺少必要的前后文；
- 长录音中的语义召回质量不稳定。

### 7.2 节点内部结构

不新增独立 pipeline 节点。主题边界识别和确定性构建都属于“从最终连续发言生成检索分块”这一项业务职责，应内聚在 `build_search_chunks` 目录：

```text
backend/src/audio_processing/stages/build_search_chunks/
├── __init__.py
├── stage.py
├── contracts.py
├── prompt.py
├── detector.py
└── builder.py
```

建议职责：

- `stage.py`：读取 artifact、报告进度、调用 detector 和 builder、输出 artifact。
- `contracts.py`：主题区间和模型结构化输出。
- `prompt.py`：主题边界识别 prompt。
- `detector.py`：调用本地 LLM，校验连续区间。
- `builder.py`：按主题区间、字符数和时长构建最终 chunk。

因为节点内部使用本地 LLM，资源队列应从 `CPU` 调整为 `GPU_NORMAL`。确定性 fallback 本身仍在同一个资源任务内执行。

### 7.3 主题边界识别

LLM 的任务不是将整篇录音分类后把相同主题全局聚合，而是识别时间连续的主题区间。

输入是带 index 的最终连续发言：

```json
{
  "utterances": [
    {"index": 0, "speaker": "Speaker A", "text": "……"},
    {"index": 1, "speaker": "Speaker B", "text": "……"}
  ]
}
```

输出示例：

```json
{
  "sections": [
    {
      "start_utterance_index": 0,
      "end_utterance_index": 8,
      "topic": "硅光技术路线"
    },
    {
      "start_utterance_index": 9,
      "end_utterance_index": 15,
      "topic": "商业模式与成本"
    }
  ]
}
```

服务端必须验证：

- 区间按 index 升序；
- 区间不重叠；
- 区间不能越界；
- 每个 index 恰好被一个区间覆盖；
- 不允许为了主题相似而合并时间不连续的发言；
- 过大的主题区间仍需由 builder 按硬限制切分。

LLM 输出无效、超时或模型不可用时，回退到现有的字符数 + 时间跨度算法。

### 7.4 硬边界

主题区间不能替代确定性限制。最终 builder 始终执行：

- 最大字符数；
- 最大时间跨度；
- 最大 utterance 数；
- 单条超长 utterance 的安全处理；
- 空文本过滤；
- 稳定、连续的 `chunk_index`。

主题区间大于硬限制时可以拆成多个 chunk；较短的相邻主题默认不跨主题合并，除非后续通过配置明确允许。

### 7.5 overlap 策略

不建议把前后重叠文本直接复制进相邻 chunk 并持久化，因为会：

- 产生重复 embedding；
- 增加向量存储；
- 让多个近似 chunk 同时占据召回结果；
- 在回答上下文中重复相同录音内容；
- 使 source 引用范围变得模糊。

建议使用“核心 chunk 持久化 + 检索时动态扩窗”：

1. embedding 只针对核心 chunk 文本生成。
2. 向量检索命中核心 chunk。
3. 根据 `source_utterance_indexes` 向前、向后加载少量连续发言。
4. grade 和 answer 接收扩窗后的正文。
5. source 仍引用核心 chunk 和实际使用的 utterance 范围。

默认可以各扩展 1 条 utterance，后续根据录音类型和上下文长度配置。动态扩窗属于 RAG retrieval 层，不属于 `BuildSearchChunksStage`。

## 8. 数据库投影

节点和投影调整如下：

| 节点 | Artifact | 业务投影 |
| --- | --- | --- |
| `transcribe_*` | `transcript.raw` | `transcriptions`、`transcription_segments` |
| `correct_text` | `transcript.corrected` | 不新增独立业务表 |
| `build_utterances` | `utterances.final` | `utterance_segments` |
| `build_search_chunks` | `search.chunks` | 暂不单独投影 |
| `embedding_indexing` | `search.embedding_index` | `embedding_models`、`recording_search_chunks` |
| `generate_summary` | `summary.recording` | `recording_summaries` |

`RecordingProjectionService` 中原先由 `correct_text` 触发 `_project_utterances` 的逻辑，需要迁移为由 `build_utterances` 触发。

`transcriptions.full_text` 继续表示原始 ASR 全文还是最终校正全文，需要明确语义。本次建议：

- `transcriptions` 和 `transcription_segments` 保留原始 ASR 结果，便于排查和重新处理；
- 页面正文、总结和 RAG 使用 `utterance_segments` 中的最终校正文本；
- 不用校正结果覆盖原始 `transcription_segments`。

这样可以同时保留原始证据和最终可读文本。如果后续需要在页面切换“原始/校正”，无需重新运行 ASR。

## 9. 重试与重建语义

### 9.1 普通流水线重试

新依赖链中，节点失败时由现有 pipeline runtime 管理状态和重试：

```text
correct_text 失败
  → build_utterances / build_search_chunks / embedding / summary 均不可运行

build_utterances 失败
  → search 和 summary 分支均不可运行

embedding 失败
  → 最终转写和总结仍然可用

summary 失败
  → 最终转写和向量索引仍然可用
```

### 9.2 单节点人工重跑

现有“重试生成向量索引”和“重新生成总结”的接口需要适配新的 artifact 来源：

- embedding 重试读取 `build_search_chunks` 的 `search.chunks`。
- summary 重试读取 `build_utterances` 的 `utterances.final`。

启动时会清理临时 artifact，因此重试链路仍需保留当前的 artifact 恢复策略：当文件缺失时，从上游成功 stage 的持久化 `output_payload` 恢复对应 artifact。

### 9.3 文本校正重跑

本次不立即提供“只重跑文本校正”的产品入口。未来如果增加，不能只重跑单个节点后结束，因为它会使所有下游派生数据失效。正确行为应是：

```text
correct_text
  → build_utterances
      ├→ build_search_chunks → embedding_indexing
      └→ generate_summary
```

这需要 pipeline runtime 支持“从指定节点创建新的运行分支”或“使全部后继节点失效并重新排队”，应作为独立能力设计，不能用数据库手工改状态实现。

## 10. 配置建议

新增配置建议使用根目录 `.env`，由 `backend/src/settings.py` 转换为强类型配置：

```dotenv
# 校正单元
TEXT_CORRECTION_UNIT_MAX_GAP_MS=1200
TEXT_CORRECTION_UNIT_MAX_DURATION_MS=30000
TEXT_CORRECTION_UNIT_MAX_CHARS=300

# LLM 校正批次
TEXT_CORRECTION_BATCH_MAX_UNITS=16
TEXT_CORRECTION_BATCH_MAX_CHARS=4000
TEXT_CORRECTION_CONTEXT_UNITS=1

# 最终连续发言
UTTERANCE_MAX_GAP_MS=1200
UTTERANCE_FINAL_MAX_DURATION_MS=60000
UTTERANCE_MAX_CHARS=500

# 检索分块
SEARCH_CHUNK_TOPIC_DETECTION_ENABLED=true
SEARCH_CHUNK_MAX_CHARS=1200
SEARCH_CHUNK_MAX_DURATION_MS=180000
SEARCH_CHUNK_MAX_UTTERANCES=30
RAG_CHUNK_CONTEXT_WINDOW_UTTERANCES=1
```

具体默认值需要通过短录音、会议录音和长录音测试集调优，不能只依据单个样本固定。

## 11. 目录调整

建议的最终目录：

```text
backend/src/audio_processing/stages/
├── correct_text/
│   ├── __init__.py
│   ├── stage.py
│   ├── contracts.py
│   ├── unit_builder.py
│   ├── local_llm.py
│   ├── prompt.py
│   └── pycorrector.py
├── build_utterances/
│   ├── __init__.py
│   ├── stage.py
│   └── builder.py
├── build_search_chunks/
│   ├── __init__.py
│   ├── stage.py
│   ├── contracts.py
│   ├── prompt.py
│   ├── detector.py
│   └── builder.py
└── embedding_indexing/
    ├── __init__.py
    ├── stage.py
    └── engine.py
```

原则是：

- 每个业务 stage 的实现与其辅助类放在同一目录。
- stage 之间只通过 Pydantic artifact contract 连接。
- 校正单元构建器只属于 `correct_text`，不能放进通用 `pipeline`。
- 主题 detector 只属于 `build_search_chunks`。
- embedding 模型加载和推理由 `embedding_indexing` 内聚管理。

是否立即把所有单文件 stage 改为目录，不影响数据流设计；可以在实现对应改造时逐步完成，避免一次产生大量纯移动变更。

## 12. 迁移步骤

### 第一步：引入新契约并调整校正节点

1. 新增 `CorrectionUnit` 和 `CorrectedTextOutput`。
2. 将 `CorrectTextInput` 改为读取 `transcript.raw`。
3. 实现确定性的 `CorrectionUnitBuilder`。
4. 修改 LLM corrector，支持带 index 的上下文批次和结构化输出。
5. `correct_text` 输出改为 `transcript.corrected`。
6. 保留 pycorrector / LLM 失败回退。
7. 增加契约、聚合、批次校正和回退单元测试。

### 第二步：调整最终连续发言

1. 将 `BuildUtterancesInput` 改为读取 `transcript.corrected`。
2. 使 `build_utterances` 输出 `utterances.final`。
3. 增加来源 index 的完整性校验。
4. 将 `utterance_segments` 投影触发点从 `correct_text` 移到 `build_utterances`。
5. 修改 summary 和 search 的上游 artifact。

### 第三步：更新 DAG 和版本

1. 依赖顺序改为 `transcribe → correct_text → build_utterances`。
2. `build_search_chunks` 和 `generate_summary` 依赖 `build_utterances`。
3. 提升有契约变化的 stage version。
4. 提升 `recording_processing` pipeline version。
5. 更新前端节点中文名和展示顺序。

建议版本：

```text
recording_processing: 8
correct_text: 4
build_utterances: 2
build_search_chunks: 2
```

如果语义分块没有与前三步同时实现，可以先将 `build_search_chunks` 提升为版本 2 完成输入契约切换，随后实现主题边界时再提升为版本 3。

### 第四步：实现主题感知的检索分块

1. 将 `build_search_chunks.py` 改为内聚目录。
2. 实现主题边界 prompt 和 Pydantic 输出契约。
3. 实现 LLM detector。
4. 实现区间完整性校验。
5. 保留确定性 fallback builder。
6. 将 stage 资源声明调整为 `GPU_NORMAL`。
7. 把 topic 和 build method 写入 chunk metadata。

### 第五步：实现检索时动态扩窗

1. RAG retrieval 命中核心 chunk。
2. 根据来源 utterance index 加载相邻连续发言。
3. 合并重叠扩窗，避免重复 evidence。
4. grade 和 answer 使用扩窗后的证据正文。
5. source payload 不持久化正文，只保存 recording、chunk 和时间范围等引用信息。

### 第六步：端到端验证

更新真实音频 E2E：

```text
normalize
  → diarize
  → ASR
  → contextual correction
  → final utterances
  → topic-aware search chunks
  → embedding
  → summary
```

验证 pipeline status、artifact、业务投影和 RAG 召回均一致。

## 13. 测试方案

### 13.1 校正单元测试

- 同一说话人的相邻短片段可以合并。
- 不同说话人不能合并。
- gap、时长或字符数超限时切分。
- 空片段不会产生空单元。
- 来源 indexes 和 diarization IDs 完整且顺序稳定。
- 中英文、数字之间不会错误粘连。

### 13.2 LLM 校正测试

- 多单元批次能保持 index 和数量。
- 模型返回乱序时服务端恢复正确顺序。
- 模型缺少、重复或增加 index 时整批回退。
- 单个批次失败不影响其他批次。
- LLM 不可用时输出 pycorrector / 规则结果。
- 只读上下文不会重复写入输出。

### 13.3 最终连续发言测试

- 输入必须是 `CorrectedTextOutput`。
- 最终文本来自校正后单元。
- 合并规则符合 speaker、gap、duration 和 chars 限制。
- `utterance_index` 连续稳定。
- 来源引用无丢失、无重复。

### 13.4 检索分块测试

- 主题区间完整覆盖全部 utterance。
- 无效、重叠、越界或有缺口的模型输出触发 fallback。
- 单个过长主题按硬限制切分。
- 时间不连续的相同主题不会被合并。
- chunk metadata 正确记录 topic 和 build method。
- fallback 结果稳定且不丢失来源。

### 13.5 投影与集成测试

- `correct_text` 不直接写 `utterance_segments`。
- `build_utterances` 幂等覆盖 `utterance_segments`。
- embedding 投影能解析新的 chunk metadata。
- summary 使用最终连续发言。
- embedding 单节点重试能够恢复 `search.chunks` artifact。
- 真实音频 E2E 中最终 `utterance_segments` 是校正后的文本。

## 14. 可观测性

前端仍以 pipeline 节点为粒度展示进度，不展示节点内部的临时校正单元构建步骤。

建议进度：

### `correct_text`

```text
2%   读取原始转写
8%   构建校正单元
10%～35%  pycorrector
40%  术语和规则校正
45%  加载本地润色模型
50%～92%  分批 LLM 润色
97%  校验并整理校正结果
100% 写入 artifact
```

### `build_utterances`

```text
10%  读取校正文本
20%～85% 构建最终连续发言
95%  校验来源关系
100% 写入 artifact
```

### `build_search_chunks`

```text
5%   读取最终连续发言
10%  准备主题识别输入
15%～65% 识别连续主题区间
70%  校验主题区间或决定 fallback
75%～95% 构建有界 chunk
100% 写入 artifact
```

日志需要包含：

- 输入 ASR 片段数；
- 生成的校正单元数；
- LLM 校正批次数和回退批次数；
- 最终连续发言数；
- 主题区间数；
- 最终 chunk 数；
- 是否使用 deterministic fallback；
- 不打印完整录音正文。

## 15. 最终结论

本次重构采用以下边界：

```text
ASR 原始片段
  → correct_text
      └─ 节点内部临时构造上下文校正单元
  → build_utterances
      └─ 生成最终业务连续发言
  → build_search_chunks
      └─ 识别连续主题边界并构建核心检索 chunk
  → embedding_indexing
```

这样既不会因为逐个 ASR 短片段润色而降低质量，也能保证页面、总结和检索使用的是校正后的最终连续发言。同时，连续发言、检索分块和向量索引仍保持独立生命周期，后续可以分别重建、调试和升级。
