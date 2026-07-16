# Qwen ASR 连续语音窗口与说话人对齐方案

## 1. 目标与结论

当前转写流程先使用 pyannote 的 speaker turn 裁剪音频，再分别交给 Qwen ASR。这会在连读、弱读或 diarization 边界不准时，将一个音节切成两段，导致 ASR 缺少必要的声学上下文。

本方案将转写链路改为“**先识别连续语音、再按时间归属 speaker**”的模式：

```text
normalize_audio
  ├→ preprocess_asr_audio（轻量 FFmpeg 前置处理）
  └→ diarize_pyannote（speaker 区间、静音/重叠信息）

audio.asr_preprocessed + diarization.pyannote
  → transcribe_qwen_asr（聚合连续语音窗口，但不按 speaker 裁音频）
  → transcript.asr_windows

transcript.asr_windows
  → correct_asr_windows（保守校正窗口原文，保留窗口范围）
  → transcript.corrected_windows

transcript.corrected_windows + audio.asr_preprocessed + diarization.pyannote
  → align_transcript（文字时间对齐 + token → speaker 归属）
  → build_utterances / summary / search
```

本次新增三个独立 stage：

- `preprocess_asr_audio`：对整段标准化录音做轻量 DSP，产出时间轴不变的 `audio.asr_preprocessed`；
- `correct_asr_windows`：对窗口级 ASR 原文做保守校正，产出仍可对应原窗口音频的最终展示文本；
- `align_transcript`：消费校正后的窗口文本、同一音频和 pyannote 结果，产出字/词时间戳及最终 speaker transcript。

因此 Qwen ASR 不再持有 aligner 配置，也不依赖 speaker turn 才能转写。第一版的 `align_transcript` 实现使用 `Qwen3-ForcedAligner-0.6B`，但输入/输出契约不绑定 Qwen；未来可替换为其他 forced aligner，或复用到其他 ASR provider。ForcedAligner 只执行一次，且对齐的是校正后的最终展示文本。

### 1.1 Stage 职责边界

```text
transcribe_qwen_asr
  输入：前置处理音频 + pyannote 原始区间
  作用：将连续语音聚合成窗口，识别文字
  输出：窗口范围 + 原始窗口文本；不输出 speaker 归属

correct_asr_windows
  输入：原始窗口文本
  作用：保守地校正错字、术语、标点和断句；不跨窗口搬移内容
  输出：保留相同窗口范围的最终展示文本

align_transcript
  输入：前置处理音频 + 校正后窗口范围/文本 + pyannote 原始区间
  作用：ForcedAligner 生成 token 时间戳，再按时间归属 speaker
  输出：最终 speaker transcript segment + 可选 token
```

因此，**最终 speaker 分段在 `align_transcript` stage 中生成**。ForcedAligner 模型本身不识别 speaker；它只提供“文字在何时说出”，`align_transcript` 代码再将该时间与 pyannote 的原始区间求交集，聚合出“谁在何时说了什么”。

## 2. 非目标

- 不由 Qwen 或 ForcedAligner 识别 speaker。speaker 身份仍完全来自 pyannote。
- 不在第一版处理真正的双人重叠语音分离；重叠区域须显式标记为不确定或按既定主 speaker 规则处理。
- 不在第一版实现逐字编辑，也不试图让 LLM 校正后的新增/改写文字沿用原始字级时间戳。
- 第一版只为 Qwen ASR 接入连续窗口输出；`align_transcript` 的契约保持 provider 无关，Fun-ASR-Nano 可在后续复用。

## 3. 模型能力与约束

`qwen-asr==0.0.6` 提供可独立调用的 `Qwen3ForcedAligner`：

```python
aligner = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    ...,
)
result = aligner.align(audio=window_audio, text=window_text, language=language)
```

返回的是 ForcedAligner 对“音频 + 已识别文本”的对齐结果；文本不要求来自 Qwen。中文的最小对齐单位通常为单个汉字；英文等空格语言通常为词。它只回答“这段文字何时说出”，不回答“谁说出”。

当前本地模型权重约为：

| 模型 | 缓存权重约占用 |
| --- | ---: |
| `Qwen3-ASR-1.7B` | 4.36 GB |
| `Qwen3-ForcedAligner-0.6B` | 1.7 GB |
| 合计 | 6.1 GB |

项目在 MPS 上使用 `float16`，其占用为统一内存。加上加载临时副本、特征、attention 中间张量、MPS allocator 缓存和 pyannote，30–60 秒窗口同时加载的实测峰值预计约为 10–14 GB，且应以实际监控为准。

## 4. 后端设计

### 4.1 配置与安装

新增以下 provider 无关的配置：

```env
ASR_AUDIO_PREPROCESSING_ENABLED=true
TRANSCRIPT_ALIGNMENT_ENABLED=true
TRANSCRIPT_ALIGNMENT_PROVIDER=qwen_forced_aligner
TRANSCRIPT_ALIGNMENT_MODEL=Qwen/Qwen3-ForcedAligner-0.6B
ASR_SPEECH_WINDOW_TARGET_DURATION_MS=30000
ASR_SPEECH_WINDOW_MAX_DURATION_MS=60000
ASR_SPEECH_WINDOW_OVERLAP_MS=500
```

`scripts/install_audio_dependencies.sh` 必须在 `TRANSCRIPT_ALIGNMENT_ENABLED=true` 时下载 aligner 到 `model-cache/huggingface/hub`。运行时只从本地 snapshot 加载，不得触发联网下载。

### 4.2 `preprocess_asr_audio` stage

输入 `audio.normalized`，输出 `audio.asr_preprocessed`。第一版只运行高通、温和动态压缩、响度归一化和 limiter，必须保持采样率、单声道和原始时间轴不变。

- `diarize_pyannote` 继续读取 `audio.normalized`，避免 DSP 改变 speaker embedding；
- 所有 ASR provider 与 aligner 读取 `audio.asr_preprocessed`；
- stage 使用 CPU 队列；失败时可显式回退到 `audio.normalized`，并在 artifact metadata 中记录 `preprocessing_applied=false`；
- 原先 `transcribe_qwen_asr` 内的整录音前置处理和逐片段低音量增强应移除，避免重复处理。

### 4.3 `transcribe_qwen_asr` stage

输入包括 `audio.asr_preprocessed` 和 `diarization.pyannote`。它使用所有 pyannote 原始区间构造连续语音窗口、进行 ASR，并输出不带 speaker 的 `transcript.asr_windows`：

```python
class AsrWindowTranscript(BaseModel):
    window_index: int
    # 实际送入 ASR 的音频范围，包含 overlap；align stage 必须按此范围重新裁音频。
    input_start_ms: int
    input_end_ms: int
    # 窗口独占文本归属范围，不包含 overlap；用于 token 去重。
    core_start_ms: int
    core_end_ms: int
    language: str | None
    text: str
    # 仅用于溯源/诊断，不用于在此阶段给文字分配 speaker。
    core_diarization_segment_ids: list[str]

class AsrWindowTranscriptOutput(BaseModel):
    provider: str
    model_name: str
    alignment_supported: bool
    windows: list[AsrWindowTranscript]
```

对应的 artifact JSON 示例：

```json
{
  "provider": "qwen_asr",
  "model_name": "Qwen/Qwen3-ASR-1.7B",
  "alignment_supported": true,
  "windows": [
    {
      "window_index": 0,
      "input_start_ms": 0,
      "input_end_ms": 30500,
      "core_start_ms": 0,
      "core_end_ms": 30000,
      "language": "Chinese",
      "text": "我们先确认一下这个方案的时间安排。",
      "core_diarization_segment_ids": ["speaker-00-0", "speaker-01-12400"]
    }
  ]
}
```

窗口聚合规则仅使用 pyannote 区间来判断连续语音、静音 gap 与窗口边界；**不会**按 `speaker_cluster_id` 将音频拆开，也不会在该 stage 给文字分配 speaker。`text` 必须是未经过 LLM/规则润色的 ASR 原文。窗口范围、core 范围和覆盖的原始 diarization segment ID 必须随文本完整保留到对齐完成。

### 4.4 `correct_asr_windows` stage

输入 `transcript.asr_windows`，输出 `transcript.corrected_windows`。它只校正单个窗口中的错字、术语、标点和断句，不允许跨窗口搬移、摘要、扩写或重排内容；必须原样保留每个窗口的 `window_index`、输入范围、core 范围和语言。

```python
class CorrectedAsrWindowTranscript(AsrWindowTranscript):
    original_text: str
    text: str  # 最终面向用户展示、并将送入 ForcedAligner 的文本

class CorrectedAsrWindowTranscriptOutput(BaseModel):
    provider: str
    corrector_name: str | None
    windows: list[CorrectedAsrWindowTranscript]
```

校正提示词必须明确要求“只修正识别错误，不改写、不总结、不补充未说出的内容”。若某个窗口的编辑比例超过保守阈值，或校正器失败，则该窗口回退为 `original_text`，保证后续对齐的文本仍接近真实语音。

### 4.5 `align_transcript` stage

输入为 `audio.asr_preprocessed`、`diarization.pyannote`、`transcript.corrected_windows`；输出为最终展示和下游消费的 `transcript.aligned`，包含 speaker segment 与可选 alignment tokens。它对每个窗口的“音频 + **校正后文本**”对齐后，再以 token 时间与**同一份原始 diarization 区间**求交集，将文字分配回原始 speaker turn。

该 stage 的完整算法：

1. 根据 `AsrWindowTranscript.input_start_ms/input_end_ms` 从 `audio.asr_preprocessed` 裁出与 ASR 完全相同的窗口音频；
2. 将窗口音频、`text`、`language` 交给 `TRANSCRIPT_ALIGNMENT_PROVIDER`，得到字/词级 token 时间戳；
3. 仅保留 token 时间中点位于 `core_start_ms/core_end_ms` 的 token，消除相邻 ASR 窗口 overlap 引入的重复；
4. 将每个保留 token 与 `diarization.pyannote` 的**原始**区间按重叠时长匹配，得到 speaker 或 `ambiguous/unmatched`；
5. 将时间连续、同 speaker 的 token 聚合为最终 `TranscriptSegment`；speaker 切换、长停顿、标点和不确定 token 都是分段边界；
6. 输出 `transcript.aligned`，供 `build_utterances`、数据库投影及前端使用。

即使一个 30–60 秒窗口中包含多位 speaker，也只会被 Qwen 识别一次；speaker 分段只在本 stage 的第 4、5 步完成。

第一版采用统一顺序模式：`transcribe_qwen_asr` 完成所有窗口后自然释放 1.7B ASR 模型；`align_transcript` 再加载 0.6B ForcedAligner，对全部“窗口音频 + 已识别文本”批量对齐。每个 stage 只加载一种模型，避免 MPS 同时常驻 2.3B 参数。

实现可由 `TRANSCRIPT_ALIGNMENT_PROVIDER` 选择，但所有实现必须遵守相同 `transcript.corrected_windows → transcript.aligned` artifact 契约。后续可加入 Qwen 与 aligner 并发的性能优化，但不应影响 stage 边界。

### 4.6 连续语音窗口

pyannote 的结果继续保留，但不再作为 ASR 音频裁剪边界。窗口构建规则：

1. 将所有 diarization segments 的时间区间合并为“有语音”时间轴；speaker 不同也可以属于同一连续窗口。
2. 优先在静音 gap 处切分；目标核心时长 30 秒，最大 60 秒。
3. 若没有适合的静音点，按 30 秒核心边界切；输入窗口在两端扩展 500ms，但不得超出录音范围。
4. 相邻输入窗口可重叠，但每个窗口定义独占的 `core_start_ms`、`core_end_ms`。
5. 对齐后的 token 仅在其时间中点落入当前 core 区间时保留；这比按文本去重稳定，也不会重复边界词。

示例：

```text
窗口 A 输入：  0.0–30.5s，核心：0.0–30.0s
窗口 B 输入： 29.5–60.5s，核心：30.0–60.0s

token 中点 < 30.0s 归 A；>= 30.0s 归 B。
```

### 4.7 token 到 speaker 的归属

对每个对齐 token：

1. 计算 token `[start_ms, end_ms]` 与所有 pyannote speaker turn 的重叠时长。
2. 有唯一最大重叠：归属该 speaker，保存 `attribution_status="matched"`。
3. 未与任一 speaker turn 重叠：保存 `speaker=None`，状态为 `unmatched`。
4. 最大重叠比例低于阈值，或 token 位于 pyannote overlap 区域：保存 `speaker=None`，状态为 `ambiguous`；第一版不强行猜测。
5. 按时间顺序合并相邻且同 speaker 的 token，遇到 speaker 变化、长停顿、标点或不确定 token 时断开，得到 `TranscriptSegment`。

建议初始阈值：最大 overlap 至少覆盖 token 时长的 60%，并且比第二候选至少多 100ms；这些值必须通过真实录音抽样校准。

### 4.8 数据契约与持久化

现有 `TranscriptSegment` 只有一个 `source_diarization_segment_id`，不能表达“由多个 turn 共同覆盖”或 token 级 speaker 归属。新增 alignment 数据，而不污染最终连续发言语义。

建议 artifact 扩展：

```python
class AlignedTranscriptToken(BaseModel):
    token_index: int
    text: str
    start_ms: int
    end_ms: int
    speaker_cluster_id: str | None
    speaker_label: str | None
    attribution_status: Literal["matched", "ambiguous", "unmatched"]
    source_window_index: int

class TranscriptOutput(BaseModel):
    provider: Literal["qwen_asr", "funasr_nano"]
    model_name: str
    language: str | None
    segments: list[TranscriptSegment]
    alignment_tokens: list[AlignedTranscriptToken] | None = None
    alignment_model_name: str | None = None
```

`alignment_tokens` 是**可选能力**：旧 artifact、Fun-ASR、未启用 timestamp alignment 的 Qwen 录音均为 `None`，不得以空数组假装“已完成但没有 token”。

数据库迁移新增可选的 `transcription_tokens` 明细表，而不是将大量 token JSON 塞进 `transcription_segments`：

```text
id UUID PK
recording_id UUID FK
transcription_id UUID FK
token_index INT
transcription_segment_id UUID NULL FK
source_window_index INT
text TEXT
start_ms INT
end_ms INT
speaker_cluster_id TEXT NULL
speaker_label TEXT NULL
attribution_status TEXT
```

要求：

- `(transcription_id, token_index)` 唯一；
- `start_ms/end_ms` 建索引，支持按播放时间查询 token；
- token 若属于最终 transcription segment，写入 `transcription_segment_id`；不确定 token 可以为 `NULL`；
- 现有 `transcription_segments`、`utterance_segments` 保持兼容，API 继续可用。
- token 表对每个 transcription 都是可选的一对零/多关系；没有 token 行表示该 transcription 不具备逐字对齐能力，不能视为失败。

本方案不向用户展示“原始文本 / 润色文本”两个主视图。`transcript.asr_windows` 的原始 Qwen 文本仅作为内部溯源 artifact；`transcript.aligned` 才是唯一的用户可见文本与逐字回跳来源。

因为 ForcedAligner 接收的是 `correct_asr_windows` 输出的最终展示文本，所有展示 token 都直接由“最终文本 + 实际音频”对齐产生，不需要通过字符串 diff 推测时间戳。前提是校正器必须保守：不得改写语义、补写未说内容或跨窗口移动内容。

对齐质量保护：

- `correct_asr_windows` 记录 `original_text` 与最终 `text`，并计算编辑比例；超过阈值时回退原始文本；
- `align_transcript` 对空文本、token 数异常或时间戳明显异常的窗口标记失败；该窗口回退到原始文本后重试一次；
- 若重试仍失败，整个对齐 stage 失败并可单独重跑，不能把无法对齐的润色文本伪装成精确字级时间。

这样前端只需读取 `transcript.aligned` 及其可选 token：润色后的文字、speaker 分段、逐字点击和播放同步高亮始终来自同一份最终数据。

### 4.9 Stage、投影与失败回退

`preprocess_asr_audio`、连续窗口版 `transcribe_qwen_asr`、`correct_asr_windows` 和 `align_transcript` 均应拥有新 stage version；pipeline definition 也必须升级，以避免复用旧 artifact。

建议进度按 stage 分开报告：

```text
preprocess_asr_audio   0–100%  整录音轻量 DSP
transcribe_qwen_asr   0–100%  构造窗口、加载 ASR、窗口转写、写入 transcript.asr_windows
correct_asr_windows   0–100%  保守校正窗口文本、写入 transcript.corrected_windows
align_transcript      0–100%  加载 aligner、时间对齐、token → speaker、聚合 transcript.aligned
```

失败策略：

- `preprocess_asr_audio` 失败：按显式配置回退 `audio.normalized` 或终止，不得静默改变输入来源；
- `align_transcript` 失败但 ASR/校正成功：该 stage 失败并可单独重试；保留 `transcript.corrected_windows`，无需重跑 ASR 或校正；
- 单个窗口失败：保留窗口索引和错误；初版建议整条任务重试，避免静默丢字。

### 4.10 并发加载试验与验收阈值

先用 10 分钟、30 分钟、60 分钟各一条真实录音进行试验，记录：

- MPS/系统统一内存峰值、swap、每分钟音频的处理时间；
- ASR CER 抽样、speaker attribution 错配率、overlap 区域占比；
- 窗口边界附近（前后 1 秒）与旧流程相比的漏字/错字数量；
- token 时间点击误差（抽样应在 300ms 内）。

验收建议：无 swap 或可接受的轻微 swap；60 分钟录音端到端耗时可接受；边界错字显著下降；speaker 归属不劣于现有 pyannote 直出结果。concurrent 仅是未来实现优化，不属于第一版的运行时配置，也不改变 artifact、API 或前端。

## 5. API 设计

现有 `GET /api/recordings/{id}` 保留 `transcription_segments` 与 `utterance_segments`，以确保详情页、总结、检索和旧客户端不受影响。

新增可按需加载的 endpoint，避免详情页一次返回数万 token：

```text
GET /api/recordings/{recording_id}/transcription-tokens
  ?start_ms=0
  &end_ms=60000
```

返回：

```json
{
  "available": true,
  "alignment_model_name": "Qwen/Qwen3-ForcedAligner-0.6B",
  "items": [
    {
      "token_index": 128,
      "text": "我",
      "start_ms": 12340,
      "end_ms": 12480,
      "speaker_label": "Speaker 1",
      "speaker_cluster_id": "SPEAKER_00",
      "attribution_status": "matched"
    }
  ]
}
```

API 规则：

- 参数必须成对提供或都省略；默认和最大查询跨度设为 60 秒；
- 按 `token_index` 排序；
- 未生成 alignment 时返回明确的能力状态，而不是错误：

```json
{ "available": false, "alignment_model_name": null, "items": [] }
```

- 详情接口的段级对象可渐进增加可选的 `alignmentAvailable?: boolean` 与 `tokenCount?: number` 摘要字段，但不内联 token；字段缺失时前端按 `false` 处理，兼容旧响应；
- 不向客户端暴露内部 ASR 输入窗口的临时文件路径。

## 6. 前端设计

现有 `RecordingPlayer` 已支持 `recording-play-segment` 自定义事件，`UtteranceList` 已支持整段点击播放。本次不改播放器协议，只扩展事件和文本展示。

### 6.1 第一版：句段回跳 + token 阅读模式

1. 最终连续发言保持现有显示与点击整段播放，确保润色后的可读性不变。
2. 仅当 token endpoint 返回 `available: true` 时，在“转写分段”或新建“逐字对齐”折叠区展示最终对齐文本；不再额外展示 ASR 原文。
3. 前端根据播放器当前时间，在已加载 token 中寻找 `start_ms <= current_ms < end_ms` 的 token，高亮它及所属 speaker。
4. 点击 token：派发 `recording-play-segment`，其中 `startMs=token.start_ms`，`endMs=min(token.end_ms + 800, 所属段结束)`；用户可立即听到该字及少量后文。
5. 只在播放器附近的可见时间范围请求 token API；首次加载当前段所在的 60 秒范围，滚动/播放跨范围再预取相邻窗口。
6. `ambiguous` 与 `unmatched` token 使用弱提示样式（如虚线底纹/提示图标），不伪装成已可靠归属的 speaker。

事件可扩展为：

```ts
interface PlaySegmentEventDetail {
  startMs: number;
  endMs: number;
  tokenIndex?: number;
  autoplay?: boolean;
}
```

旧调用方只传 `startMs/endMs`，无需改动。

### 6.2 第二版：播放同步高亮

- `RecordingPlayer` 向页面状态或 context 上报 `timeupdate` 的当前毫秒；不要为每一帧触发网络请求。
- 对已加载 token 使用二分查找定位当前 token，避免每次 `timeupdate` 扫描全量列表。
- 当前 token 滚出阅读容器时，平滑滚动到可视区域；用户手动滚动期间暂停自动滚动，避免抢焦点。
- 支持快捷键：左右方向键在 token 间跳转，Enter 播放当前 token 所在的短片段。

### 6.3 可访问性与降级

- 每个 token 必须是可聚焦按钮或具有等价键盘操作，不仅依赖鼠标点击。
- 无 alignment token、Fun-ASR provider 或旧录音：继续展示现有段级 UI，不显示逐字功能入口；段落点击播放、URL `?t=` 回跳和 speaker 展示必须不受影响。
- 对超长录音采用虚拟列表或时间窗口加载，禁止把全部 token 一次渲染到 DOM。

## 7. 实施顺序

1. **模型与基准试验**：下载 aligner，先验证 MPS sequential 模式的字级 timestamp、30/60 秒窗口内存与耗时；随后再以 concurrent 作为性能对照。
2. **后端最小闭环**：实现 `preprocess_asr_audio`、连续窗口版 Qwen ASR、独立 `align_transcript` 和 token → pyannote 归属；先只输出 artifact 和日志，不改数据库投影。
3. **数据契约与迁移**：新增 `transcript.asr_windows`，为 `TranscriptOutput` 增加 alignment metadata，创建 `transcription_tokens` 表、投影和读 API；升级相关 stage 与 pipeline version。
4. **前端第一版**：段级回跳保持不变，新加 raw token 阅读/点击回跳；以 feature flag 隐藏未完成能力。
5. **评估与优化**：对真实录音 A/B 对比 sequential 与旧流程；仅在 sequential 稳定后，才评估 concurrent 是否值得作为可选优化，无需改 API 与前端。
6. **前端第二版**：实现播放同步逐字高亮、预取和键盘操作。

## 8. 测试清单

后端单元测试：

- 连续窗口按静音优先切分、最大时长与 overlap core ownership；
- token 不重复、不遗漏地归属到唯一核心窗口；
- token 与单一/多个/零个 speaker turn 的归属；
- overlap 或低 overlap 比例返回 `ambiguous`；
- token 聚合为最终 speaker segment 时的时间、文本和断句；
- concurrent 试验 OOM 后顺序模式回退；
- 旧 `TranscriptOutput` artifact 与 Fun-ASR 路径兼容。

集成测试：

- 包含两位说话人轮换、短停顿、连读边界和 overlap 的固定音频；
- 字级 timestamp 可映射为正确的 speaker 段；
- `build_utterances`、总结和检索消费 `transcript.aligned`，且不依赖旧的 `correct_text` artifact；
- 迁移后重跑录音不会留下旧 token。

前端测试：

- 点击 token 派发正确时间范围并驱动播放器；
- 播放时间变化高亮正确 token；
- 未对齐或旧录音稳定降级到段级 UI；
- 60 分钟录音 token 分页/虚拟列表不导致页面卡顿。
