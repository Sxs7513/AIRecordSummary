# Phase 1 实施说明：录音入库、音频转码与分析、说话人识别与文本校正

## 1. 文档目的

本文档用于说明 Phase 1 当前已经实现的范围、技术方案、运行方式、数据结构、后台任务流、页面能力和验收口径。

相比最初版本，Phase 1 已经从“上传录音并完成转写/说话人分离”扩展为一个完整的本地后台处理闭环：

- TypeScript + Next.js App Router 前端与后台一体化应用
- PostgreSQL 自动建库建表
- 本地文件存储
- 批量上传录音
- 内嵌 scheduler + Node.js worker thread 线程池处理音频任务
- 多 ASR provider，可配置切换
- Qwen3-ASR + fsmn-vad 当前作为主要 ASR 路线
- pyannote speaker diarization
- SpeechBrain 目标人物识别
- utterance 展示层合并连续发言
- pycorrector、规则替换和本地 LLM 文本校正
- LLM 对同 speaker 相邻 utterance 做受限语义合并和断句
- 实时任务进度轮询
- 处理耗时落表，并在列表页展示整条录音累计处理耗时
- 失败任务重试、已完成文本校正和目标人物识别重试
- 录音与目标人物删除，并清理数据库和本地文件
- 本地模型缓存目录统一收口到项目内

Phase 1 仍然不做全文检索、向量数据库、跨录音问答和 AI 总结。这些能力进入 Phase 2。

## 2. 技术选型

Phase 1 遵循 `technical-architecture.md` 中的主技术路线：

- 前端与后台：TypeScript + Next.js App Router
- 数据库：PostgreSQL
- 数据访问：`pg` 连接池与 SQL schema 初始化
- 文件存储：项目本地 `uploads`
- 后台任务：应用启动后内嵌 scheduler + Node.js worker threads
- 音频分析：Python venv 中运行 ASR、pyannote、SpeechBrain、pycorrector、llama.cpp
- 模型缓存：项目本地 `model-cache`

应用启动入口是 `instrumentation.ts`。在 `NEXT_RUNTIME=nodejs` 时会按顺序执行：

1. 加载并写入全局应用配置
2. 检查并安装音频依赖
3. 初始化 PostgreSQL 数据库和表结构
4. 启动内嵌后台任务调度器

## 3. 配置与启动

### 3.1 `.env`

项目使用 `.env` 作为本地运行配置文件。

主要配置项包括：

- 数据库：`DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD`、`DB_NAME`、`DB_ADMIN_DATABASE`、`DB_SSL`
- 存储：`LOCAL_STORAGE_ROOT`、`PUBLIC_FILE_BASE_URL`
- 启动期依赖：`AUDIO_DEPS_AUTO_INSTALL`
- 模型缓存：`AUDIO_MODEL_CACHE_ROOT`
- 任务调度：`EMBEDDED_WORKER_ENABLED`、`EMBEDDED_WORKER_CONCURRENCY`、`EMBEDDED_WORKER_INTERVAL_MS`、`EMBEDDED_WORKER_BATCH_SIZE`
- ASR provider：`ASR_PROVIDER`
- Whisper：`WHISPER_PYTHON_BIN`、`WHISPER_MODEL`、`WHISPER_LANGUAGE`、`WHISPER_INITIAL_PROMPT_CONFIG`
- Qwen3-ASR：`QWEN_ASR_*`
- SenseVoice：`SENSEVOICE_*`
- Paraformer：`PARAFORMER_*`
- HF Whisper/BELLE：`HF_WHISPER_*`
- pyannote：`PYANNOTE_PYTHON_BIN`、`PYANNOTE_AUTH_TOKEN`
- SpeechBrain：`SPEECHBRAIN_PYTHON_BIN`
- 文本校正：`TRANSCRIPTION_CORRECTION_ENABLED`、`LLM_CORRECTION_ENABLED`、`LLM_CORRECTION_*`
- utterance 合并阈值：`UTTERANCE_MAX_DURATION_MS`、`UTTERANCE_MAX_TEXT_CHARS`、`UTTERANCE_LLM_MERGE_*`

运行 `npm` 命令前，本地建议先执行：

```bash
source ~/.bash_profile
```

### 3.2 全局配置

`lib/config/app-config.ts` 会把环境变量解析成统一的 `AppConfig`，并通过 `setGlobalAppConfig()` 写入：

- `globalThis.__aiRecordSummaryConfig`
- `process.env.AI_RECORD_SUMMARY_CONFIG`

worker thread 启动时会把完整配置通过环境变量注入子线程，确保 worker 内也能访问数据库、模型路径、Python binary、HuggingFace token 和任务参数。

### 3.3 数据库初始化

`lib/db/init.ts` 在 server 初始化阶段自动执行：

1. 使用 admin 连接检查 `DB_NAME` 是否存在
2. 不存在时自动创建数据库
3. 连接业务数据库
4. 执行 `sql/base.sql`
5. 创建或补齐所有 Phase 1 表、索引和兼容字段

本地首次启动应用时，不需要手动建库建表。

如果需要清空当前数据库中的表，可以使用：

```bash
source ~/.bash_profile
npm run db:drop-tables -- --confirm airecord --yes
npm run db:init
```

`db:drop-tables` 只删除 `.env` 当前连接的数据库中 `public` schema 下的表，不删除数据库本身。

### 3.4 音频依赖初始化

`scripts/install_audio_dependencies.sh` 会在应用启动阶段由 `ensureAudioDependencies()` 调用。

脚本行为：

- 创建或复用 `.venv-audio`
- 仅在首次创建 venv 时升级 `pip`、`setuptools`、`wheel`
- 默认使用阿里云 PyPI 镜像源
- 如果系统没有 `ffmpeg`，使用 `imageio-ffmpeg` 在 venv 中创建 ffmpeg wrapper
- 安装或检查 `torch`
- 按 `ASR_PROVIDER` 安装对应 ASR 依赖
- 安装或检查 `pyannote.audio`
- 安装或检查 `speechbrain`
- 安装或检查 `pycorrector`
- 当 `LLM_CORRECTION_ENABLED=true` 时，安装 `llama-cpp-python` 并下载 GGUF 模型到项目内缓存

## 4. ASR Provider

Phase 1 当前支持多个 ASR provider，通过 `ASR_PROVIDER` 切换：

- `whisper`
- `sensevoice`
- `paraformer`
- `hf_whisper`
- `qwen_asr`

### 4.1 Qwen3-ASR

当前主要推荐路线：

```env
ASR_PROVIDER=qwen_asr
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B
QWEN_ASR_USE_OWN_SEGMENTS=false
```

Qwen3-ASR 实测识别质量较好，但自身长音频分片不稳定，因此当前实现固定使用：

```text
fsmn-vad 分段 -> Qwen3-ASR 逐段转写
```

Qwen3-ASR 支持 `context`，项目会从 `config/initial-prompt.json` 中读取 `terms`、`phrases`、`people`、`notes`，构造带领域说明的上下文提示。

如果本地 HuggingFace 缓存中已经有完整模型快照，Qwen provider 会自动启用 HuggingFace 离线模式，避免每次启动时访问远端 metadata。

### 4.2 SenseVoiceSmall

`sensevoice` provider 使用 FunASR `FunAudioLLM/SenseVoiceSmall`。

当前实现：

- 使用 `fsmn-vad` 做语音活动检测
- 对 VAD 片段做可配置合并
- 逐段调用 SenseVoiceSmall
- 支持 `SENSEVOICE_LANGUAGE=auto`
- 支持 `SENSEVOICE_USE_ITN=true`

SenseVoiceSmall 是当前效果很稳的备选路线。

### 4.3 Paraformer

`paraformer` provider 使用 FunASR `paraformer-zh`。

当前实现：

- 使用 `fsmn-vad`
- 使用 `ct-punc-c`
- 显式指定 `model_revision=v2.0.4`
- 支持从 `config/initial-prompt.json` 读取 `terms + protectTerms` 作为 hotword

### 4.4 HF Whisper / BELLE Whisper

`hf_whisper` provider 用于 HuggingFace Whisper 类模型，例如 BELLE 中文微调模型。

当前实现：

- 使用 Transformers pipeline
- 使用 `fsmn-vad` 先分段
- 逐段转写
- 支持指定语言

### 4.5 OpenAI Whisper

`whisper` provider 使用 `openai-whisper`。

当前实现保留 Whisper 原始输出能力，但 Whisper large-v3 在部分中文录音中容易出现重复幻觉，因此当前不作为默认推荐路线。

## 5. 页面能力

### 5.1 录音列表页

路径：

- `/recordings`

能力：

- 批量上传录音
- 上传后调用后端 API 创建录音记录和初始任务
- 展示录音列表
- 展示录音处理状态
- 展示所有处理链路累计耗时
- 支持状态筛选
- 支持进入详情页
- 支持删除录音

删除录音时，系统会同时清理：

- `recordings` 及级联关联数据
- 本地上传文件
- 内存中的实时进度

### 5.2 录音详情页

路径：

- `/recordings/[id]`

能力：

- 展示音频播放器
- 展示录音基础信息
- 展示任务状态表
- 在任务状态表中展示实时进度百分比
- 展示失败任务的精简错误信息
- hover 错误信息时展示完整异常
- 支持失败任务重试
- 支持已完成的文本校正任务重新执行
- 支持已完成的目标人物识别任务重新执行
- 优先展示 `utterance_segments`
- 展示原始完整转写
- 展示原始转写分段
- 展示 speaker diarization 分段
- 展示目标人物命中结果

详情页中的文本展示分为两层：

- 原始层：`transcriptions` 和 `transcription_segments`，保留 ASR 原始输出
- 展示层：`utterance_segments`，保存合并、规则替换、pycorrector、LLM 校正和 LLM 语义合并后的结果

### 5.3 目标人物管理页

路径：

- `/speaker-profiles`

能力：

- 新建目标人物
- 上传目标人物参考音频样本
- 展示样本列表
- 删除目标人物

删除目标人物时，系统会同时清理：

- `speaker_profiles`
- `speaker_profile_samples`
- 本地样本文件

## 6. 数据模型

### 6.1 核心表

`recordings`

- 录音主表
- 保存标题、文件名、本地存储路径、MIME 类型、文件大小、时长、整体状态和错误信息

`transcriptions`

- 整条录音的 ASR 转写结果
- 保存语言、模型名称、完整原始文本和 segment 数量

`transcription_segments`

- ASR 原始分段
- 保存原始文本、开始时间、结束时间
- speaker diarization 后会回写 speaker label、cluster id 和目标人物识别结果
- 不保存 pycorrector、replacement 或 LLM 润色后的文本

`speaker_diarization_segments`

- pyannote 输出的说话人时间片段
- 保存 speaker cluster、speaker label、时间范围和目标人物识别结果

`utterance_segments`

- Phase 1 的连续发言展示层
- 基于 `transcription_segments` 和 diarization 对齐结果生成
- 保存最终展示文本
- 通过 `source_transcription_segment_ids` 记录来源 ASR segment
- `merge_reason` 记录合并原因，例如 `same_speaker` 或 `llm_same_speaker_semantic_merge`

`speaker_profiles`

- 目标人物主表
- 保存目标人物名称、状态和备注

`speaker_profile_samples`

- 目标人物参考样本表
- 保存样本文件信息、处理状态和错误信息

`processing_jobs`

- 后台任务表
- 保存任务类型、状态、重试次数、错误信息、开始时间、结束时间和处理耗时
- `processing_duration_ms` 会在任务成功或失败时落表

### 6.2 任务类型

`processing_jobs.job_type` 当前支持：

- `transcription`
- `speaker_diarization`
- `speaker_identification`
- `text_correction`

### 6.3 状态类型

录音状态：

- `uploaded`
- `processing`
- `completed`
- `failed`

任务状态：

- `pending`
- `running`
- `completed`
- `failed`

## 7. 后台任务流

### 7.1 上传触发

用户上传录音后，`POST /api/recordings` 会：

1. 保存上传文件到本地存储
2. 创建 `recordings` 记录
3. 创建初始 `transcription` 任务
4. 发送 PostgreSQL `notify processing_jobs`
5. 触发内嵌 scheduler 尝试消费任务

### 7.2 调度器

后台任务不需要单独运行 `npm run worker`。

应用启动后，`startEmbeddedJobScheduler()` 会启动内嵌调度器。调度器负责：

- 监听 PostgreSQL `processing_jobs` 通知
- 定时轮询兜底
- 维护 worker thread pool
- 根据空闲 worker 数量拉取 pending job
- 避免同一录音被多个 worker 同时处理
- 在任务完成后继续 drain 下一批任务

并发由 `EMBEDDED_WORKER_CONCURRENCY` 控制。

### 7.3 并发与防重复

当前任务领取逻辑有三层保护：

- scheduler 内存中的 `activeRecordingIds`
- 数据库查询使用 `for update skip locked`
- 数据库事务内使用 advisory lock

这样可以避免多个 worker 同时取到同一条录音的任务。

### 7.4 任务链路

完整链路如下：

1. `transcription`
   - 调用当前 `ASR_PROVIDER`
   - 保存 `transcriptions`
   - 保存原始 `transcription_segments`
   - 创建下一步 `speaker_diarization`

2. `speaker_diarization`
   - 调用 pyannote
   - 自动探测设备：`cuda -> mps -> cpu`
   - 保存 `speaker_diarization_segments`
   - 将 diarization 与 ASR segment 按时间对齐
   - 生成未校正的 `utterance_segments`
   - 创建下一步 `speaker_identification`

3. `speaker_identification`
   - 调用 SpeechBrain
   - 使用目标人物样本做声纹比对
   - 按说话人聚类聚合可用音频后再识别
   - 回写 diarization、transcription segment 和 utterance 的目标人物命中结果
   - 创建下一步 `text_correction`

4. `text_correction`
   - 重新生成 `utterance_segments`
   - 对 utterance 文本执行 pycorrector、规则替换和本地 LLM 校正
   - 生成同 speaker 相邻候选组
   - 使用本地 LLM 做受限语义合并和断句
   - 校验 LLM 输出的 sourceIds、顺序和连续性
   - 再执行一次 replacements
   - 完成后将录音状态置为 `completed`

## 8. 说话人分离与目标人物识别

### 8.1 pyannote speaker diarization

pyannote 通过 `.venv-audio/bin/python` 执行 `run_pyannote.py`。

当前实现：

- 需要 `PYANNOTE_AUTH_TOKEN`
- HuggingFace 账号需要接受相关 gated model 的使用条件
- 音频会先用 ffmpeg 转为 16k mono waveform
- waveform 会预加载到内存，避免 torchcodec/ffmpeg 动态库兼容问题
- 自动探测设备：`cuda -> mps -> cpu`
- `pipeline.to(device)`
- waveform 也会移动到同一个 device
- stderr 会打印实际使用设备，例如 `pyannote device: mps`

### 8.2 SpeechBrain 目标人物识别

SpeechBrain 用于把 diarization 片段和目标人物参考样本做相似度比对。

当前模型：

- `speechbrain/spkrec-ecapa-voxceleb`

当前策略：

- 不再逐个极短 diarization segment 独立识别
- 先按 speaker cluster 聚合可用音频
- 过滤过短片段
- 最多取一定时长做目标人物比对
- 识别结果回写到该 speaker cluster 下的相关片段

模型缓存位于项目内 `model-cache`，不是项目根目录的 `pretrained_models`。

## 9. 文本校正、替换与 LLM 合并

Phase 1 当前把文本后处理作为必要链路，而不是可选增强。

文本校正只写入 `utterance_segments`，不会覆盖 ASR 原始分段。

处理顺序：

1. pycorrector
2. replacements / regexReplacements
3. 本地 LLM 单段润色
4. replacements / regexReplacements
5. 本地 LLM 对同 speaker 候选组做语义合并和断句
6. replacements / regexReplacements

### 9.1 配置文件

配置文件：

- `config/initial-prompt.json`

主要字段：

- `intro`：ASR 初始提示的开头说明
- `prompt`：ASR 上下文说明
- `terms`：给 ASR context 或 hotword 使用的专业词
- `protectTerms`：给 pycorrector 白名单和部分 provider hotword 使用的保护词
- `llmTerms`：预留给本地 LLM 校正使用的专业词
- `phrases`：固定表达或上下文提示
- `people`：人名
- `notes`：其他提示
- `llmCorrection.system`：本地 LLM 单段润色 system prompt
- `llmCorrection.userTemplate`：本地 LLM 单段润色 user prompt
- `llmCorrection.mergeSystem`：本地 LLM 合并断句 system prompt
- `llmCorrection.mergeUserTemplate`：本地 LLM 合并断句 user prompt
- `replacements`：确定性字符串替换
- `regexReplacements`：正则替换

### 9.2 pycorrector

pycorrector 用于基础中文纠错。

如果 pycorrector 或其可选依赖不可用，系统会回退到原文本和规则替换，不会让整个任务失败。

### 9.3 规则替换

规则替换用于处理确定性的专业词错误，例如：

- 同音错字
- 英文缩写规范化
- 专业名词统一
- 人名或产品名统一

replacements 会在 LLM 单段润色后和 LLM 合并后再次执行，避免 LLM 把确定性替换结果改回去。

### 9.4 本地 LLM 单段润色

本地 LLM 用于处理“文本语义不通，但能根据上下文和专业词表判断”的场景。

当前实现：

- 使用 `llama-cpp-python`
- 使用 GGUF 模型
- 模型下载到 `model-cache/llm-correction`
- 当前默认模型配置为 `Qwen/Qwen2.5-7B-Instruct-GGUF`
- 支持多分片 GGUF 文件
- 自动检测 `llama.cpp` 是否支持 GPU/Metal offload
- 支持时设置 `n_gpu_layers=-1`
- 不支持时回退 CPU

### 9.5 LLM 语义合并与断句

LLM 语义合并发生在 `text_correction` job 的末段。

代码先生成安全候选组，LLM 只在候选组内判断是否合并。

候选规则可配置：

- `UTTERANCE_LLM_MERGE_MAX_GAP_MS`
- `UTTERANCE_LLM_MERGE_MAX_DURATION_MS`
- `UTTERANCE_LLM_MERGE_MAX_TEXT_CHARS`

默认规则：

- 同 speaker
- 不跨目标人物识别边界
- 相邻 gap 不超过 1200ms
- 合并后总时长不超过 20000ms
- 合并后文本不超过 300 字

LLM 输出必须满足：

- 所有 sourceIds 必须来自输入
- 不能遗漏、重复或改变 sourceIds 顺序
- 每个 group 的 sourceIds 必须连续
- 不能跨 speaker
- text 不能为空

校验失败时，会回退到单段润色结果。

## 10. 进度、耗时、错误与重试

### 10.1 进度

进度不落表。

worker thread 会把实时进度发送给主线程，主线程写入内存中的 progress store。前端通过：

- `GET /api/recordings/[id]/progress`

轮询获取当前录音进度。

详情页任务状态表中展示进度百分比，不展示底层模型名、frame 数或具体技术实现。

### 10.2 处理耗时

任务处理耗时会落表到：

- `processing_jobs.processing_duration_ms`

任务成功或失败时，应用根据 `started_at` 和 `finished_at` 计算耗时。

录音列表页展示的是同一条录音所有处理链路累计耗时。

### 10.3 错误展示

后台会记录任务失败错误。

前端展示策略：

- 默认只展示精简错误
- hover 时通过 tooltip 展示完整异常
- HuggingFace token 等敏感信息会在日志和错误信息中脱敏

### 10.4 重试

重试入口：

- `POST /api/jobs/[id]/retry`

支持：

- 失败任务重试
- 已完成 `text_correction` 任务重新执行
- 已完成 `speaker_identification` 任务重新执行

已完成校正任务可以重试，是为了让用户调整 `terms`、`phrases`、`replacements` 或 LLM prompt 后重新生成最终展示文本。

## 11. API 清单

录音：

- `POST /api/recordings`
- `GET /api/recordings/[id]`
- `POST /api/recordings/[id]/delete`
- `GET /api/recordings/[id]/progress`

任务：

- `GET /api/jobs/[id]`
- `POST /api/jobs/[id]/retry`

目标人物：

- `POST /api/speaker-profiles`
- `POST /api/speaker-profiles/[id]/samples`
- `POST /api/speaker-profiles/[id]/delete`

文件访问：

- `GET /uploads/[...path]`

## 12. 目录结构

音频转码与分析相关代码集中在：

- `lib/audio-transcoding-analysis`

子目录：

- `transcription`：ASR provider 分发、prompt/context 拼装
- `transcription/providers/whisper`：OpenAI Whisper provider
- `transcription/providers/sensevoice`：SenseVoiceSmall provider
- `transcription/providers/paraformer`：Paraformer provider
- `transcription/providers/hf-whisper`：HuggingFace Whisper/BELLE provider
- `transcription/providers/qwen-asr`：Qwen3-ASR provider
- `diarization`：pyannote speaker diarization
- `speaker-identification`：SpeechBrain 目标人物识别
- `text-correction`：pycorrector、规则替换、本地 LLM 润色和 LLM 合并
- `jobs`：内嵌 scheduler、任务处理、worker thread、进度 store
- `runtime`：Python 运行环境、依赖检查和 JSON 执行工具
- `scripts`：非 ASR provider 专属的 Python 脚本

其他关键目录：

- `app/api`：后端 API routes
- `app/recordings`：录音列表和详情页
- `app/speaker-profiles`：目标人物管理页
- `config`：ASR context 和文本校正配置
- `sql`：数据库 schema
- `scripts`：启动期依赖安装、数据库初始化、模型下载辅助脚本
- `uploads`：本地上传文件
- `model-cache`：本地模型缓存

## 13. 验收标准

Phase 1 当前验收口径：

- 首次启动应用时，可以自动检查并创建 PostgreSQL 数据库和表
- 首次启动应用时，可以自动准备 `.venv-audio` 和音频分析依赖
- 用户可以批量上传录音
- 上传后每条录音都会自动创建 `transcription` 任务
- 后台 scheduler 可以自动消费任务，不需要单独运行 worker 命令
- 多个录音可以通过 worker thread pool 并发处理
- 同一录音不会被多个 worker 重复处理
- 录音详情页可以看到任务状态和进度百分比
- 录音列表页可以看到整条录音累计处理耗时
- ASR 原始转写会保存到 `transcriptions` 和 `transcription_segments`
- pyannote 结果会保存到 `speaker_diarization_segments`
- 目标人物识别结果会回写到相关分段
- 文本校正最终结果只保存到 `utterance_segments`
- LLM 可以在受限候选组内做同 speaker 语义合并和断句
- 失败任务可以重试
- 已完成文本校正任务可以重试
- 已完成目标人物识别任务可以重试
- 录音删除会清理数据库和本地录音文件
- 目标人物删除会清理数据库和本地样本文件

## 14. Phase 1 暂不处理

以下能力不属于当前 Phase 1：

- 全文检索
- 向量检索
- pgvector、Milvus 或 Qdrant 接入
- 跨录音统一说话人身份
- AI 总结
- AI 问答
- LangGraph 工作流
- 多用户权限
- 云端对象存储
- 生产级任务队列

这些能力可以进入 Phase 2 或后续阶段设计。
