# Phase 1 实施说明：录音入库、转写、Speaker Diarization 与目标人物识别

## 1. 文档目的

本文档定义项目第一阶段的实施范围、页面清单、数据结构、接口、后台任务流程和验收标准。

Phase 1 必须完成以下闭环：

- 用户可以上传录音
- 系统可以异步完成转写
- 系统可以完成 speaker diarization
- 系统可以识别已知目标人物是否出现在录音中
- 用户可以在后台页面查看录音、转写结果、说话人标签和目标人物命中结果

本阶段不做检索和 AI 问答，但必须把录音结构化成后续可检索、可分析的数据基础。

## 2. Phase 1 范围

### 2.1 本阶段要完成的能力

- 音频文件上传
- 录音管理页
- 录音处理状态管理
- Whisper 异步转写
- Speaker diarization
- 目标人物参考样本管理
- 目标人物识别
- 录音详情页展示完整文本、时间戳片段、`Speaker A / B / C` 标签
- 录音详情页展示目标人物命中片段和置信度

### 2.2 本阶段不做的能力

- 全文检索
- 向量检索
- AI 总结回答
- LangGraph 工作流
- 跨录音统一归并说话人身份
- 多目标人物复杂权限和协作管理

### 2.3 实施边界

- Phase 1 在单条录音内产出稳定的匿名说话人标签，如 `Speaker A`、`Speaker B`
- Phase 1 不要求跨录音复用同一个匿名标签
- Phase 1 识别已知目标人物，并将识别结果挂到说话片段和转写片段上
- Phase 1 页面保持后台风格，优先可用性，不追求复杂交互

## 3. 阶段目标与验收口径

### 3.1 阶段目标

当用户提供一条真实录音时，系统必须完成以下闭环：

1. 上传文件
2. 创建录音记录
3. 创建处理任务
4. 异步执行转写
5. 异步执行 speaker diarization
6. 将说话片段与转写片段对齐
7. 对说话片段与目标人物参考样本做比对
8. 保存完整文本、时间戳片段、说话人标签和目标人物识别结果
9. 在页面中查看录音、转写结果、说话人结构和目标人物命中信息

### 3.2 验收标准

- 用户可以在 Web 页面上传一条或多条录音
- 上传成功后，录音管理页中可见新记录
- 每条录音有明确状态，如 `uploaded`、`processing`、`completed`、`failed`
- 转写完成后，录音详情页展示完整转写文本
- 详情页展示带时间范围的基础分段
- 详情页展示 `Speaker A / B / C` 等匿名说话人标签
- 用户可以录入至少一个目标人物参考样本
- 录音处理完成后，详情页展示目标人物疑似命中片段和置信度
- 失败任务在页面中可见，并带错误信息

## 4. 页面清单

### 4.1 录音管理页

路径：

- `/recordings`

用途：

- 作为后台默认入口页
- 提供上传入口
- 展示录音列表和处理状态
- 进入录音详情

核心内容：

- 上传按钮
- 状态筛选
- 录音列表
- 状态统计卡片

列表字段：

- 标题或文件名
- 音频时长
- 上传时间
- 当前状态
- 最近更新时间

交互：

- 点击进入详情页
- 支持基础状态筛选
- 支持分页，每页 10 条

### 4.2 录音详情页

路径：

- `/recordings/[id]`

用途：

- 查看单条录音的完整处理结果

核心内容：

- 音频播放器
- 录音基本信息
- 当前处理状态
- 完整转写文本
- 转写分段列表
- 说话人分段列表
- 目标人物命中结果

展示要求：

- 每段显示 `start_time - end_time`
- 每段显示对应文本
- 每段显示对应 `speaker_label`
- 命中目标人物的片段显示高亮或标记

### 4.3 目标人物管理页

路径：

- `/speaker-profiles`

用途：

- 管理目标人物参考样本

核心内容：

- 目标人物名称
- 参考音频样本上传入口
- 样本列表
- 当前启用状态

## 5. 数据模型

### 5.1 recording

用途：

- 保存录音基础信息和总体状态

字段：

- `id`
- `title`
- `file_name`
- `storage_path`
- `mime_type`
- `file_size_bytes`
- `duration_seconds`
- `status`
- `error_message`
- `uploaded_at`
- `created_at`
- `updated_at`

状态取值：

- `uploaded`
- `processing`
- `completed`
- `failed`

### 5.2 transcription

用途：

- 保存整条录音的转写结果

字段：

- `id`
- `recording_id`
- `language`
- `model_name`
- `full_text`
- `segment_count`
- `created_at`
- `updated_at`

### 5.3 transcription_segment

用途：

- 保存转写后的时间片段

字段：

- `id`
- `recording_id`
- `transcription_id`
- `segment_index`
- `start_ms`
- `end_ms`
- `text`
- `speaker_label`
- `speaker_cluster_id`
- `speaker_confidence`
- `is_target_person`
- `target_person_confidence`
- `created_at`

说明：

- `speaker_label` 用于页面展示 `Speaker A / B / C`
- `speaker_cluster_id` 用于内部标识同一条录音中的同一说话人
- `speaker_confidence` 用于标记 diarization 结果可信度

### 5.4 speaker_diarization_segment

用途：

- 保存 diarization 输出的原始说话片段

字段：

- `id`
- `recording_id`
- `speaker_cluster_id`
- `speaker_label`
- `start_ms`
- `end_ms`
- `confidence`
- `is_target_person`
- `target_person_confidence`
- `matched_speaker_profile_id`
- `created_at`

说明：

- 这张表保存“谁在什么时间段说话”
- `transcription_segment` 保存“哪段文本对应哪个说话人”

### 5.5 speaker_profile

用途：

- 保存目标人物基础信息

字段：

- `id`
- `display_name`
- `status`
- `notes`
- `created_at`
- `updated_at`

### 5.6 speaker_profile_sample

用途：

- 保存目标人物参考音频样本

字段：

- `id`
- `speaker_profile_id`
- `file_name`
- `storage_path`
- `mime_type`
- `file_size_bytes`
- `duration_seconds`
- `status`
- `error_message`
- `created_at`
- `updated_at`

### 5.7 processing_job

用途：

- 保存后台处理任务状态

字段：

- `id`
- `recording_id`
- `job_type`
- `status`
- `attempt_count`
- `error_message`
- `started_at`
- `finished_at`
- `created_at`
- `updated_at`

取值：

- `job_type`: `transcription`、`speaker_diarization`、`speaker_identification`
- `status`: `pending`、`running`、`completed`、`failed`

## 6. 存储设计

### 6.1 音频文件存储

Phase 1 使用以下两种存储策略之一：

- 开发环境使用本地文件存储
- 生产环境使用对象存储

数据库中保存相对路径或对象存储 key。

### 6.2 数据库存储

Phase 1 使用：

- `PostgreSQL`

## 7. 工具与处理步骤说明

### 7.1 工具分工

Phase 1 的语音处理链路使用以下工具：

- `Whisper`：负责转写和时间戳输出
- `pyannote.audio`：负责 speaker diarization
- `SpeechBrain`：负责目标人物识别

职责边界如下：

- `Whisper` 负责“这段音频说了什么”
- `pyannote.audio` 负责“这段音频是谁在说”
- `SpeechBrain` 负责“这个说话人是不是目标人物”

### 7.2 Speaker Diarization 的具体步骤

speaker diarization 在 Phase 1 中按以下步骤执行：

1. 输入整条录音到 `pyannote.audio`
2. `pyannote.audio` 识别语音活动区间
3. `pyannote.audio` 为每个语音区间提取说话人特征
4. `pyannote.audio` 将相似说话人特征聚类
5. 系统为聚类结果分配稳定匿名标签，如 `Speaker A`、`Speaker B`
6. 系统输出每个说话片段的时间范围、`speaker_cluster_id`、`speaker_label` 和置信度

### 7.3 具体到每一步由什么工具负责

#### 1. 语音活动区间识别

这一步使用：

- `pyannote.audio`

职责：

- 找出录音中哪些时间段存在语音
- 过滤明显静音区间和无效区间

输出：

- 一组带开始时间和结束时间的语音区间

#### 2. 说话人特征提取

这一步使用：

- `pyannote.audio`

职责：

- 为每个语音区间提取 speaker embedding
- speaker embedding 表示该片段的音色和说话特征，而不是文本语义

输出：

- 每个语音区间对应的说话人特征向量

#### 3. 说话人聚类

这一步使用：

- `pyannote.audio`

职责：

- 将相似说话人特征归为同一类
- 形成单条录音内的说话人 cluster

输出：

- `speaker_cluster_id`
- 每个 cluster 对应的多个时间片段

#### 4. 匿名标签分配

这一步由应用层完成。

职责：

- 将 cluster 映射为页面可展示的 `Speaker A`、`Speaker B`、`Speaker C`
- 保证同一条录音内标签稳定

输出：

- `speaker_label`

#### 5. 转写片段与说话片段对齐

这一步由应用层完成。

职责：

- 将 `Whisper` 产出的转写片段和 `pyannote.audio` 产出的说话片段按时间范围对齐
- 为每个转写片段补上 `speaker_label`、`speaker_cluster_id` 和 `speaker_confidence`

输出：

- 带说话人标签的 `transcription_segment`

#### 6. 目标人物识别

这一步使用：

- `SpeechBrain`

职责：

- 将 diarization 后的说话片段与目标人物参考样本做相似度比对
- 判断该说话片段是否属于目标人物

输出：

- `is_target_person`
- `target_person_confidence`
- `matched_speaker_profile_id`

### 7.4 为什么 Phase 1 采用这套组合

原因如下：

- `Whisper` 负责转写，适合输出文本和时间戳
- `pyannote.audio` 直接负责 diarization，不需要 Phase 1 自己拆出单独的 VAD、embedding 和 clustering 工具
- `SpeechBrain` 更适合处理已知目标人物的声纹比对

这意味着 Phase 1 的实现不是手动拼装多个独立模型，而是：

- 用 `Whisper` 完成转写
- 用 `pyannote.audio` 完成 diarization 全流程
- 用 `SpeechBrain` 完成目标人物识别

## 8. API

### 8.1 上传录音

- 方法：`POST`
- 路径：`/api/recordings`

职责：

- 接收上传文件
- 保存文件
- 创建 `recording`
- 创建 `processing_job`

### 8.2 查询录音列表

- 方法：`GET`
- 路径：`/api/recordings`

职责：

- 返回录音列表和基础状态信息

查询参数：

- `status`
- `page`
- `pageSize`

### 8.3 查询录音详情

- 方法：`GET`
- 路径：`/api/recordings/:id`

职责：

- 返回单条录音的元数据、状态、转写内容、转写分段、说话人分段和目标人物识别结果

### 8.4 查询任务状态

- 方法：`GET`
- 路径：`/api/jobs/:id`

职责：

- 返回后台任务状态

### 8.5 创建目标人物

- 方法：`POST`
- 路径：`/api/speaker-profiles`

职责：

- 创建目标人物基础信息

### 8.6 上传目标人物参考样本

- 方法：`POST`
- 路径：`/api/speaker-profiles/:id/samples`

职责：

- 上传参考音频样本
- 保存样本文件
- 关联到目标人物

### 8.7 查询目标人物列表

- 方法：`GET`
- 路径：`/api/speaker-profiles`

职责：

- 返回目标人物及样本状态

## 9. 后台任务流程

### 9.1 总体处理链路

1. 用户上传音频文件
2. 服务端保存文件
3. 写入 `recording`
4. 写入处理任务
5. Worker 拉取待处理任务
6. 调用 Whisper 执行转写
7. 保存 `transcription`
8. 调用 speaker diarization 模型输出说话片段
9. 保存 `speaker_diarization_segment`
10. 将 diarization 结果与转写片段对齐
11. 对每个说话片段与目标人物参考样本做比对
12. 将 `speaker_label`、目标人物识别结果回写到 `transcription_segment`
13. 更新 `recording.status = completed`
14. 更新处理任务状态为 `completed`

### 9.2 Speaker Diarization 流程说明

speaker diarization 的目标是回答一句话：

- 录音里的每一段话是谁在说

Phase 1 的 diarization 流程如下：

1. 输入整条录音
2. 模型识别语音活动区间，过滤纯静音片段
3. 模型提取每个语音区间的说话人特征
4. 将相似的说话片段聚成同一个 speaker cluster
5. 为每个 cluster 分配稳定匿名标签，如 `Speaker A`、`Speaker B`
6. 输出每个说话片段的开始时间、结束时间、`speaker_cluster_id`、`speaker_label`
7. 再将这些说话片段与 Whisper 的转写片段对齐，得到“这段文本是谁说的”

说明：

- diarization 先解决“区分不同人”
- 目标人物识别再解决“其中哪个人是已知目标人物”
- 这两个流程必须串联，而不是二选一

### 9.3 转写输出要求

Whisper 输出必须保留：

- 完整文本
- 片段级文本
- 每段开始时间
- 每段结束时间
- 使用的模型名称

### 9.4 Diarization 输出要求

diarization 输出必须保留：

- `speaker_cluster_id`
- `speaker_label`
- 每段开始时间
- 每段结束时间
- `confidence`

### 9.5 目标人物识别输出要求

目标人物识别输出必须保留：

- 是否命中目标人物
- 命中片段对应的时间范围
- 每个片段的置信度
- 命中的目标人物 ID

### 9.6 Worker 设计

Phase 1 使用独立 worker 进程处理后台任务。

要求：

- Web 请求不直接同步执行转写、diarization 或识别
- Worker 可单独运行
- Worker 出错后可以定位问题
- 没有有效目标人物样本时，系统跳过目标人物识别并记录状态

## 10. 前端实现

### 10.1 页面结构

页面结构固定为：

- `/recordings`
- `/recordings/[id]`
- `/speaker-profiles`

### 10.2 UI 优先级

第一阶段优先保证：

- 信息清楚
- 状态可见
- 操作路径短
- 目标人物命中结果清晰可审阅
- 说话人标签清晰可审阅

### 10.3 状态展示

录音状态：

- `uploaded`
- `processing`
- `completed`
- `failed`

目标人物识别结果状态：

- `matched`
- `not_matched`
- `low_confidence`

## 11. 目录与模块

- `app/`：页面与路由
- `components/`：页面组件
- `lib/db/`：数据库访问
- `lib/storage/`：文件存储封装
- `lib/transcription/`：Whisper 调用封装
- `lib/diarization/`：speaker diarization 封装
- `lib/speaker-identification/`：目标人物识别封装
- `lib/jobs/`：任务调度与 worker 逻辑
- `lib/types/`：公共类型定义

## 12. 实施顺序

1. 建立项目基础骨架和页面路由
2. 完成数据库表结构
3. 完成目标人物和参考样本数据结构
4. 完成文件上传与录音入库
5. 完成录音管理页、详情页和目标人物管理页
6. 完成后台任务模型和 worker
7. 接入 Whisper 转写
8. 接入 speaker diarization
9. 完成转写片段与说话片段对齐
10. 接入目标人物识别
11. 完成结果落库与页面展示
12. 补充错误处理和状态展示

## 13. 风险与注意事项

### 13.1 长音频处理时长

风险：

- 长录音处理时间长，用户容易误以为系统卡住

应对：

- 明确展示处理中状态
- 页面轮询状态

### 13.2 音频格式兼容问题

风险：

- 用户上传的录音格式不统一

应对：

- Phase 1 限制支持格式
- 对不支持格式直接返回错误

### 13.3 转写质量不稳定

风险：

- 噪音、多说话人、低质量录音会影响转写结果

应对：

- 先接受基础可用
- 用真实数据迭代模型和预处理策略

### 13.4 Diarization 误分

风险：

- 重叠说话、短语音、噪音会导致说话人分段不准

应对：

- 页面展示匿名说话人和置信度
- 后续阶段再补人工修正能力

### 13.5 目标人物识别误判

风险：

- 样本质量和场景差异会导致误判或漏判

应对：

- 展示置信度
- 不把识别结果展示成绝对结论
- 优先使用清晰参考样本

## 14. Phase 1 完成定义

满足以下条件时，Phase 1 完成：

- 可以上传真实录音并成功保存
- 可以看到录音列表和状态变化
- 可以异步完成转写
- 可以异步完成 speaker diarization
- 可以在详情页看到完整文本和时间戳分段
- 可以在详情页看到 `Speaker A / B / C` 标签
- 可以管理目标人物参考样本
- 可以看到目标人物命中片段和置信度
- 失败情况可见且可定位

## 15. Phase 2 入口

Phase 1 稳定后，下一阶段进入：

- chunk 切分优化
- 全文检索
- 向量检索
- 搜索结果页
- 基于检索结果的问答
- 跨录音统一说话人身份管理

进入 Phase 2 前必须确认：

- 真实录音转写质量满足基础要求
- diarization 结果达到可用水平
- 目标人物识别准确率达到可用水平
- 当前表结构满足检索扩展
