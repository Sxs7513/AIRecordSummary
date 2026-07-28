# Pipeline 示例

先看 [example.py](example.py)。构建一条流水线只需两部分：

1. `PipelineDefinition`：声明节点及依赖关系。
2. `StageRegistry`：注册 Definition 中节点对应的 stage 插件。

`run_example()` 只是一个无数据库的演示运行器；它直接遍历 `example_pipeline` 的拓扑顺序，不手写节点执行顺序。生产业务域负责把自己的 Definition、stage registry 与资源调度器组合起来；`pipeline` 本身不包含具体业务流水线。

示例的 Definition 为：

```text
PipelineDefinition
  prepare ──► consume
      │            │
      │ output     │ input_payload["upstream_outputs"]
      └────────────┘
```

业务域构建真实 pipeline 时遵循相同结构：

1. 实现 `name`、`version`、`resource_queue`、`retry_policy` 和 `async run()`。
2. 在应用启动时通过 `StageRegistry.register()` 显式注册实例。
3. 在 `PipelineDefinition` 中用同一组 `name/version` 声明节点与依赖。
4. 业务域自己的持久化与协调器将 Definition 映射为运行记录；`ResourceScheduler` 只按 CPU/GPU 队列执行 callable，业务协调器消费结果后再推进下游。

可以不连接数据库直接运行示例：

```bash
backend/.venv/bin/python -c \
  'import asyncio; from l1_foundation.pipeline.example import run_example; print(asyncio.run(run_example()))'
```

输出中的 `upstream_outputs.prepare` 就是 `consume` 节点如何读取 `prepare` 的结果。
