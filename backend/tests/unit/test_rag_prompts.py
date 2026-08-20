from l2_core.rag.prompts import answer_plan_prompt, answer_prompt, grade_prompt, retrieval_terms_prompt, route_prompt


def test_route_prompt_relies_on_response_schema_without_embedding_format_instructions() -> None:
    prompt, values, _parser = route_prompt("最近的录音讲了什么", "[]")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "只输出唯一一个符合 schema 的 JSON 对象" in rendered
    assert "确定录音范围、选择检索策略，并为正文检索提取 content_query" in rendered
    assert "不提取 topic" in rendered
    assert "不判断现实实体、证据充分性或答案是否存在" in rendered
    assert "scope_summary" in rendered
    assert "累计发言时长" in rendered
    assert "file_names" in rendered
    assert "必须先完成录音范围解析，再选择问答策略" in rendered
    assert "范围条件不能作为选择 strategy 的依据" in rendered
    assert "只移除已经由本次 route 结果表达的录音集合范围条件" in rendered
    assert "不得机械删除所有时间、人名或地点" in rendered
    assert "每个完整文件名都必须逐字出现在 inferred_filters.file_names" in rendered
    assert "metadata_lookup 可访问每条录音的可信元数据" in rendered
    assert "可对它们计数、筛选、分组、排序、比较和聚合" in rendered
    assert '"speaking_duration_seconds": "number"' in rendered
    assert "fact_lookup" in rendered
    assert "时间长度中的数字不得写入 recording_limit" in rendered
    assert "录音范围可以缺省" in rendered
    assert "当前用户全部可访问的已完成录音" in rendered
    assert "time_range 只表示对录音集合本身的创建或上传时间约束" in rendered
    assert "根据语法关系和查询意图判断时间表达" in rendered
    assert "用户没有明确提供修饰录音集合的时间约束时，time_range 必须为 null" in rendered
    assert "只有已经确认存在这种明确约束、但仍无法可靠计算边界时" in rendered
    assert "未使用的可选字段必须为 null，不得用 0、空字符串或空对象代替" in rendered
    assert "不得根据 sources 或历史文本推断人物、话题、录音内容" in rendered
    assert "conversation_history 是按时间升序排列的历史对话数组" in rendered
    assert "sources 只绑定到所在 assistant 消息" in rendered
    assert "current_query 可可靠关联到一条或多条 assistant 历史消息" in rendered
    assert "显式补全 current_query 中的指代和省略" in rendered
    assert "这种补全不视为增加新事实" in rendered
    assert "time_range.start 和 time_range.end" in rendered
    assert "左闭右开区间 [start,end)" in rendered
    assert "一周从星期一开始" in rendered
    assert '"properties"' not in rendered
    assert "history_messages" not in rendered


def test_grade_prompt_only_evaluates_the_evidence_it_receives() -> None:
    prompt, values, _parser = grade_prompt("最近一条录音有几个说话人", "结构化说话人标签数量：9")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "结构化字段已经由上游可信流程确定" not in rendered
    assert "最近”或“第 N 条" not in rendered


def test_retrieval_terms_prompt_prepares_a_scope_free_content_query() -> None:
    prompt, values, _parser = retrieval_terms_prompt(
        "最近的录音里，王总说 API v2 的上线时间最后定了吗？"
    )

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "不生成录音范围过滤条件" in rendered
    assert "用于录音正文检索和证据判断的 content_query" in rendered
    assert "判断依据是成分在当前问题中的语义作用" in rendered
    assert "不得按词语表机械删除" in rendered
    assert "改写可能改变问题含义时，必须保留" in rendered
    assert "不得补充同义词" in rendered
    assert "最近的录音里，王总说 API v2 的上线时间最后定了吗？" in rendered


def test_grade_prompt_uses_a_conservative_answerability_process() -> None:
    prompt, values, _parser = grade_prompt("硅光的方案", "录音讨论了硅光集成路径和生产成本。")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "不直接回答用户" in rendered
    assert "决定性约束" in rendered
    assert "最强证据子集" in rendered
    assert "具体、保守且不误导的候选回答" in rendered
    assert "实际行为、活动目的和上下文" in rendered
    assert "名称未逐字出现不足以 abstain" in rendered
    assert "不要做字面匹配或按证据数量投票" in rendered


def test_grade_prompt_uses_a_general_qualified_answer_rule() -> None:
    prompt, values, _parser = grade_prompt(
        "对方是否真的想收购",
        "A 对 B 说：之前经常被你们‘采风’，聊完也没有下文。",
    )

    messages = prompt.invoke(values).to_messages()
    rendered = "\n".join(str(message.content) for message in messages)
    system = str(messages[0].content)

    assert "qualified_answer" in rendered
    assert "只支持部分内容" in rendered
    assert "名称、性质、阶段、范围等仍需限定" in rendered
    assert "无需猜测或保守限定" in rendered
    assert "反讽" not in system
    assert "收购" not in system


def test_grade_prompt_keeps_explicit_recording_answers_direct() -> None:
    prompt, values, _parser = grade_prompt(
        "I2C 的时延规定是多少",
        "I2C 有规定，时延不能大于五微秒。",
    )

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "若证据直接陈述了问题所求的结论" in rendered
    assert "必须选择此项" in rendered
    assert "不得仅因内容来自录音转写而降为 qualified_answer" in rendered


def test_grade_prompt_does_not_decide_answer_planning() -> None:
    prompt, values, _parser = grade_prompt("发布日期是什么", "发布日期是 8 月 1 日。")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "planning_required" not in rendered
    assert "retrieve_more" not in rendered


def test_answer_plan_prompt_only_organizes_grade_approved_evidence() -> None:
    prompt, values, _parser = answer_plan_prompt(
        "硅光现状讲了什么",
        "[1] 硅光正在进入规模应用",
        "direct_answer",
    )

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "证据评估已确认现有证据可以回答问题" in rendered
    assert "不要再次判断证据是否充分" in rendered
    assert "not_enough_evidence" not in rendered
    assert "supported_claims" not in rendered


def test_direct_answer_prompt_does_not_require_or_expose_an_answer_plan() -> None:
    prompt, values = answer_prompt(
        "发布日期是什么",
        None,
        "[1] 发布日期是 8 月 1 日。",
        "direct_answer",
    )

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "回答计划：" not in rendered
    assert "仅依据给出的证据" in rendered
    assert "证据评估" not in rendered
    assert "每个可核实的事实性陈述后必须紧跟一个或多个引用标记" in rendered
    assert "不得创造编号" in rendered


def test_answer_prompt_uses_the_same_user_facing_style_for_all_verdicts() -> None:
    qualified_prompt, qualified_values = answer_prompt(
        "对方是否真的想收购",
        None,
        "A：之前经常被你们采风，聊完也没有下文。",
        "qualified_answer",
    )
    direct_prompt, direct_values = answer_prompt(
        "发布日期是什么",
        None,
        "发布日期是 8 月 1 日。",
        "direct_answer",
    )

    qualified_rendered = "\n".join(str(message.content) for message in qualified_prompt.invoke(qualified_values).to_messages())
    direct_rendered = "\n".join(str(message.content) for message in direct_prompt.invoke(direct_values).to_messages())

    assert "像熟悉录音内容的助手一样" in qualified_rendered
    assert "简单问题直接用一句话或一个短段落回答" in qualified_rendered
    assert "先给结论，再补充对回答有帮助的必要信息" in qualified_rendered
    assert "不要说明答案如何得出" in qualified_rendered
    assert "不要使用“证据内容”“推测/分析”“回答计划”等内部标签" in qualified_rendered
    assert "不得把不确定的解释写成确定事实" not in qualified_rendered
    assert qualified_rendered.replace("对方是否真的想收购", "").replace("A：之前经常被你们采风，聊完也没有下文。", "") == direct_rendered.replace("发布日期是什么", "").replace("发布日期是 8 月 1 日。", "")


def test_answer_prompt_without_assessment_does_not_invent_assessment_constraints() -> None:
    prompt, values = answer_prompt("录音有多长", None, "时长：120 秒")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "证据评估" not in rendered
    assert "supported_claims" not in rendered


def test_answer_prompt_does_not_receive_conversation_history() -> None:
    prompt, values = answer_prompt("I2C 的时延是多少", None, "[1] I2C 的时延不能大于五微秒。")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "近期对话" not in rendered
    assert "history" not in values
