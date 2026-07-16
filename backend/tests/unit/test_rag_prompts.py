from rag.prompts import answer_plan_prompt, answer_prompt, grade_prompt, route_prompt


def test_route_prompt_includes_json_schema_and_exclusive_json_instruction() -> None:
    prompt, values, _parser = route_prompt("最近的录音讲了什么", "[]", "[]")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "只输出唯一一个符合 schema 的 JSON 对象" in rendered
    assert '"status"' in rendered
    assert '"strategy"' in rendered


def test_grade_prompt_treats_scope_and_speaker_facts_as_trusted_metadata() -> None:
    prompt, values, _parser = grade_prompt("最近一条录音有几个说话人", "结构化说话人标签数量：9")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "查询范围已经由上游路由和筛选流程确定" in rendered
    assert "最近”或“第 N 条" not in rendered
    assert "每个不同标签代表一个说话人聚类" in rendered


def test_grade_prompt_checks_answerability_without_demanding_topic_completeness() -> None:
    prompt, values, _parser = grade_prompt("硅光的方案", "录音讨论了硅光集成路径和生产成本。")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "不负责评价用户的问题" in rendered
    assert "不要求 evidence 全面、系统" in rendered
    assert "不得擅自增加用户未提出的关注维度" in rendered
    assert "只要 evidence 包含与 topic 直接相关的实质信息" in rendered


def test_grade_prompt_only_requires_plan_for_structurally_complex_answers() -> None:
    prompt, values, _parser = grade_prompt("发布日期是什么", "发布日期是 8 月 1 日。")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "简单事实、单一结论、单一观点和简单局部总结令 planning_required=false" in rendered
    assert "多子问题、对象比较、时间线、按人物或主题分组、跨录音综合" in rendered
    assert "不要仅因为 evidence 数量大于一就要求 plan" in rendered


def test_answer_plan_prompt_only_organizes_grade_approved_evidence() -> None:
    prompt, values, _parser = answer_plan_prompt("硅光现状讲了什么", "[1] 硅光正在进入规模应用")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "Grade 已确认现有证据足以回答问题" in rendered
    assert "不要再次判断证据是否充分" in rendered
    assert "not_enough_evidence" not in rendered


def test_direct_answer_prompt_does_not_require_or_expose_an_answer_plan() -> None:
    prompt, values = answer_prompt("发布日期是什么", None, "[1] 发布日期是 8 月 1 日。", "")

    rendered = "\n".join(str(message.content) for message in prompt.invoke(values).to_messages())

    assert "回答计划" not in rendered
    assert "仅依据给出的证据" in rendered
