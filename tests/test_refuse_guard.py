"""拒识 / 不可答 规则与兜底（不依赖向量库）"""


def test_is_unanswerable_patterns():
    from src.agents.query_analyzer import is_unanswerable_query, needs_rag, make_refuse_answer

    bad, why = is_unanswerable_query("INTJ昨天中午在火星吃了什么？")
    assert bad
    assert "火星" in why or "不可答" in why or "模式" in why

    bad2, _ = is_unanswerable_query("请给出不存在的公司DeepRAG宇宙总部地址门牌号")
    assert bad2

    bad3, _ = is_unanswerable_query("本知识库里有没有介绍量子纠缠实验设备型号？")
    assert bad3

    ok, _ = is_unanswerable_query("INTJ的主导功能是什么？")
    assert not ok

    needed, reason = needs_rag("INTJ昨天中午在火星吃了什么？")
    assert needed is False
    assert reason.startswith("不可答")

    ans = make_refuse_answer("测试")
    assert "无法" in ans or "未找到" in ans
    assert "【直接回答】" in ans


def test_ensure_answer_or_refuse_empty():
    from src.graph import _ensure_answer_or_refuse

    out = _ensure_answer_or_refuse(
        "随便问点什么",
        {"answer": "", "relevant_count": 0, "history": []},
    )
    assert out.get("no_knowledge") is True
    assert out.get("answer")
    assert "无法" in out["answer"] or "未找到" in out["answer"]


def test_ensure_answer_or_refuse_force_unanswerable():
    from src.graph import _ensure_answer_or_refuse

    out = _ensure_answer_or_refuse(
        "INTJ昨天中午在火星吃了什么？",
        {
            "answer": "【直接回答】他吃了土豆泥。",
            "relevant_count": 3,
            "history": [],
            "no_knowledge": False,
        },
    )
    assert out.get("no_knowledge") is True
    assert "土豆泥" not in out.get("answer", "")
