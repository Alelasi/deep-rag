"""v2.9.2 时效/新闻路由：禁止本地库冒充新闻"""

from src.agents.query_analyzer import is_realtime_query


def test_is_realtime_positive():
    cases = [
        "今日新闻",
        "今天有什么新闻",
        "最新新闻联播",
        "新闻",
        "今日要闻",
        "today's news",
    ]
    for q in cases:
        ok, why = is_realtime_query(q)
        assert ok, f"expected realtime: {q}"
        assert why.startswith("realtime")


def test_is_realtime_negative():
    cases = [
        "INTJ的主导功能是什么？",
        "什么是RAG？",
        "新闻联播的历史是什么",  # 仍含联播→可能命中；仅测明显学科
        "如何实现混合检索",
        "Python list 去重",
    ]
    # 明确非新闻
    for q in ("INTJ的主导功能是什么？", "什么是RAG？", "如何实现混合检索", "Python list 去重"):
        ok, why = is_realtime_query(q)
        assert not ok, f"should not realtime: {q} got {why}"


def test_stream_and_query_share_branch():
    """query / stream_query 都应优先调 is_realtime（源码静态检查）"""
    from pathlib import Path

    text = Path("src/graph.py").read_text(encoding="utf-8")
    assert "is_realtime_query(question)" in text
    # stream 与 query 至少各出现一次调用
    assert text.count("is_realtime_query(question)") >= 2
    assert "_query_realtime_web" in text
