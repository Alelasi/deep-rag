"""Agentic RAG 集成测试 - 验证 ENABLE_AGENTIC_RAG=true 时主 Pipeline 正常工作"""
import os
import pytest
from src.graph import query, get_indexer


@pytest.fixture(scope="module")
def setup_agentic_mode():
    """临时启用 Agentic RAG 模式"""
    original = os.environ.get("ENABLE_AGENTIC_RAG")
    os.environ["ENABLE_AGENTIC_RAG"] = "true"
    yield
    if original is None:
        os.environ.pop("ENABLE_AGENTIC_RAG", None)
    else:
        os.environ["ENABLE_AGENTIC_RAG"] = original


def test_agentic_rag_pipeline(setup_agentic_mode):
    """测试 Agentic RAG 完整 Pipeline"""
    # 准备测试数据
    indexer = get_indexer("test_agentic")
    indexer.index_texts([
        {"content": "LangGraph 是 LangChain 的状态机编排框架", "metadata": {"source": "doc1"}},
        {"content": "Agentic RAG 通过 Router 动态选择检索工具", "metadata": {"source": "doc2"}},
    ])

    # 执行查询
    result = query("什么是 Agentic RAG", collection_name="test_agentic", max_retries=1)

    # 验证
    assert result["answer"], "应该生成答案"
    assert len(result["retrieved_docs"]) > 0, "应该检索到文档"
    assert result["current_step"] == "completed", "应该完成全流程"
    # 验证 history 中包含 Agentic 标记
    history_str = " ".join(result.get("history", []))
    assert "via" in history_str or "Retrieved" in history_str, "history 应记录检索方式"


def test_agentic_router_decision(setup_agentic_mode):
    """测试 Router 决策逻辑（规则路由）"""
    from src.retrieval.agent_router import RuleBasedRouter

    router = RuleBasedRouter()

    # 测试精确匹配触发
    assert router.route("查询用户ID 12345") == "exact_match"
    assert router.route("订单号 ABC123") == "exact_match"

    # 测试关系查询触发
    assert router.route("A 和 B 之间的关系") == "graph_search"
    assert router.route("依赖关系") == "graph_search"

    # 测试时效查询触发
    assert router.route("2026 年最新特性") == "web_search"
    assert router.route("最新的文档") == "web_search"

    # 测试默认路由
    assert router.route("如何使用 LangGraph") == "vector_search"


def test_agentic_vs_hybrid_consistency(setup_agentic_mode):
    """对比 Agentic 和 Hybrid 模式的结果一致性"""
    # 准备数据
    indexer = get_indexer("test_consistency")
    indexer.index_texts([
        {"content": "Python 是一门编程语言", "metadata": {"source": "doc1"}},
    ])

    # Agentic 模式
    os.environ["ENABLE_AGENTIC_RAG"] = "true"
    result_agentic = query("什么是 Python", collection_name="test_consistency", max_retries=0)

    # Hybrid 模式
    os.environ["ENABLE_AGENTIC_RAG"] = "false"
    result_hybrid = query("什么是 Python", collection_name="test_consistency", max_retries=0)

    # 验证两种模式都能检索到文档
    assert len(result_agentic["retrieved_docs"]) > 0
    assert len(result_hybrid["retrieved_docs"]) > 0
    # 验证都能生成答案
    assert result_agentic["answer"]
    assert result_hybrid["answer"]
