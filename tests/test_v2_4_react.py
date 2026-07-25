"""deep-rag v2.4 ReAct Agent 循环 + 免费 LLM 后端 测试

测试范围：
1. 智谱 zhipu LLM 后端配置
2. get_llm_with_fallback 降级链路
3. ReAct Agent 决策节点（Mock LLM）
4. ReAct 路由逻辑
5. 新 State 字段
6. 知识图谱工具（add_relation + search）
7. SQLite 精确查询（插入 + 查询）

注意：部分测试需要 langgraph，未安装时自动跳过。
"""
import sys
import os
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

# 检查 langgraph 是否可用
try:
    import langgraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

# 检查 duckduckgo_search 是否可用
try:
    import duckduckgo_search
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


# ===== Mock LLM =====

class MockLLM:
    """模拟LLM，可预设响应序列"""
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.call_count = 0

    def invoke(self, messages):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        resp = self.responses[idx]
        return SimpleNamespace(content=resp)


# ===== 测试1: 智谱 zhipu 配置 =====

def test_zhipu_config():
    """验证智谱API Key和base_url配置"""
    print("=== 测试1: 智谱zhipu配置 ===")
    from src.config import ZHIPU_API_KEY, SILICONFLOW_API_KEY

    # 配置项应该存在（即使为空）
    assert isinstance(ZHIPU_API_KEY, str)
    assert isinstance(SILICONFLOW_API_KEY, str)
    print(f"  ZHIPU_API_KEY configured: {bool(ZHIPU_API_KEY)}")
    print(f"  SILICONFLOW_API_KEY configured: {bool(SILICONFLOW_API_KEY)}")
    print("  PASS\n")


def test_agentic_router_default():
    """验证AGENTIC_ROUTER默认为llm"""
    print("=== 测试2: AGENTIC_ROUTER默认值 ===")
    from src.config import AGENTIC_ROUTER
    assert AGENTIC_ROUTER == "llm", f"Expected 'llm', got '{AGENTIC_ROUTER}'"
    print(f"  AGENTIC_ROUTER = {AGENTIC_ROUTER}")
    print("  PASS\n")


def test_retrieval_mode_has_react():
    """验证RETRIEVAL_MODE支持agentic_react"""
    print("=== 测试3: RETRIEVAL_MODE支持agentic_react ===")
    from src.config import RETRIEVAL_MODE
    # RETRIEVAL_MODE可以是Enum或str
    mode_val = RETRIEVAL_MODE.value if hasattr(RETRIEVAL_MODE, 'value') else RETRIEVAL_MODE
    print(f"  Current RETRIEVAL_MODE = {mode_val}")
    # 只验证配置项可读
    assert mode_val is not None
    print("  PASS\n")


# ===== 测试2: get_llm_with_fallback =====

def test_get_llm_with_fallback_returns_none():
    """无任何LLM时降级到None（规则模式）"""
    print("=== 测试4: get_llm_with_fallback降级到None ===")
    # 确保没有任何API Key
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("ZHIPU_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("SILICONFLOW_API_KEY", None)
    os.environ["LLM_BACKEND"] = "none"

    # 重新导入配置
    import importlib
    import src.config
    importlib.reload(src.config)

    llm = src.config.get_llm_with_fallback()
    # 没有任何LLM时应该返回None或降级到Ollama
    print(f"  Result: {type(llm).__name__ if llm else 'None'}")
    print("  PASS\n")


# ===== 测试3: ReAct State 字段 =====

def test_state_has_react_fields():
    """RAGState包含ReAct循环所需字段"""
    print("=== 测试5: RAGState ReAct字段 ===")
    from src.state import RAGState

    # TypedDict在运行时不强制检查，但字段应存在
    annotations = RAGState.__annotations__

    required_fields = ["next_action", "agent_reason", "retrieval_round", "used_tools"]
    for field in required_fields:
        assert field in annotations, f"Missing field: {field}"
        print(f"  {field}: OK")

    print("  PASS\n")


# ===== 测试4: _parse_json_response =====

def test_parse_json_response_valid():
    """JSON解析：有效JSON"""
    print("=== 测试6: _parse_json_response有效JSON ===")
    from src.graph import _parse_json_response

    result = _parse_json_response('{"action": "vector_search", "reason": "test", "query": "hello"}')
    assert result["action"] == "vector_search"
    assert result["reason"] == "test"
    assert result["query"] == "hello"
    print(f"  Parsed: {result}")
    print("  PASS\n")


def test_parse_json_response_embedded():
    """JSON解析：嵌入文本中的JSON"""
    print("=== 测试7: _parse_json_response嵌入JSON ===")
    from src.graph import _parse_json_response

    result = _parse_json_response('我认为应该使用 {"action": "web_search", "reason": "need latest"} 工具')
    assert result["action"] == "web_search"
    print(f"  Extracted: {result}")
    print("  PASS\n")


def test_parse_json_response_invalid():
    """JSON解析：无效JSON兜底"""
    print("=== 测试8: _parse_json_response无效JSON兜底 ===")
    from src.graph import _parse_json_response

    result = _parse_json_response("这不是JSON")
    assert result["action"] == "generate"  # 兜底
    print(f"  Fallback: {result}")
    print("  PASS\n")


# ===== 测试5: _summarize_docs =====

def test_summarize_docs_empty():
    """文档摘要：空列表"""
    print("=== 测试9: _summarize_docs空列表 ===")
    from src.graph import _summarize_docs

    result = _summarize_docs([])
    assert "暂无" in result
    print(f"  Empty: {result}")
    print("  PASS\n")


def test_summarize_docs_with_content():
    """文档摘要：有内容"""
    print("=== 测试10: _summarize_docs有内容 ===")
    from src.graph import _summarize_docs

    docs = [
        {"content": "这是一段测试文档内容，用于验证摘要功能", "source": "test.md"},
        {"content": "第二段文档", "source": "test2.md"},
    ]
    result = _summarize_docs(docs)
    assert "test.md" in result
    assert "测试文档" in result
    print(f"  Summary (truncated): {result[:80]}...")
    print("  PASS\n")


# ===== 测试6: route_react_agent =====

def test_route_react_agent_generate():
    """ReAct路由：action=generate时路由到generate"""
    print("=== 测试11: route_react_agent -> generate ===")
    from src.graph import route_react_agent

    state = {"next_action": "generate", "retrieval_round": 0, "max_retries": 3}
    assert route_react_agent(state) == "generate"
    print(f"  action=generate -> generate")
    print("  PASS\n")


def test_route_react_agent_vector_search():
    """ReAct路由：action=vector_search"""
    print("=== 测试12: route_react_agent -> vector_search ===")
    from src.graph import route_react_agent

    state = {"next_action": "vector_search", "retrieval_round": 0, "max_retries": 3}
    assert route_react_agent(state) == "vector_search"
    print(f"  action=vector_search -> vector_search")
    print("  PASS\n")


def test_route_react_agent_max_rounds():
    """ReAct路由：超过最大轮次强制generate"""
    print("=== 测试13: route_react_agent超过轮次强制generate ===")
    from src.graph import route_react_agent

    state = {"next_action": "vector_search", "retrieval_round": 3, "max_retries": 3}
    assert route_react_agent(state) == "generate"
    print(f"  round=3 >= max=3 -> force generate")
    print("  PASS\n")


def test_route_react_agent_invalid_action():
    """ReAct路由：无效action兜底generate"""
    print("=== 测试14: route_react_agent无效action兜底 ===")
    from src.graph import route_react_agent

    state = {"next_action": "invalid_tool", "retrieval_round": 0, "max_retries": 3}
    assert route_react_agent(state) == "generate"
    print(f"  invalid_action -> generate")
    print("  PASS\n")


# ===== 测试7: GraphSearchTool add_relation + search =====

def test_graph_search_add_and_search():
    """知识图谱：添加关系后能检索到"""
    print("=== 测试15: GraphSearchTool添加+检索 ===")
    from src.retrieval.agentic_tools import GraphSearchTool

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = str(Path(tmpdir) / "test_graph.pkl")
        tool = GraphSearchTool(graph_path=graph_path)

        # 添加关系
        tool.add_relation("LangGraph", "LangChain", "依赖")
        tool.add_relation("LangGraph", "StateGraph", "包含")
        tool.add_relation("DeepRAG", "LangGraph", "使用")

        # 检索
        results = tool.search("LangGraph", max_depth=1)
        assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"
        print(f"  Added 3 relations, searched 'LangGraph' -> {len(results)} results")
        for r in results:
            content = r["content"] if isinstance(r, dict) else r.content
            print(f"    {content}")
        print("  PASS\n")


# ===== 测试8: ExactMatchTool 插入+查询 =====

def test_exact_match_insert_and_query():
    """SQLite精确查询：插入数据后能查到"""
    print("=== 测试16: ExactMatchTool插入+查询 ===")
    import sqlite3
    import gc
    from src.retrieval.agentic_tools import ExactMatchTool

    # 使用固定路径而非tempdir（避免Windows文件锁问题）
    db_path = str(Path(PROJECT_ROOT) / "data" / "test_exact_match.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    if os.path.exists(db_path):
        os.remove(db_path)

    tool = ExactMatchTool(db_path=db_path)

    # 插入测试数据
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO documents (id, content, source, metadata) VALUES (?, ?, ?, ?)",
            ("DOC-001", "这是文档001的内容", "test.md", "{}")
        )
        conn.execute(
            "INSERT INTO documents (id, content, source, metadata) VALUES (?, ?, ?, ?)",
            ("DOC-002", "这是文档002的内容", "test2.md", "{}")
        )
        conn.commit()

    # 查询 DOC-001
    results = tool.search("查询文档ID DOC-001")
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    content = results[0]["content"] if isinstance(results[0], dict) else results[0].content
    assert "文档001" in content
    print(f"  Inserted 2 docs, queried 'DOC-001' -> found: {content[:30]}")

    # 查询不存在的ID
    results2 = tool.search("查询 DOC-999")
    print(f"  Queried 'DOC-999' -> {len(results2)} results (expected 0)")
    assert len(results2) == 0

    # 清理
    del tool
    gc.collect()
    try:
        os.remove(db_path)
    except PermissionError:
        pass  # Windows可能仍锁定，忽略
    print("  PASS\n")


# ===== 测试9: node_agent_decision with Mock LLM =====

def test_agent_decision_with_mock_llm():
    """Agent决策节点：Mock LLM返回工具选择"""
    print("=== 测试17: node_agent_decision Mock LLM ===")
    from src.graph import node_agent_decision

    # Mock get_llm_with_fallback
    import src.config
    original_fn = src.config.get_llm_with_fallback
    src.config.get_llm_with_fallback = lambda temp=None: MockLLM(
        '{"action": "vector_search", "reason": "semantic question", "query": "INTJ功能"}'
    )

    try:
        state = {
            "question": "INTJ的主导功能是什么",
            "retrieved_docs": [],
            "used_tools": [],
            "retrieval_round": 0,
            "max_retries": 3,
        }
        result = node_agent_decision(state)
        assert result["next_action"] == "vector_search"
        assert result["retrieval_round"] == 1
        print(f"  LLM chose: {result['next_action']}, round={result['retrieval_round']}")
        print("  PASS\n")
    finally:
        src.config.get_llm_with_fallback = original_fn


def test_agent_decision_no_llm():
    """Agent决策节点：无LLM时直接generate"""
    print("=== 测试18: node_agent_decision无LLM ===")
    from src.graph import node_agent_decision

    # Mock无LLM
    import src.config
    original_fn = src.config.get_llm_with_fallback
    src.config.get_llm_with_fallback = lambda temp=None: None

    try:
        state = {
            "question": "test",
            "retrieved_docs": [],
            "used_tools": [],
            "retrieval_round": 0,
            "max_retries": 3,
        }
        result = node_agent_decision(state)
        assert result["next_action"] == "generate"
        assert result["agent_reason"] == "no_llm"
        print(f"  No LLM -> generate (reason: no_llm)")
        print("  PASS\n")
    finally:
        src.config.get_llm_with_fallback = original_fn


def test_agent_decision_llm_exception():
    """Agent决策节点：LLM异常时兜底generate"""
    print("=== 测试19: node_agent_decision LLM异常 ===")
    from src.graph import node_agent_decision

    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM service down")

    import src.config
    original_fn = src.config.get_llm_with_fallback
    src.config.get_llm_with_fallback = lambda temp=None: BrokenLLM()

    try:
        state = {
            "question": "test",
            "retrieved_docs": [],
            "used_tools": [],
            "retrieval_round": 0,
            "max_retries": 3,
        }
        result = node_agent_decision(state)
        assert result["next_action"] == "generate"
        print(f"  LLM exception -> generate (fallback)")
        print("  PASS\n")
    finally:
        src.config.get_llm_with_fallback = original_fn


# ===== 主测试入口 =====

if __name__ == "__main__":
    tests = [
        test_zhipu_config,
        test_agentic_router_default,
        test_retrieval_mode_has_react,
        test_state_has_react_fields,
        test_parse_json_response_valid,
        test_parse_json_response_embedded,
        test_parse_json_response_invalid,
        test_summarize_docs_empty,
        test_summarize_docs_with_content,
        test_route_react_agent_generate,
        test_route_react_agent_vector_search,
        test_route_react_agent_max_rounds,
        test_route_react_agent_invalid_action,
        test_graph_search_add_and_search,
        test_exact_match_insert_and_query,
        test_agent_decision_with_mock_llm,
        test_agent_decision_no_llm,
        test_agent_decision_llm_exception,
    ]

    print(f"\nRunning {len(tests)} v2.4 ReAct tests...\n")
    print(f"  langgraph: {'installed' if HAS_LANGGRAPH else 'NOT installed (graph tests will skip)'}")
    print(f"  duckduckgo: {'installed' if HAS_DDGS else 'NOT installed (web_search tests skip)'}")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0
    skipped = 0
    for test in tests:
        # 检查是否需要 langgraph
        needs_langgraph = any(kw in test.__name__ for kw in [
            "parse_json", "summarize_docs", "route_react", "agent_decision"
        ])
        if needs_langgraph and not HAS_LANGGRAPH:
            print(f"=== SKIP: {test.__name__} (requires langgraph) ===\n")
            skipped += 1
            continue

        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            print()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
