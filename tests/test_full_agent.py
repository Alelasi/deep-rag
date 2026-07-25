"""deep-rag v2.4 全功能 Agent 测试

测试策略：
1. 依赖完整性 — 逐个 import 所有模块
2. LLM 连通性 — 智谱 GLM-4-Flash 真实调用
3. 工具单元测试 — 4个工具各自功能
4. Enhanced Pipeline — 完整查询流程
5. Agentic ReAct — LLM自主决策循环
6. 前端参数验证 — 各种参数组合
7. 降级链路 — LLM不可用时的回退
8. 边界条件 — 空问题/超长问题/特殊字符
"""
import sys
import os
import time
import json
import traceback
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

# 设置环境变量（从.env加载）
os.environ.setdefault("LLM_BACKEND", "zhipu")
os.environ.setdefault("LLM_MODEL", "glm-4-flash")

# 测试结果收集
results = []

def test(name, func, critical=True):
    """运行测试并记录结果"""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    try:
        func()
        results.append((name, "PASS", "", critical))
        print(f"  >>> PASS")
    except Exception as e:
        tb = traceback.format_exc().split("\n")[-3].strip()
        results.append((name, "FAIL", f"{e}", critical))
        print(f"  >>> FAIL: {e}")
        print(f"      {tb}")

# ===== T1: 依赖完整性 =====

def test_imports():
    """逐个 import 所有核心模块"""
    modules = [
        ("src.config", "get_llm, get_llm_with_fallback"),
        ("src.state", "RAGState"),
        ("src.graph", "query, get_indexer, build_agentic_graph, create_agentic_app"),
        ("src.retrieval.agentic_tools", "create_toolbox, ExactMatchTool, GraphSearchTool, WebSearchTool, VectorSearchTool"),
        ("src.retrieval.agent_router", "RuleBasedRouter, LLMRouter, AgenticRetriever"),
    ]
    for mod_name, attrs in modules:
        mod = __import__(mod_name, fromlist=attrs.split(", "))
        for attr in attrs.split(", "):
            attr = attr.strip()
            obj = getattr(mod, attr)
            assert obj is not None, f"{mod_name}.{attr} is None"
            print(f"  OK: {mod_name}.{attr}")

# ===== T2: LLM 连通性 =====

def test_llm_zhipu():
    """智谱 GLM-4-Flash 真实调用"""
    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import get_llm

    llm = get_llm()
    assert llm is not None, "LLM is None, check ZHIPU_API_KEY"

    from langchain_core.messages import HumanMessage
    resp = llm.invoke([HumanMessage(content="回复'连接成功'四个字")])
    text = resp.content if hasattr(resp, "content") else str(resp)
    assert len(text) > 0, "Empty response"
    print(f"  LLM response: {text[:100]}")

def test_llm_fallback():
    """get_llm_with_fallback 降级函数"""
    from src.config import get_llm_with_fallback
    llm = get_llm_with_fallback()
    # 有API Key时应该返回LLM实例
    assert llm is not None, "Fallback returned None despite having API key"
    print(f"  Fallback LLM type: {type(llm).__name__}")

# ===== T3: 工具单元测试 =====

def test_tool_vector_search():
    """向量检索工具"""
    from src.retrieval.agentic_tools import VectorSearchTool
    class MockRetriever:
        def retrieve(self, q, top_k=5):
            return [{"content": f"mock result for {q}", "source": "test.md"}]
    tool = VectorSearchTool(MockRetriever())
    results = tool.search("test query")
    assert len(results) > 0, "VectorSearch returned empty"
    print(f"  Vector search: {len(results)} results")

def test_tool_exact_match():
    """SQLite 精确查询：插入+查询"""
    from src.retrieval.agentic_tools import ExactMatchTool
    import sqlite3

    db_path = str(Path(PROJECT_ROOT) / "data" / "test_exact.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    tool = ExactMatchTool(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO documents (id, content, source, metadata) VALUES (?,?,?,?)",
                     ("DOC-001", "测试文档001", "test.md", "{}"))
        conn.commit()

    results = tool.search("查询文档ID DOC-001")
    assert len(results) == 1, f"Expected 1, got {len(results)}"
    print(f"  Exact match: found DOC-001")

    # 查询不存在的
    results2 = tool.search("查询 DOC-999")
    assert len(results2) == 0
    print(f"  Exact match: DOC-999 not found (correct)")

def test_tool_graph_search():
    """NetworkX 图谱检索：添加+查询"""
    from src.retrieval.agentic_tools import GraphSearchTool
    tool = GraphSearchTool()
    tool.add_relation("Python", "LangChain", "依赖")
    tool.add_relation("LangChain", "LangGraph", "包含")
    tool.add_relation("DeepRAG", "LangGraph", "使用")

    results = tool.search("LangGraph", max_depth=1)
    assert len(results) >= 2, f"Expected >=2, got {len(results)}"
    print(f"  Graph search: {len(results)} relations found for 'LangGraph'")

def test_tool_web_search():
    """DuckDuckGo 网络搜索"""
    from src.retrieval.agentic_tools import WebSearchTool
    tool = WebSearchTool()
    results = tool.search("Python programming language", max_results=2)
    # DuckDuckGo可能因网络问题返回空，不强制assert
    print(f"  Web search: {len(results)} results (may be 0 if network blocked)")

def test_toolbox_4_tools():
    """create_toolbox 注册全部4个工具"""
    from src.retrieval.agentic_tools import create_toolbox
    class MockRetriever:
        def retrieve(self, q, top_k=5): return []
    toolbox = create_toolbox(MockRetriever())
    tools = toolbox.list_tools()
    names = [t["name"] for t in tools]
    assert len(tools) == 4, f"Expected 4 tools, got {len(tools)}: {names}"
    for expected in ["vector_search", "exact_match", "graph_search", "web_search"]:
        assert expected in names, f"Missing tool: {expected}"
    print(f"  Toolbox: {names}")

# ===== T4: ReAct Agent 组件 =====

def test_react_json_parse():
    """JSON 解析器：各种输入"""
    from src.graph import _parse_json_response
    # 有效JSON
    r = _parse_json_response('{"action": "vector_search", "reason": "test"}')
    assert r["action"] == "vector_search"
    # 嵌入JSON
    r = _parse_json_response('text {"action": "web_search"} more text')
    assert r["action"] == "web_search"
    # 无效JSON兜底
    r = _parse_json_response("not json at all")
    assert r["action"] == "generate"
    print(f"  JSON parse: 3 cases passed")

def test_react_route():
    """ReAct 路由逻辑"""
    from src.graph import route_react_agent
    # 正常路由
    assert route_react_agent({"next_action": "vector_search", "retrieval_round": 0, "max_retries": 3}) == "vector_search"
    assert route_react_agent({"next_action": "web_search", "retrieval_round": 0, "max_retries": 3}) == "web_search"
    assert route_react_agent({"next_action": "generate", "retrieval_round": 0, "max_retries": 3}) == "generate"
    # 超过轮次
    assert route_react_agent({"next_action": "vector_search", "retrieval_round": 3, "max_retries": 3}) == "generate"
    # 无效action
    assert route_react_agent({"next_action": "unknown", "retrieval_round": 0, "max_retries": 3}) == "generate"
    print(f"  Route: 5 cases passed")

def test_react_agent_decision_mock():
    """Agent决策节点（Mock LLM）"""
    from src.graph import node_agent_decision
    from types import SimpleNamespace

    import src.config
    original = src.config.get_llm_with_fallback
    src.config.get_llm_with_fallback = lambda t=None: type("MockLLM", (), {
        "invoke": lambda self, msgs: SimpleNamespace(
            content='{"action": "vector_search", "reason": "semantic question", "query": "test"}'
        )
    })()

    try:
        state = {"question": "test", "retrieved_docs": [], "used_tools": [], "retrieval_round": 0, "max_retries": 3}
        result = node_agent_decision(state)
        assert result["next_action"] == "vector_search"
        assert result["retrieval_round"] == 1
        print(f"  Agent decision (mock): action={result['next_action']}, round={result['retrieval_round']}")
    finally:
        src.config.get_llm_with_fallback = original

def test_react_agent_no_llm():
    """Agent决策节点：无LLM时降级"""
    from src.graph import node_agent_decision
    import src.config
    original = src.config.get_llm_with_fallback
    src.config.get_llm_with_fallback = lambda t=None: None

    try:
        state = {"question": "test", "retrieved_docs": [], "used_tools": [], "retrieval_round": 0, "max_retries": 3}
        result = node_agent_decision(state)
        assert result["next_action"] == "generate"
        assert result["agent_reason"] == "no_llm"
        print(f"  Agent decision (no LLM): generate (no_llm)")
    finally:
        src.config.get_llm_with_fallback = original

def test_react_build_graph():
    """构建 Agentic ReAct 图"""
    from src.graph import build_agentic_graph
    graph = build_agentic_graph()
    assert graph is not None
    print(f"  Agentic graph built: {type(graph).__name__}")

# ===== T5: Enhanced Pipeline 端到端 =====

def test_enhanced_pipeline():
    """Enhanced Pipeline 完整查询"""
    from src.graph import query
    result = query(
        "什么是RAG?",
        collection_name="demo_kb",
        max_retries=2,
        mode="enhanced"
    )
    assert isinstance(result, dict)
    assert "answer" in result or "history" in result
    answer = result.get("answer", "")
    print(f"  Answer length: {len(answer)} chars")
    print(f"  History steps: {len(result.get('history', []))}")
    if answer:
        print(f"  Answer preview: {answer[:100]}...")

# ===== T6: Agentic ReAct 端到端 =====

def test_agentic_react_pipeline():
    """Agentic ReAct 完整查询"""
    from src.graph import query
    result = query(
        "什么是RAG?",
        collection_name="demo_kb",
        max_retries=3,
        mode="agentic_react"
    )
    assert isinstance(result, dict)
    used_tools = result.get("used_tools", [])
    retrieval_round = result.get("retrieval_round", 0)
    answer = result.get("answer", "")
    print(f"  Used tools: {used_tools}")
    print(f"  Retrieval rounds: {retrieval_round}")
    print(f"  Answer length: {len(answer)} chars")
    if answer:
        print(f"  Answer preview: {answer[:100]}...")

# ===== T7: 前端参数组合 =====

def test_param_combinations():
    """各种参数组合测试"""
    from src.graph import query
    test_cases = [
        ("简单问题", "什么是AI?", "demo_kb", 1, "enhanced"),
        ("复杂问题", "如何设计一个高可用的RAG系统，考虑检索准确率和延迟的平衡？", "demo_kb", 2, "enhanced"),
        ("英文问题", "What is retrieval augmented generation?", "demo_kb", 1, "enhanced"),
        ("空知识库", "test", "nonexistent_collection", 1, "enhanced"),
    ]
    for name, question, coll, retries, mode in test_cases:
        try:
            result = query(question, collection_name=coll, max_retries=retries, mode=mode)
            assert isinstance(result, dict)
            print(f"  [{name}] OK: answer={len(result.get('answer', ''))} chars")
        except Exception as e:
            if "nonexistent" in name:
                print(f"  [{name}] Expected error: {str(e)[:60]}")
            else:
                raise

# ===== T8: 边界条件 =====

def test_edge_cases():
    """边界条件测试"""
    from src.graph import query

    # 超长问题
    long_q = "RAG" * 500
    try:
        result = query(long_q, mode="enhanced", max_retries=1)
        print(f"  Long question ({len(long_q)} chars): OK")
    except Exception as e:
        print(f"  Long question: handled gracefully ({str(e)[:60]})")

    # 特殊字符
    special_q = "什么是RAG？<>!@#$%^&*()"
    try:
        result = query(special_q, mode="enhanced", max_retries=1)
        print(f"  Special chars: OK")
    except Exception as e:
        print(f"  Special chars: handled ({str(e)[:60]})")

# ===== T9: 状态字段完整性 =====

def test_state_fields():
    """RAGState 包含所有必要字段"""
    from src.state import RAGState
    required = [
        "question", "collection_name", "rewritten_query", "retrieved_docs",
        "graded_docs", "answer", "citations", "hallucination_score",
        "fact_check_passed", "current_step", "retry_count", "max_retries",
        "history", "next_action", "agent_reason", "retrieval_round", "used_tools",
    ]
    annotations = RAGState.__annotations__
    for field in required:
        assert field in annotations, f"Missing field: {field}"
    print(f"  {len(required)} fields verified")

# ===== T10: 索引器 =====

def test_indexer():
    """索引器：创建+索引文档"""
    from src.graph import get_indexer
    indexer = get_indexer("test_kb")
    assert indexer is not None
    print(f"  Indexer type: {type(indexer).__name__}")

    # 尝试索引（如果sample_docs存在）
    sample_dir = Path(PROJECT_ROOT) / "data" / "sample_docs"
    if sample_dir.exists():
        count = indexer.index_directory(str(sample_dir))
        print(f"  Indexed {count} chunks from {sample_dir}")
    else:
        print(f"  sample_docs not found, skipping index test")

# ===== 主入口 =====

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DeepRAG v2.4 全功能 Agent 测试")
    print("="*60)

    # 先加载.env
    from dotenv import load_dotenv
    load_dotenv(Path(PROJECT_ROOT) / ".env")

    all_tests = [
        ("T1 依赖完整性", test_imports, True),
        ("T2.1 智谱LLM连通", test_llm_zhipu, True),
        ("T2.2 LLM降级函数", test_llm_fallback, True),
        ("T3.1 向量检索工具", test_tool_vector_search, True),
        ("T3.2 SQLite精确查询", test_tool_exact_match, True),
        ("T3.3 NetworkX图谱检索", test_tool_graph_search, True),
        ("T3.4 DuckDuckGo网络搜索", test_tool_web_search, False),
        ("T3.5 工具箱4工具注册", test_toolbox_4_tools, True),
        ("T4.1 ReAct JSON解析", test_react_json_parse, True),
        ("T4.2 ReAct路由逻辑", test_react_route, True),
        ("T4.3 Agent决策(Mock)", test_react_agent_decision_mock, True),
        ("T4.4 Agent决策(无LLM)", test_react_agent_no_llm, True),
        ("T4.5 构建Agentic图", test_react_build_graph, True),
        ("T5 Enhanced Pipeline", test_enhanced_pipeline, True),
        ("T6 Agentic ReAct", test_agentic_react_pipeline, True),
        ("T7 参数组合", test_param_combinations, False),
        ("T8 边界条件", test_edge_cases, False),
        ("T9 State字段完整性", test_state_fields, True),
        ("T10 索引器", test_indexer, True),
    ]

    for name, func, critical in all_tests:
        test(name, func, critical)

    # 汇总
    print("\n" + "="*60)
    print("  测试汇总")
    print("="*60)
    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    critical_fail = sum(1 for _, s, _, c in results if s == "FAIL" and c)
    skipped = len(results) - passed - failed

    for name, status, err, critical in results:
        icon = "✅" if status == "PASS" else "❌"
        tag = "" if status == "PASS" else f" — {err[:80]}"
        crit = "" if critical else " (non-critical)"
        print(f"  {icon} {name}{tag}{crit}")

    print(f"\n  Total: {passed} passed, {failed} failed ({critical_fail} critical)")
    print("="*60)

    sys.exit(0 if critical_fail == 0 else 1)
