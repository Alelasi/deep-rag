"""Agent决策路由器单元测试"""
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.agent_router import (
    BaseRouter,
    RuleBasedRouter,
    LLMRouter,
    AgenticRetriever,
)
from src.retrieval.agentic_tools import (
    AgenticRAGToolbox,
    VectorSearchTool,
    ExactMatchTool,
    WebSearchTool,
    GraphSearchTool,
)
from src.state import Document


# ===== Mock组件 =====

class MockRetriever:
    def __init__(self, results=None):
        self.results = results or []

    def retrieve(self, query, top_k=5):
        return self.results


class MockDB:
    def query(self, filters):
        return []


class MockLLM:
    """模拟LLM，可预设响应"""
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.call_count = 0

    def invoke(self, prompt):
        from types import SimpleNamespace
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return SimpleNamespace(content=self.responses[idx])


def make_doc(doc_id, content="content"):
    return Document(doc_id=doc_id, content=content, source="t.md",
                    page=1, metadata={})


def make_test_toolbox():
    """创建包含4种工具的测试工具箱（v2.4全部真实实现）"""
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search",
                          VectorSearchTool(MockRetriever([make_doc("v1")])))
    toolbox.register_tool("exact_match", ExactMatchTool())   # v2.4: SQLite
    toolbox.register_tool("graph_search", GraphSearchTool())  # v2.4: NetworkX
    toolbox.register_tool("web_search", WebSearchTool())      # v2.4: DuckDuckGo
    return toolbox


# ===== 测试：RuleBasedRouter =====

def test_rule_router_default_to_vector():
    """RuleBasedRouter: 普通问题路由到vector_search"""
    print("=== 测试1: 默认路由到vector_search ===")
    router = RuleBasedRouter()
    assert router.route("INTJ的主导功能是什么") == "vector_search"
    assert router.route("解释一下RAG系统") == "vector_search"
    print(f"  Default questions -> vector_search")
    print("  PASS\n")


def test_rule_router_exact_match():
    """RuleBasedRouter: 精确ID查询路由到exact_match"""
    print("=== 测试2: 精确ID -> exact_match ===")
    router = RuleBasedRouter()
    cases = [
        "查询用户ID 12345的信息",
        "找订单号 ORD-789",
        "版本v3.5的发布说明",
        "ID为A001的记录",
        "编号:NX-100",
    ]
    for q in cases:
        chosen = router.route(q)
        assert chosen == "exact_match", f"'{q}' -> got {chosen}"
        print(f"  '{q[:30]}' -> {chosen}")
    print("  PASS\n")


def test_rule_router_graph():
    """RuleBasedRouter: 关系查询路由到graph_search"""
    print("=== 测试3: 关系查询 -> graph_search ===")
    router = RuleBasedRouter()
    cases = [
        "Spring Security和WebFlux之间的关系",
        "A模块依赖哪些其他模块",
        "ServiceA和ServiceB的关联",
    ]
    for q in cases:
        chosen = router.route(q)
        assert chosen == "graph_search", f"'{q}' -> got {chosen}"
        print(f"  '{q[:30]}' -> {chosen}")
    print("  PASS\n")


def test_rule_router_web():
    """RuleBasedRouter: 时效查询路由到web_search"""
    print("=== 测试4: 时效查询 -> web_search ===")
    router = RuleBasedRouter()
    cases = [
        "2026年AI最新趋势",
        "近期发布的Claude 4模型",
        "今年的技术大会",
    ]
    for q in cases:
        chosen = router.route(q)
        assert chosen == "web_search", f"'{q}' -> got {chosen}"
        print(f"  '{q[:30]}' -> {chosen}")
    print("  PASS\n")


def test_rule_router_priority():
    """RuleBasedRouter: 多触发条件按优先级路由"""
    print("=== 测试5: 路由优先级 ===")
    router = RuleBasedRouter()
    # 同时含"用户ID"（exact）和"最新"（web），exact优先级更高
    q = "查询用户ID 12345的最新信息"
    chosen = router.route(q)
    assert chosen == "exact_match", f"Priority broken: got {chosen}"
    print(f"  Mixed signals: '{q}' -> {chosen} (exact > web)")
    print("  PASS\n")


def test_rule_router_custom_default():
    """RuleBasedRouter: 自定义默认工具"""
    print("=== 测试6: 自定义默认工具 ===")
    router = RuleBasedRouter(default_tool="custom_default")
    assert router.route("普通问题") == "custom_default"
    print(f"  Custom default: custom_default")
    print("  PASS\n")


# ===== 测试：LLMRouter =====

def test_llm_router_basic():
    """LLMRouter: LLM输出工具名"""
    print("=== 测试7: LLMRouter基础决策 ===")
    toolbox = make_test_toolbox()
    llm = MockLLM(responses="vector_search")
    router = LLMRouter(llm=llm, toolbox=toolbox)

    chosen = router.route("INTJ的主导功能")
    assert chosen == "vector_search"
    assert llm.call_count == 1
    print(f"  LLM returned 'vector_search' -> chosen: {chosen}")
    print("  PASS\n")


def test_llm_router_extracts_tool_from_text():
    """LLMRouter: 从混合文本中提取工具名"""
    print("=== 测试8: LLMRouter文本解析 ===")
    toolbox = make_test_toolbox()
    # LLM返回带前缀/解释的文本
    llm = MockLLM(responses="我建议使用 exact_match 工具")
    router = LLMRouter(llm=llm, toolbox=toolbox)

    chosen = router.route("查用户ID")
    assert chosen == "exact_match"
    print(f"  Extracted 'exact_match' from explanatory text")
    print("  PASS\n")


def test_llm_router_fallback_on_no_match():
    """LLMRouter: LLM返回无效工具名时fallback"""
    print("=== 测试9: LLMRouter无匹配fallback ===")
    toolbox = make_test_toolbox()
    llm = MockLLM(responses="invalid_tool_name")
    router = LLMRouter(llm=llm, toolbox=toolbox, fallback_tool="vector_search")

    chosen = router.route("question")
    assert chosen == "vector_search"
    print(f"  No match -> fallback to: {chosen}")
    print("  PASS\n")


def test_llm_router_fallback_on_exception():
    """LLMRouter: LLM调用异常时fallback"""
    print("=== 测试10: LLMRouter LLM异常fallback ===")

    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("LLM service down")

    toolbox = make_test_toolbox()
    router = LLMRouter(llm=BrokenLLM(), toolbox=toolbox,
                       fallback_tool="vector_search")
    chosen = router.route("question")
    assert chosen == "vector_search"
    print(f"  LLM exception handled -> fallback: {chosen}")
    print("  PASS\n")


# ===== 测试：AgenticRetriever =====

def test_agentic_retriever_basic():
    """AgenticRetriever: Router + Toolbox组合检索"""
    print("=== 测试11: AgenticRetriever基础流程 ===")
    toolbox = make_test_toolbox()
    router = RuleBasedRouter()
    retriever = AgenticRetriever(toolbox, router)

    results = retriever.retrieve("INTJ的主导功能")
    assert isinstance(results, list)
    assert len(results) == 1  # MockRetriever返回1个doc
    print(f"  Retrieved {len(results)} docs via auto-routing")
    print("  PASS\n")


def test_agentic_retriever_with_decision():
    """AgenticRetriever: retrieve_with_decision返回决策信息"""
    print("=== 测试12: 检索带决策信息 ===")
    toolbox = make_test_toolbox()
    router = RuleBasedRouter()
    retriever = AgenticRetriever(toolbox, router)

    info = retriever.retrieve_with_decision("最新AI技术")
    assert info["tool"] == "web_search"  # "最新" -> web
    assert "documents" in info
    assert info["question"] == "最新AI技术"
    print(f"  Question: '{info['question']}' -> Tool: {info['tool']}")
    print("  PASS\n")


def test_agentic_retriever_force_tool():
    """AgenticRetriever: force_tool绕过Router"""
    print("=== 测试13: force_tool绕过Router ===")
    toolbox = make_test_toolbox()
    router = RuleBasedRouter()
    retriever = AgenticRetriever(toolbox, router)

    # 普通问题正常会走vector_search，强制走web_search
    results = retriever.retrieve("普通问题", force_tool="web_search")
    assert isinstance(results, list)  # web_search返回空list
    print(f"  Forced web_search bypassing router")
    print("  PASS\n")


# ===== 主测试入口 =====

if __name__ == "__main__":
    tests = [
        test_rule_router_default_to_vector,
        test_rule_router_exact_match,
        test_rule_router_graph,
        test_rule_router_web,
        test_rule_router_priority,
        test_rule_router_custom_default,
        test_llm_router_basic,
        test_llm_router_extracts_tool_from_text,
        test_llm_router_fallback_on_no_match,
        test_llm_router_fallback_on_exception,
        test_agentic_retriever_basic,
        test_agentic_retriever_with_decision,
        test_agentic_retriever_force_tool,
    ]

    print(f"\nRunning {len(tests)} agent router tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
