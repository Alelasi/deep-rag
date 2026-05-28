"""Agentic RAG工具箱单元测试
验证工具箱注册、查询、执行流程；不依赖外部服务
"""
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.agentic_tools import (
    RetrievalTool,
    ExactMatchTool,
    VectorSearchTool,
    GraphSearchTool,
    WebSearchTool,
    AgenticRAGToolbox,
    create_toolbox,
)
from src.state import Document


# ===== Mock组件 =====

class MockRetriever:
    """模拟向量检索器"""
    def __init__(self, mock_results=None):
        self.mock_results = mock_results or []
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query: str, top_k: int = 5):
        self.last_query = query
        self.last_top_k = top_k
        return self.mock_results


class MockDatabase:
    """模拟数据库"""
    def __init__(self):
        self.queries = []

    def query(self, filters):
        self.queries.append(filters)
        return []


# ===== 测试：基类契约 =====

def test_retrieval_tool_is_abstract():
    """RetrievalTool是抽象基类，不能直接实例化"""
    print("=== 测试1: RetrievalTool抽象契约 ===")
    try:
        RetrievalTool()
        assert False, "Abstract class should not be instantiable"
    except TypeError:
        print("  RetrievalTool correctly enforces abstract methods")
    print("  PASS\n")


# ===== 测试：VectorSearchTool =====

def test_vector_search_tool_basic():
    """VectorSearchTool: 基础检索流程"""
    print("=== 测试2: VectorSearchTool基础检索 ===")
    mock_docs = [
        Document(doc_id="d1", content="INTJ的主导功能是Ni",
                 source="mbti.md", page=1, metadata={}),
        Document(doc_id="d2", content="ENFP的主导功能是Ne",
                 source="mbti.md", page=2, metadata={}),
    ]
    retriever = MockRetriever(mock_results=mock_docs)
    tool = VectorSearchTool(retriever)

    results = tool.search("INTJ的功能", top_k=5)

    assert len(results) == 2, f"Expected 2 docs, got {len(results)}"
    assert retriever.last_query == "INTJ的功能"
    assert retriever.last_top_k == 5
    print(f"  Retrieved {len(results)} documents")
    print(f"  Query passed correctly: {retriever.last_query}")
    print("  PASS\n")


def test_vector_search_tool_description():
    """VectorSearchTool: 描述符合Agent决策需求"""
    print("=== 测试3: VectorSearchTool描述 ===")
    tool = VectorSearchTool(MockRetriever())
    desc = tool.get_description()

    assert "向量检索" in desc or "语义" in desc
    assert len(desc) > 20, "Description should be informative"
    print(f"  Description: {desc[:60]}...")
    print("  PASS\n")


# ===== 测试：ExactMatchTool =====

def test_exact_match_tool_calls_db():
    """ExactMatchTool: 接受过滤条件"""
    print("=== 测试4: ExactMatchTool过滤查询 ===")
    db = MockDatabase()
    tool = ExactMatchTool(db)

    results = tool.search("查用户12345", filters={"user_id": "12345"})

    assert isinstance(results, list)
    print(f"  Tool accepts filters: {{'user_id': '12345'}}")
    print(f"  Returns list: {type(results).__name__}")
    print("  PASS\n")


def test_exact_match_tool_description():
    """ExactMatchTool: 描述包含场景关键词"""
    print("=== 测试5: ExactMatchTool描述 ===")
    tool = ExactMatchTool(MockDatabase())
    desc = tool.get_description()

    assert "精确" in desc or "ID" in desc
    print(f"  Description mentions exact match scenarios")
    print("  PASS\n")


# ===== 测试：GraphSearchTool =====

def test_graph_search_tool_basic():
    """GraphSearchTool: 实体关系查询"""
    print("=== 测试6: GraphSearchTool基础查询 ===")
    tool = GraphSearchTool(graph_db=None)

    results = tool.search(entity="Spring Security", relation="依赖")

    assert isinstance(results, list)
    desc = tool.get_description()
    assert "图" in desc or "关系" in desc
    print(f"  Tool returns list (graph_db=None means empty)")
    print("  PASS\n")


# ===== 测试：WebSearchTool =====

def test_web_search_tool_basic():
    """WebSearchTool: 网络搜索接口"""
    print("=== 测试7: WebSearchTool基础查询 ===")
    tool = WebSearchTool(search_api="tavily")

    results = tool.search("2026 AI最新趋势", max_results=3)

    assert isinstance(results, list)
    desc = tool.get_description()
    assert "网络" in desc or "互联网" in desc or "搜索" in desc
    print(f"  Web search tool created with api=tavily")
    print("  PASS\n")


# ===== 测试：AgenticRAGToolbox =====

def test_toolbox_register_and_get():
    """Toolbox: 工具注册与获取"""
    print("=== 测试8: 工具箱注册与获取 ===")
    toolbox = AgenticRAGToolbox()
    vector_tool = VectorSearchTool(MockRetriever())

    toolbox.register_tool("vector_search", vector_tool)

    fetched = toolbox.get_tool("vector_search")
    assert fetched is vector_tool, "Should return same instance"

    missing = toolbox.get_tool("nonexistent")
    assert missing is None, "Missing tool should return None"
    print(f"  Registered: vector_search")
    print(f"  Fetched same instance: {fetched is vector_tool}")
    print(f"  Missing tool returns None: {missing is None}")
    print("  PASS\n")


def test_toolbox_list_tools():
    """Toolbox: 列出所有工具供Agent决策"""
    print("=== 测试9: 工具箱列出工具 ===")
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector", VectorSearchTool(MockRetriever()))
    toolbox.register_tool("exact", ExactMatchTool(MockDatabase()))
    toolbox.register_tool("web", WebSearchTool())

    tools = toolbox.list_tools()
    assert len(tools) == 3
    names = [t["name"] for t in tools]
    assert "vector" in names
    assert "exact" in names
    assert "web" in names

    # 每个工具必须有描述供Agent判断
    for tool in tools:
        assert "description" in tool
        assert len(tool["description"]) > 10
    print(f"  Listed {len(tools)} tools: {names}")
    print("  PASS\n")


def test_toolbox_execute_tool():
    """Toolbox: 执行指定工具"""
    print("=== 测试10: 工具箱执行工具 ===")
    mock_docs = [
        Document(doc_id="d1", content="test content",
                 source="test.md", page=1, metadata={}),
    ]
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector", VectorSearchTool(MockRetriever(mock_docs)))

    results = toolbox.execute_tool("vector", "test query", top_k=3)
    assert len(results) == 1
    assert results[0]["doc_id"] == "d1"
    print(f"  Executed vector tool, got {len(results)} results")
    print("  PASS\n")


def test_toolbox_execute_unknown_raises():
    """Toolbox: 执行未知工具抛出ValueError"""
    print("=== 测试11: 工具箱执行未知工具 ===")
    toolbox = AgenticRAGToolbox()
    try:
        toolbox.execute_tool("nonexistent", "query")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "not found" in str(e).lower() or "nonexistent" in str(e)
    print(f"  Correctly raises ValueError for unknown tool")
    print("  PASS\n")


# ===== 测试：create_toolbox工厂 =====

def test_create_toolbox_factory():
    """create_toolbox: 工厂函数注册默认工具"""
    print("=== 测试12: create_toolbox工厂 ===")
    retriever = MockRetriever()
    toolbox = create_toolbox(retriever)

    # 至少应注册vector_search
    vector_tool = toolbox.get_tool("vector_search")
    assert vector_tool is not None
    assert isinstance(vector_tool, VectorSearchTool)
    print(f"  Factory registered: vector_search (and possibly more)")
    print("  PASS\n")


# ===== 集成测试：Agent决策模拟 =====

def test_agent_decision_flow():
    """模拟Agent根据问题选择工具的完整流程"""
    print("=== 测试13: Agent决策流程模拟 ===")
    # 准备3种工具
    toolbox = AgenticRAGToolbox()

    semantic_docs = [Document(doc_id="s1", content="语义结果",
                              source="a.md", page=1, metadata={})]
    toolbox.register_tool("vector", VectorSearchTool(MockRetriever(semantic_docs)))
    toolbox.register_tool("exact", ExactMatchTool(MockDatabase()))
    toolbox.register_tool("web", WebSearchTool())

    # 模拟问题路由（简化的关键词路由，真实场景由LLM决策）
    def route(question: str) -> str:
        if "ID" in question or "查询" in question:
            return "exact"
        elif "最新" in question or "2026" in question:
            return "web"
        else:
            return "vector"

    test_cases = [
        ("INTJ的主导功能是什么", "vector"),
        ("查询用户ID 12345", "exact"),
        ("2026年最新AI趋势", "web"),
    ]
    for question, expected in test_cases:
        chosen = route(question)
        assert chosen == expected, f"For '{question}', expected {expected}, got {chosen}"
        results = toolbox.execute_tool(chosen, question)
        assert isinstance(results, list)
        print(f"  Question: '{question[:30]}' -> Tool: {chosen}, Results: {len(results)}")
    print("  PASS\n")


# ===== 主测试入口 =====

if __name__ == "__main__":
    tests = [
        test_retrieval_tool_is_abstract,
        test_vector_search_tool_basic,
        test_vector_search_tool_description,
        test_exact_match_tool_calls_db,
        test_exact_match_tool_description,
        test_graph_search_tool_basic,
        test_web_search_tool_basic,
        test_toolbox_register_and_get,
        test_toolbox_list_tools,
        test_toolbox_execute_tool,
        test_toolbox_execute_unknown_raises,
        test_create_toolbox_factory,
        test_agent_decision_flow,
    ]

    print(f"\nRunning {len(tests)} agentic tools tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}\n")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
