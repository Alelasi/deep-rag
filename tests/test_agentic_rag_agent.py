"""Agentic RAG Agent 测试脚本

测试场景：
1. 简单语义查询（1步完成）
2. 复杂查询（多步检索）
3. 时效性查询（Web搜索）
4. Agent决策能力展示
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.agentic_rag_agent import AgenticRAGAgent, AgentStatus
from src.retrieval.agentic_tools import AgenticRAGToolbox, VectorSearchTool
from src.retrieval.agent_router import RuleBasedRouter
from src.retrieval.indexer import Indexer
from src.retrieval.hybrid import HybridRetriever


def test_simple_query():
    """测试场景1：简单查询（预期1步完成）"""
    print("\n" + "="*80)
    print("测试场景1：简单语义查询")
    print("="*80)

    # 准备环境
    indexer = Indexer("test_collection")
    hybrid = HybridRetriever(indexer)

    # 创建工具箱（只注册vector_search）
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search", VectorSearchTool(hybrid))

    # 创建Router
    router = RuleBasedRouter(default_tool="vector_search")

    # 创建Agent
    agent = AgenticRAGAgent(
        toolbox=toolbox,
        router=router,
        max_steps=5,
        min_docs_threshold=2,
        quality_threshold=0.7
    )

    # 运行
    question = "LangChain 是什么？"
    result = agent.run(question)

    # 展示结果
    print(f"\n问题：{question}")
    print(f"状态：{result['status'].value}")
    print(f"推理：{result['reasoning']}")
    print(f"文档数：{len(result['documents'])}")
    print(f"\n执行步骤：")
    for step in result['steps']:
        print(f"\nStep {step.step_num}:")
        print(f"  Reasoning: {step.reasoning}")
        print(f"  Action: {step.action}")
        print(f"  Observation: {step.observation}")
        print(f"  Reflection: {step.reflection}")

    return result


def test_complex_query():
    """测试场景2：复杂查询（预期多步）"""
    print("\n" + "="*80)
    print("测试场景2：复杂查询（Agent多步决策）")
    print("="*80)

    # 准备环境
    indexer = Indexer("test_collection")
    hybrid = HybridRetriever(indexer)

    # 创建工具箱
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search", VectorSearchTool(hybrid))

    # 创建Router
    router = RuleBasedRouter(default_tool="vector_search")

    # 创建Agent（降低质量阈值，触发多步）
    agent = AgenticRAGAgent(
        toolbox=toolbox,
        router=router,
        max_steps=3,
        min_docs_threshold=5,  # 提高阈值，触发多步
        quality_threshold=0.9   # 提高阈值，触发多步
    )

    # 运行
    question = "如何在LangChain中实现自定义的RAG系统？"
    result = agent.run(question)

    # 展示结果
    print(f"\n问题：{question}")
    print(f"状态：{result['status'].value}")
    print(f"推理：{result['reasoning']}")
    print(f"文档数：{len(result['documents'])}")
    print(f"\n执行步骤：")
    for step in result['steps']:
        print(f"\nStep {step.step_num}:")
        print(f"  Reasoning: {step.reasoning}")
        print(f"  Action: {step.action}")
        print(f"  Observation: {step.observation}")
        print(f"  Reflection: {step.reflection}")
        print(f"  Status: {step.status.value}")

    return result


def test_web_search_query():
    """测试场景3：时效性查询（触发Web搜索）"""
    print("\n" + "="*80)
    print("测试场景3：时效性查询（Agent选择Web搜索）")
    print("="*80)

    # 准备环境
    indexer = Indexer("test_collection")
    hybrid = HybridRetriever(indexer)

    # 创建工具箱
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search", VectorSearchTool(hybrid))
    # 注意：web_search工具需要真实实现才能测试

    # 创建Router
    router = RuleBasedRouter(default_tool="vector_search")

    # 创建Agent
    agent = AgenticRAGAgent(
        toolbox=toolbox,
        router=router,
        max_steps=3,
        min_docs_threshold=2,
        quality_threshold=0.7
    )

    # 运行
    question = "LangChain 2026年最新版本有什么新特性？"
    result = agent.run(question)

    # 展示结果
    print(f"\n问题：{question}")
    print(f"状态：{result['status'].value}")
    print(f"推理：{result['reasoning']}")
    print(f"\n执行步骤：")
    for step in result['steps']:
        print(f"\nStep {step.step_num}:")
        print(f"  Reasoning: {step.reasoning}")
        print(f"  Action: {step.action} (Router识别到时效性关键词)")
        print(f"  Observation: {step.observation}")
        print(f"  Reflection: {step.reflection}")

    return result


def test_agent_decision_showcase():
    """测试场景4：展示Agent决策能力"""
    print("\n" + "="*80)
    print("测试场景4：Agent决策能力展示")
    print("="*80)

    print("\n【场景说明】")
    print("Agent会根据检索结果质量自主决定：")
    print("1. 如果第一次检索结果好 → 1步完成")
    print("2. 如果第一次结果不够 → 继续检索")
    print("3. 如果多次检索仍不够 → 最多3步后强制结束")
    print()

    # 准备环境
    indexer = Indexer("test_collection")
    hybrid = HybridRetriever(indexer)

    # 创建工具箱
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search", VectorSearchTool(hybrid))

    # 创建Router
    router = RuleBasedRouter(default_tool="vector_search")

    # 测试不同阈值下的行为
    test_cases = [
        ("宽松阈值（min_docs=2, quality=0.5）", 2, 0.5),
        ("中等阈值（min_docs=3, quality=0.7）", 3, 0.7),
        ("严格阈值（min_docs=5, quality=0.9）", 5, 0.9),
    ]

    question = "什么是RAG？"

    for name, min_docs, quality in test_cases:
        print(f"\n{name}")
        print("-" * 60)

        agent = AgenticRAGAgent(
            toolbox=toolbox,
            router=router,
            max_steps=3,
            min_docs_threshold=min_docs,
            quality_threshold=quality
        )

        result = agent.run(question)

        print(f"  结果：{result['status'].value}")
        print(f"  步数：{len(result['steps'])}步")
        print(f"  推理：{result['reasoning']}")

    return None


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Agentic RAG Agent 完整测试")
    print("="*80)
    print("\n【测试目标】")
    print("1. 验证Agent的ReAct循环（Reasoning → Acting → Observation → Reflection）")
    print("2. 展示Agent的自主决策能力（不同于固定Pipeline）")
    print("3. 演示多步检索场景（Agent根据质量决定是否继续）")
    print()

    # 运行测试
    try:
        test_simple_query()
        test_complex_query()
        test_web_search_query()
        test_agent_decision_showcase()

        print("\n" + "="*80)
        print("✅ 所有测试完成")
        print("="*80)

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
