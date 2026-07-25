"""Agentic RAG Agent 演示脚本（使用模拟数据）

目标：展示Agent的ReAct循环和自主决策能力
不依赖真实向量数据库，使用模拟检索结果
"""
import sys
sys.path.append("D:/文档/ai提问相关/工作/deep-rag")

from src.agents.agentic_rag_agent import AgenticRAGAgent, AgentStatus
from src.retrieval.agentic_tools import AgenticRAGToolbox, RetrievalTool
from src.retrieval.agent_router import RuleBasedRouter
from src.state import Document
from typing import List


# === 模拟工具（用于演示） ===

class MockVectorSearchTool(RetrievalTool):
    """模拟向量检索工具"""

    def __init__(self, call_count=0):
        self.call_count = call_count

    def search(self, query: str, **kwargs) -> List[Document]:
        """模拟检索：第一次返回低质量，第二次返回高质量"""
        self.call_count += 1

        if self.call_count == 1:
            # 第一次：低质量结果（模拟需要继续检索）
            return [
                Document(
                    content=f"关于{query}的部分信息...",
                    metadata={"source": "doc1.md"},
                    score=0.6
                )
            ]
        else:
            # 第二次：高质量结果（模拟可以结束）
            return [
                Document(
                    content=f"关于{query}的详细信息：LangChain是一个用于开发LLM应用的框架...",
                    metadata={"source": "doc2.md"},
                    score=0.9
                ),
                Document(
                    content=f"LangChain的核心组件包括：Chains、Agents、Memory...",
                    metadata={"source": "doc3.md"},
                    score=0.85
                ),
                Document(
                    content=f"使用LangChain可以快速构建RAG系统...",
                    metadata={"source": "doc4.md"},
                    score=0.8
                )
            ]

    def get_description(self) -> str:
        return "向量检索工具：基于语义相似度检索"


class MockWebSearchTool(RetrievalTool):
    """模拟Web搜索工具"""

    def search(self, query: str, **kwargs) -> List[Document]:
        """模拟Web搜索"""
        return [
            Document(
                content=f"来自网络的最新信息：{query}的2026年更新...",
                metadata={"source": "web:blog.com"},
                score=0.75
            ),
            Document(
                content=f"GitHub Release Notes: {query} v0.3.0新特性...",
                metadata={"source": "web:github.com"},
                score=0.85
            )
        ]

    def get_description(self) -> str:
        return "Web搜索工具：获取最新信息"


# === 演示场景 ===

def demo_simple_query():
    """场景1：简单查询（2步完成）"""
    print("\n" + "="*80)
    print("【场景1】简单查询：Agent 2步完成")
    print("="*80)

    # 创建工具箱
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search", MockVectorSearchTool())

    # 创建Router
    router = RuleBasedRouter(default_tool="vector_search")

    # 创建Agent
    agent = AgenticRAGAgent(
        toolbox=toolbox,
        router=router,
        max_steps=5,
        min_docs_threshold=2,    # 需要至少2个文档
        quality_threshold=0.7     # 质量阈值0.7
    )

    # 运行
    question = "LangChain 是什么？"
    result = agent.run(question)

    # 展示结果
    print(f"\n问题：{question}")
    print(f"状态：{result['status'].value}")
    print(f"推理：{result['reasoning']}")
    print(f"文档数：{len(result['documents'])}")

    print(f"\n执行步骤（共{len(result['steps'])}步）：")
    for step in result['steps']:
        print(f"\n  Step {step.step_num}:")
        print(f"    🧠 Reasoning: {step.reasoning}")
        print(f"    🔨 Action: {step.action}")
        print(f"    👀 Observation: {step.observation}")
        print(f"    💭 Reflection: {step.reflection}")

    print(f"\n最终文档（{len(result['documents'])}个）：")
    for i, doc in enumerate(result['documents'][:3], 1):
        score = doc.get("score", 0.5)
        content = doc.get("content", "")
        print(f"  {i}. [{score:.2f}] {content[:60]}...")

    return result


def demo_web_search_query():
    """场景2：时效性查询（触发Web搜索）"""
    print("\n" + "="*80)
    print("【场景2】时效性查询：Agent识别到关键词，选择Web搜索")
    print("="*80)

    # 创建工具箱（注册2个工具）
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search", MockVectorSearchTool())
    toolbox.register_tool("web_search", MockWebSearchTool())

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

    print(f"\n执行步骤（共{len(result['steps'])}步）：")
    for step in result['steps']:
        print(f"\n  Step {step.step_num}:")
        print(f"    🧠 Reasoning: {step.reasoning}")
        print(f"    🔨 Action: {step.action} {'✅ (识别到时效性关键词)' if step.action == 'web_search' else ''}")
        print(f"    👀 Observation: {step.observation}")
        print(f"    💭 Reflection: {step.reflection}")

    print(f"\n最终文档（{len(result['documents'])}个）：")
    for i, doc in enumerate(result['documents'], 1):
        score = doc.get("score", 0.5)
        source = doc.get("metadata", {}).get("source", "unknown")
        content = doc.get("content", "")
        print(f"  {i}. [{score:.2f}] {source} - {content[:50]}...")

    return result


def demo_multi_step_decision():
    """场景3：展示Agent的多步决策能力"""
    print("\n" + "="*80)
    print("【场景3】多步决策：展示Agent如何根据质量阈值决定检索次数")
    print("="*80)

    question = "什么是RAG？"

    # 测试不同阈值下的行为
    test_cases = [
        ("宽松阈值（min_docs=1, quality=0.5）", 1, 0.5, "✅ 预期1步完成"),
        ("中等阈值（min_docs=2, quality=0.7）", 2, 0.7, "⚠️ 预期2步完成"),
        ("严格阈值（min_docs=5, quality=0.9）", 5, 0.9, "❌ 预期达到最大步数"),
    ]

    for name, min_docs, quality, expect in test_cases:
        print(f"\n{name} - {expect}")
        print("-" * 60)

        # 创建工具箱
        toolbox = AgenticRAGToolbox()
        toolbox.register_tool("vector_search", MockVectorSearchTool())

        # 创建Router
        router = RuleBasedRouter(default_tool="vector_search")

        # 创建Agent
        agent = AgenticRAGAgent(
            toolbox=toolbox,
            router=router,
            max_steps=3,
            min_docs_threshold=min_docs,
            quality_threshold=quality
        )

        # 运行
        result = agent.run(question)

        # 展示结果
        print(f"  结果：{result['status'].value}")
        print(f"  步数：{len(result['steps'])}步")
        print(f"  文档数：{len(result['documents'])}个")
        print(f"  推理：{result['reasoning']}")


def demo_react_cycle_detail():
    """场景4：详细展示ReAct循环的每个阶段"""
    print("\n" + "="*80)
    print("【场景4】ReAct循环详解：深入展示Agent的决策过程")
    print("="*80)

    # 创建工具箱
    toolbox = AgenticRAGToolbox()
    toolbox.register_tool("vector_search", MockVectorSearchTool())

    # 创建Router
    router = RuleBasedRouter(default_tool="vector_search")

    # 创建Agent
    agent = AgenticRAGAgent(
        toolbox=toolbox,
        router=router,
        max_steps=3,
        min_docs_threshold=3,    # 需要3个文档
        quality_threshold=0.8     # 质量0.8
    )

    # 运行
    question = "如何使用LangChain构建RAG系统？"
    result = agent.run(question)

    print(f"\n问题：{question}")
    print(f"\n{'='*80}")
    print("ReAct循环详解：")
    print('='*80)

    for step in result['steps']:
        print(f"\n┌─ Step {step.step_num} ─────────────────────────────────")
        print(f"│")
        print(f"│ 🧠 Reasoning（推理）：")
        print(f"│    {step.reasoning}")
        print(f"│")
        print(f"│ 🔨 Acting（行动）：")
        print(f"│    选择工具: {step.action}")
        print(f"│    执行查询: {step.action_input[:50]}...")
        print(f"│")
        print(f"│ 👀 Observation（观察）：")
        print(f"│    {step.observation}")
        print(f"│")
        print(f"│ 💭 Reflection（反思）：")
        print(f"│    {step.reflection}")
        print(f"│")
        print(f"│ 🎯 Decision（决策）：")
        if step.step_num < len(result['steps']):
            print(f"│    → 继续检索（质量不够）")
        else:
            print(f"│    → 结束检索（质量达标或达到最大步数）")
        print(f"└────────────────────────────────────────────")

    print(f"\n最终状态：{result['status'].value}")
    print(f"总文档数：{len(result['documents'])}个")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Agentic RAG Agent 完整演示（使用模拟数据）")
    print("="*80)
    print("\n【演示目标】")
    print("1. 展示Agent的ReAct循环（Reasoning → Acting → Observation → Reflection）")
    print("2. 展示Agent的自主决策能力（根据质量决定是否继续）")
    print("3. 展示多步检索场景（不同于固定Pipeline）")
    print("4. 展示工具选择能力（根据问题类型选择工具）")

    try:
        # 运行演示
        demo_simple_query()
        demo_web_search_query()
        demo_multi_step_decision()
        demo_react_cycle_detail()

        print("\n" + "="*80)
        print("✅ 所有演示完成")
        print("="*80)

        print("\n【关键结论】")
        print("1. Agent能够自主决定检索次数（1-5步不等）")
        print("2. Agent能够根据问题类型选择工具（vector/web）")
        print("3. Agent能够评估结果质量并决定是否继续")
        print("4. 完整实现了ReAct循环的4个阶段")

    except Exception as e:
        print(f"\n❌ 演示失败：{e}")
        import traceback
        traceback.print_exc()
