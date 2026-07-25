"""将Agentic RAG Agent集成到LangGraph Pipeline

修改src/graph.py，添加agentic_retrieve节点，替代传统的固定检索流程。

集成方式：
- 传统模式（ENABLE_AGENTIC_RAG=False）：使用固定HybridRetriever
- Agentic模式（ENABLE_AGENTIC_RAG=True）：使用AgenticRAGAgent自主决策
"""

# ===== 在 src/graph.py 中添加以下代码 =====

# 1. 在文件顶部导入
from src.agents.agentic_rag_agent import AgenticRAGAgent

# 2. 添加新的节点函数（在node_retrieve之后）
def node_agentic_retrieve(state: RAGState) -> dict:
    """Agentic检索节点：Agent自主决策多步检索
    
    与传统node_retrieve的区别：
    - 传统：固定1次检索，返回top_k结果
    - Agentic：Agent动态决定检索次数和策略，直到满意
    
    Agent的决策逻辑：
    1. 分析问题类型（Reasoning）
    2. 选择合适的工具（Acting）
    3. 观察检索结果（Observation）
    4. 评估结果质量（Reflection）
    5. 决定继续或结束（Decision）
    """
    collection_name = state.get("collection_name", "default")
    log.info(f"[Agentic] Starting agent-driven retrieval for: {state['rewritten_query'][:50]}")
    
    # 获取或创建Agent检索器
    from src.retrieval.agent_router import RuleBasedRouter
    
    indexer = get_indexer(collection_name)
    hybrid = HybridRetriever(indexer)
    
    # 创建工具箱和Router
    from src.retrieval.agentic_tools import create_toolbox
    toolbox = create_toolbox(hybrid)
    router = RuleBasedRouter(default_tool="vector_search")
    
    # 创建Agent
    agent = AgenticRAGAgent(
        toolbox=toolbox,
        router=router,
        max_steps=3,              # 最多3步检索
        min_docs_threshold=3,     # 至少3个文档
        quality_threshold=0.7     # 质量阈值0.7
    )
    
    # 运行Agent
    result = agent.run(state["rewritten_query"])
    
    # 提取结果
    documents = result["documents"]
    agent_steps = result["steps"]
    
    # 记录Agent决策历史
    agent_history = []
    for step in agent_steps:
        agent_history.append(
            f"Step{step.step_num}: {step.action} → {step.observation} → {step.reflection}"
        )
    
    log.info(f"[Agentic] Agent completed in {len(agent_steps)} steps, found {len(documents)} docs")
    
    return {
        "retrieved_docs": documents,
        "retrieval_decision": result["status"].value,
        "current_step": "retrieved",
        "history": state.get("history", []) + [f"Agentic retrieval: {len(agent_steps)} steps"] + agent_history,
    }


# 3. 修改build_graph函数，添加agentic分支
def build_graph(collection_name: str = "default", enable_agentic: bool = None) -> StateGraph:
    """构建LangGraph状态机
    
    Args:
        collection_name: 知识库名称
        enable_agentic: 是否启用Agentic RAG（None时读取config.ENABLE_AGENTIC_RAG）
    """
    if enable_agentic is None:
        enable_agentic = ENABLE_AGENTIC_RAG
    
    workflow = StateGraph(RAGState)
    
    # 节点注册
    workflow.add_node("analyze_query", node_analyze_query)
    
    if enable_agentic:
        # Agentic模式：Agent自主决策
        workflow.add_node("agentic_retrieve", node_agentic_retrieve)
    else:
        # 传统模式：固定检索
        workflow.add_node("retrieve", node_retrieve)
    
    workflow.add_node("grade_documents", node_grade_documents)
    workflow.add_node("rewrite_query", node_rewrite_query)
    workflow.add_node("web_search", node_web_search)
    workflow.add_node("generate", node_generate)
    workflow.add_node("regenerate", node_regenerate)
    workflow.add_node("check_facts", node_check_facts)
    workflow.add_node("check_conflicts", node_check_conflicts)
    
    # 边连接
    workflow.set_entry_point("analyze_query")
    
    if enable_agentic:
        workflow.add_edge("analyze_query", "agentic_retrieve")
        workflow.add_edge("agentic_retrieve", "grade_documents")
    else:
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        }
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_facts")
    workflow.add_conditional_edges(
        "check_facts",
        route_after_fact_check,
        {
            "check_conflicts": "check_conflicts",
            "regenerate": "regenerate",
        }
    )
    workflow.add_edge("regenerate", "check_facts")
    workflow.add_edge("check_conflicts", END)
    
    return workflow.compile(checkpointer=InMemorySaver())


# ===== 使用示例 =====

# 方式1：通过参数控制
graph_traditional = build_graph("my_collection", enable_agentic=False)  # 传统模式
graph_agentic = build_graph("my_collection", enable_agentic=True)       # Agentic模式

# 方式2：通过config控制（全局）
# 在src/config.py中设置 ENABLE_AGENTIC_RAG = True
graph = build_graph("my_collection")  # 读取config.ENABLE_AGENTIC_RAG


# ===== 性能对比测试 =====

def compare_traditional_vs_agentic():
    """对比传统Pipeline vs Agentic Agent"""
    from time import time
    
    test_questions = [
        "LangChain 是什么？",
        "如何实现自定义RAG系统？",
        "LangChain 2026年最新版本？"
    ]
    
    print("\n" + "="*80)
    print("传统Pipeline vs Agentic Agent 性能对比")
    print("="*80)
    
    for question in test_questions:
        print(f"\n问题：{question}")
        print("-" * 60)
        
        # 传统模式
        graph_trad = build_graph("test", enable_agentic=False)
        start = time()
        result_trad = graph_trad.invoke({"question": question, "collection_name": "test"})
        time_trad = time() - start
        
        # Agentic模式
        graph_agen = build_graph("test", enable_agentic=True)
        start = time()
        result_agen = graph_agen.invoke({"question": question, "collection_name": "test"})
        time_agen = time() - start
        
        # 对比
        print(f"传统模式：{time_trad:.2f}s，文档数={len(result_trad.get('retrieved_docs', []))}")
        print(f"Agentic模式：{time_agen:.2f}s，文档数={len(result_agen.get('retrieved_docs', []))}")
        print(f"Agent步数：{len([h for h in result_agen.get('history', []) if 'Step' in h])}")


if __name__ == "__main__":
    compare_traditional_vs_agentic()
