"""
DeepRAG 主Pipeline — LangGraph状态机
7层Pipeline + Corrective RAG纠错循环 + Self-RAG事实校验循环
"""
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from src.state import RAGState
from src.agents.query_analyzer import analyze_query_offline as analyze_query
from src.agents.doc_grader import grade_documents_offline as grade_documents
from src.agents.generator import generate_answer_offline as generate_answer
from src.agents.fact_checker import check_facts_offline as check_facts
from src.agents.conflict_resolver import resolve_conflicts_offline as resolve_conflicts
from src.retrieval.indexer import Indexer
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.web_fallback import web_search_fallback
from src.config import ENABLE_AGENTIC_RAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("deeprag")

# 全局索引器（按collection_name区分知识库）
_indexers: dict[str, Indexer] = {}
# Agentic RAG 全局缓存（避免每次 node_retrieve 重建 toolbox/router）
_agentic_retrievers: dict[str, object] = {}


def get_indexer(collection_name: str) -> Indexer:
    if collection_name not in _indexers:
        _indexers[collection_name] = Indexer(collection_name)
    return _indexers[collection_name]


def get_agentic_retriever(collection_name: str):
    """Agentic RAG 检索器工厂（懒加载 + 缓存）

    组合：AgenticRAGToolbox（vector_search 真实工具） + RuleBasedRouter（零延迟规则路由）
    其他工具（exact_match/graph_search/web_search）当前为接口预留，未注册。
    """
    if collection_name in _agentic_retrievers:
        return _agentic_retrievers[collection_name]

    from src.retrieval.agentic_tools import create_toolbox
    from src.retrieval.agent_router import RuleBasedRouter, AgenticRetriever

    indexer = get_indexer(collection_name)
    hybrid = HybridRetriever(indexer)
    toolbox = create_toolbox(hybrid)  # 注册 vector_search

    router = RuleBasedRouter(default_tool="vector_search")
    retriever = AgenticRetriever(toolbox, router)

    _agentic_retrievers[collection_name] = retriever
    return retriever


# === 路由函数 ===

def route_after_grading(state: RAGState) -> str:
    """Corrective RAG路由：根据文档评分决策"""
    relevant = state["relevant_count"]
    total = len(state["graded_docs"])

    if relevant >= 1:
        return "generate"       # 有相关文档→生成答案
    elif state["retrieval_decision"] == "web_search":
        return "web_search"     # 已经web搜索过了还不行→强制生成
    elif state["retry_count"] < state["max_retries"]:
        return "rewrite_query"  # 无相关文档→改写查询重试
    else:
        return "web_search"     # 重试耗尽→web兜底


def route_after_fact_check(state: RAGState) -> str:
    """Self-RAG路由：事实校验后决策"""
    if state["fact_check_passed"]:
        return "check_conflicts"
    elif state["retry_count"] < state["max_retries"]:
        return "regenerate"     # 幻觉严重→重新生成
    else:
        return "check_conflicts"  # 重试耗尽→带着警告输出


# === 节点函数 ===

def node_analyze_query(state: RAGState) -> dict:
    """1.查询分析+改写"""
    log.info(f"Analyzing query: {state['question'][:50]}")
    result = analyze_query(state["question"])
    return {
        "question_type": result["question_type"],
        "rewritten_query": result["rewritten_query"],
        "search_queries": result["search_queries"],
        "current_step": "query_analyzed",
        "history": state.get("history", []) + [f"Query type: {result['question_type']}"],
    }


def node_retrieve(state: RAGState) -> dict:
    """2.检索（根据 ENABLE_AGENTIC_RAG 切换 Hybrid ↔ Agentic）"""
    query = state["rewritten_query"] or state["question"]
    log.info(f"Retrieving for: {query[:50]}")

    if ENABLE_AGENTIC_RAG:
        retriever = get_agentic_retriever(state["collection_name"])
        decision = retriever.retrieve_with_decision(query, top_k=8)
        docs = decision["documents"]
        chosen_tool = decision["tool"]
        log.info(f"[Agentic] Router chose tool='{chosen_tool}', retrieved {len(docs)} docs")
        history_entry = f"Retrieved {len(docs)} docs via {chosen_tool}"
    else:
        indexer = get_indexer(state["collection_name"])
        retriever = HybridRetriever(indexer)
        docs = retriever.retrieve(query, top_k=8)
        log.info(f"[Hybrid] Retrieved {len(docs)} documents")
        history_entry = f"Retrieved {len(docs)} docs"

    return {
        "retrieved_docs": docs,
        "current_step": "retrieved",
        "history": state.get("history", []) + [history_entry],
    }


def node_grade_docs(state: RAGState) -> dict:
    """3.Corrective RAG文档评分"""
    log.info("Grading document relevance...")
    graded = grade_documents(state["question"], state["retrieved_docs"])

    relevant = sum(1 for d in graded if d["grade"] == "relevant")
    irrelevant = sum(1 for d in graded if d["grade"] == "irrelevant")
    log.info(f"Grading: {relevant} relevant, {len(graded)-relevant-irrelevant} ambiguous, {irrelevant} irrelevant")

    # 决定下一步
    if relevant >= 1:
        decision = "generate"
    elif state["retry_count"] < state["max_retries"]:
        decision = "rewrite"
    else:
        decision = "web_search"

    return {
        "graded_docs": graded,
        "relevant_count": relevant,
        "irrelevant_count": irrelevant,
        "retrieval_decision": decision,
        "current_step": "graded",
        "history": state.get("history", []) + [f"Graded: {relevant}R/{irrelevant}I"],
    }


def node_rewrite_query(state: RAGState) -> dict:
    """4.查询改写（纠错循环入口）"""
    retry = state.get("retry_count", 0) + 1
    log.info(f"Rewriting query (attempt {retry})...")

    # 简单改写策略：扩展关键词
    original = state["rewritten_query"] or state["question"]
    # 如果第一次改写失败，尝试更泛化的查询
    if retry == 1:
        new_query = original + " 相关概念 定义 说明"
    else:
        new_query = original.split()[0] if " " in original else original

    return {
        "rewritten_query": new_query,
        "retry_count": retry,
        "current_step": "query_rewritten",
        "history": state.get("history", []) + [f"Query rewritten (#{retry}): {new_query[:30]}"],
    }


def node_web_search(state: RAGState) -> dict:
    """4b.Web搜索兜底"""
    log.info("Knowledge base insufficient, falling back to web search...")
    results = web_search_fallback(state["question"])
    return {
        "web_results": results,
        "retrieval_decision": "web_search",
        "current_step": "web_searched",
        "history": state.get("history", []) + ["Web fallback triggered"],
    }


def node_generate(state: RAGState) -> dict:
    """5.带引用的答案生成"""
    log.info("Generating answer with citations...")

    # 使用relevant+ambiguous文档
    source_docs = [d for d in state["graded_docs"] if d["grade"] in ("relevant", "ambiguous")]
    # 如果有web结果也加入
    if state.get("web_results"):
        for wr in state["web_results"]:
            source_docs.append({**wr, "grade": "relevant", "relevance_score": 0.5, "reasoning": "web"})

    result = generate_answer(state["question"], source_docs)
    log.info(f"Answer generated ({len(result['answer'])} chars, {len(result['citations'])} citations)")

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "current_step": "generated",
        "history": state.get("history", []) + [f"Generated {len(result['answer'])} chars"],
    }


def node_fact_check(state: RAGState) -> dict:
    """6.Self-RAG事实校验"""
    log.info("Fact-checking answer against sources...")
    source_docs = [d for d in state["graded_docs"] if d["grade"] in ("relevant", "ambiguous")]
    result = check_facts(state["answer"], source_docs)

    log.info(f"Hallucination score: {result['hallucination_score']:.2f} "
             f"({'PASS' if result['passed'] else 'FAIL'})")

    new_retry = state.get("retry_count", 0)
    if not result["passed"]:
        new_retry += 1

    return {
        "hallucination_score": result["hallucination_score"],
        "fact_check_passed": result["passed"],
        "unsupported_claims": result["unsupported_claims"],
        "retry_count": new_retry,
        "current_step": "fact_checked",
        "history": state.get("history", []) +
                  [f"Fact check: {result['hallucination_score']:.2f} ({'PASS' if result['passed'] else 'FAIL'})"],
    }


def node_check_conflicts(state: RAGState) -> dict:
    """7.多源冲突检测"""
    log.info("Checking for source conflicts...")
    relevant_docs = [d for d in state["graded_docs"] if d["grade"] == "relevant"]
    conflicts = resolve_conflicts(state["question"], relevant_docs)

    if conflicts:
        log.info(f"Found {len(conflicts)} conflicts")
    else:
        log.info("No conflicts detected")

    return {
        "conflicts": conflicts,
        "current_step": "completed",
        "history": state.get("history", []) + [f"Conflicts: {len(conflicts)}"],
    }


# === 构建图 ===

def build_graph() -> StateGraph:
    """构建DeepRAG Pipeline"""
    graph = StateGraph(RAGState)

    graph.add_node("analyze_query", node_analyze_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_docs", node_grade_docs)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("web_search", node_web_search)
    graph.add_node("generate", node_generate)
    graph.add_node("fact_check", node_fact_check)
    graph.add_node("check_conflicts", node_check_conflicts)

    # 流转
    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_docs")

    # Corrective RAG分支
    graph.add_conditional_edges("grade_docs", route_after_grading, {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search",
    })

    # 改写后重新检索（循环）
    graph.add_edge("rewrite_query", "retrieve")
    # Web搜索后直接生成
    graph.add_edge("web_search", "generate")

    # 生成后事实校验
    graph.add_edge("generate", "fact_check")

    # Self-RAG分支
    graph.add_conditional_edges("fact_check", route_after_fact_check, {
        "check_conflicts": "check_conflicts",
        "regenerate": "generate",  # 重新生成（循环）
    })

    graph.add_edge("check_conflicts", END)

    return graph


def create_app():
    graph = build_graph()
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


def query(question: str, collection_name: str = "default",
          max_retries: int = 2) -> dict:
    """执行一次RAG查询，返回完整结果"""
    app = create_app()
    config = {"configurable": {"thread_id": f"rag-{hash(question) % 10000}"}}

    initial_state: RAGState = {
        "question": question,
        "collection_name": collection_name,
        "question_type": "factual",
        "rewritten_query": "",
        "search_queries": [],
        "retrieved_docs": [],
        "graded_docs": [],
        "relevant_count": 0,
        "irrelevant_count": 0,
        "retrieval_decision": "generate",
        "answer": "",
        "citations": [],
        "hallucination_score": 0.0,
        "fact_check_passed": True,
        "unsupported_claims": [],
        "conflicts": [],
        "web_results": [],
        "current_step": "init",
        "retry_count": 0,
        "max_retries": max_retries,
        "errors": [],
        "history": [],
    }

    for event in app.stream(initial_state, config=config):
        pass  # 静默执行

    state = app.get_state(config)
    return state.values


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.graph <docs_dir> <question>")
        print("  Indexes docs_dir and answers the question.")
        sys.exit(1)

    docs_dir = sys.argv[1]
    question = " ".join(sys.argv[2:])

    # 索引文档
    indexer = get_indexer("cli_kb")
    count = indexer.index_directory(docs_dir)
    print(f"Indexed {count} chunks from {docs_dir}")

    # 查询
    result = query(question, collection_name="cli_kb")
    print(f"\nQuestion: {question}")
    print(f"Answer:\n{result.get('answer', 'No answer')}")
    print(f"\nHallucination: {result.get('hallucination_score', 0):.2f}")
    print(f"Citations: {len(result.get('citations', []))}")
    print(f"Conflicts: {len(result.get('conflicts', []))}")
    print(f"History: {result.get('history', [])}")
