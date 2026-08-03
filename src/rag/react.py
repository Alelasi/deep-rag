"""
DeepRAG ReAct 循环（v2.4新增，从 god-module src/graph.py 抽出）

包含：
- REACT_PROMPT
- _summarize_docs / _parse_json_response 工具
- node_agent_decision / route_react_agent
- 各 ReAct 工具节点（vector / exact_match / graph / web / kb_stats / generate）
- build_agentic_graph / create_agentic_app
"""
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from src.state import RAGState
from src.pipeline.caches import (
    USE_QDRANT,
    QdrantHybridRetriever,
    get_indexer,
    _batch_get_all,
)
from src.retrieval.hybrid import HybridRetriever  # 主 Hybrid：HybridRetriever(indexer)
from src.retrieval.web_fallback import web_search_fallback

log = logging.getLogger("deeprag")


REACT_PROMPT = """你是一个智能检索助手。根据用户问题和当前状态，决定下一步操作。

可用工具：
1. vector_search: 向量语义检索（适合模糊问题、概念查询）
2. exact_match: 精确ID查询（适合查特定文档编号）
3. graph_search: 关系图谱检索（适合查实体间关联关系）
4. web_search: 网络搜索（适合时效性问题或知识库没有的内容）
5. kb_stats: 系统统计（查知识库数据量、模型信息、GPU状态、缓存等全部元数据）
6. generate: 生成最终答案（检索结果足够时）
7. rewrite: 改写查询后重新检索（检索结果不相关时）

当前状态：
- 已检索文档数：{doc_count}
- 检索轮次：{round}/{max_round}
- 已用工具：{used_tools}

请输出JSON格式：{{"action": "工具名", "reason": "选择理由", "query": "检索词（如果需要检索）"}}
如果已有文档足够回答问题，action设为"generate"。
最多检索{max_round}轮，超过后必须generate。"""


def _summarize_docs(docs: list, max_chars: int = 500) -> str:
    """摘要已检索文档供Agent决策参考"""
    if not docs:
        return "（暂无文档）"
    summaries = []
    for i, doc in enumerate(docs[:5]):  # 最多摘要5个
        content = doc.get("content", "") if isinstance(doc, dict) else getattr(doc, "content", "")
        source = doc.get("source", "") if isinstance(doc, dict) else getattr(doc, "source", "")
        text = f"[{i+1}] ({source}) {content[:100]}..."
        summaries.append(text)
    return "\n".join(summaries)[:max_chars]


def _parse_json_response(text: str) -> dict:
    """解析LLM返回的JSON决策"""
    import json
    import re
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 尝试提取JSON块
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # 兜底
    return {"action": "generate", "reason": "parse_error", "query": ""}


def node_agent_decision(state: RAGState) -> dict:
    """Agent决策节点：LLM决定下一步操作

    v2.6修复：
    - 无LLM时不再直接generate，而是先检索（0条文档时必须先搜）
    - LLM失败时降级到规则路由，不跳过检索
    - 第0轮且0条文档时强制vector_search
    """
    from src.config import get_llm_with_fallback
    from langchain_core.messages import HumanMessage

    used_tools = state.get("used_tools", [])
    docs = state.get("retrieved_docs", [])
    max_round = state.get("max_retries", 3)
    retrieval_round = state.get("retrieval_round", 0)

    # v2.6: 第0轮且0条文档时，强制先检索，不让LLM跳过
    if len(docs) == 0 and "vector_search" not in used_tools:
        log.info("[ReAct] 0 docs, first round — forcing vector_search")
        return {"next_action": "vector_search", "agent_reason": "auto_retrieve_no_docs",
                "retrieval_round": retrieval_round + 1}

    # v2.8.3: 向量检索0结果且未试过web_search时，强制联网搜索
    if len(docs) == 0 and "vector_search" in used_tools and "web_search" not in used_tools:
        log.info("[ReAct] 0 docs after vector_search — forcing web_search")
        return {"next_action": "web_search", "agent_reason": "no_docs_force_web",
                "retrieval_round": retrieval_round + 1}

    llm = get_llm_with_fallback()
    if llm is None:
        # 无LLM：有文档就生成，没文档就Web搜索
        if len(docs) > 0:
            return {"next_action": "generate", "agent_reason": "no_llm_has_docs",
                    "retrieval_round": retrieval_round + 1}
        else:
            return {"next_action": "web_search", "agent_reason": "no_llm_no_docs",
                    "retrieval_round": retrieval_round + 1}

    prompt = REACT_PROMPT.format(
        doc_count=len(docs),
        round=retrieval_round,
        max_round=max_round,
        used_tools=", ".join(used_tools) if used_tools else "（无）",
    )

    try:
        response = llm.invoke([
            HumanMessage(content=f"{prompt}\n\n用户问题：{state['question']}\n\n已有文档摘要：\n{_summarize_docs(docs)}")
        ])
        text = response.content if hasattr(response, "content") else str(response)
        decision = _parse_json_response(text)
    except Exception as e:
        log.warning(f"[ReAct] LLM decision failed: {e}, using rule-based fallback")
        # v2.6: LLM失败时用规则路由，不直接generate
        if len(docs) == 0:
            if "web_search" not in used_tools:
                decision = {"action": "web_search", "reason": f"llm_fail_no_docs: {e}", "query": state["question"]}
            else:
                decision = {"action": "generate", "reason": "llm_fail_after_web", "query": ""}
        else:
            decision = {"action": "generate", "reason": f"llm_fail_has_docs: {e}", "query": ""}

    action = decision.get("action", "generate")
    reason = decision.get("reason", "")
    new_query = decision.get("query", "")

    log.info(f"[ReAct] Decision: action={action}, reason={reason}")

    result = {
        "next_action": action,
        "agent_reason": reason,
        "retrieval_round": state.get("retrieval_round", 0) + 1,
    }
    if new_query:
        result["rewritten_query"] = new_query

    return result


def route_react_agent(state: RAGState) -> str:
    """根据Agent决策路由到对应工具节点"""
    action = state.get("next_action", "generate")
    max_round = state.get("max_retries", 3)

    # 强制终止条件：超过最大轮次
    if state.get("retrieval_round", 0) >= max_round:
        log.info(f"[ReAct] Max rounds ({max_round}) reached, forcing generate")
        return "generate"

    # 验证工具名
    valid_tools = {"vector_search", "exact_match", "graph_search", "web_search", "kb_stats", "generate"}
    if action not in valid_tools:
        return "generate"

    return action


def node_react_vector_search(state: RAGState) -> dict:
    """ReAct: 向量检索（v2.7：缓存BM25+文档，避免每次重建）"""
    query = state.get("rewritten_query") or state["question"]

    # v2.7: 使用与Enhanced模式相同的缓存BM25+Hybrid+Rerank Pipeline
    try:
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.hybrid_retriever import ParallelHybridRetriever as NewHybrid
        from src.retrieval.cache import get_cached_documents, get_cached_bm25

        indexer = get_indexer(state["collection_name"])
        col_name = state["collection_name"]

        # v2.7: 缓存文档列表
        def _fetch_docs():
            import logging as _logging
            _log = _logging.getLogger("deeprag")
            bm25_docs = []

            if USE_QDRANT:
                # Qdrant: 从检索器获取所有文档
                try:
                    scroll = indexer.retriever.client.scroll(
                        collection_name=indexer.collection_name,
                        limit=10000,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for point in scroll[0]:
                        payload = point.payload or {}
                        bm25_docs.append({
                            "doc_id": payload.get("doc_id", str(point.id)),
                            "content": payload.get("content", ""),
                            "source": payload.get("source", ""),
                            "page": payload.get("page", 0),
                            "metadata": payload,
                        })
                except Exception as e:
                    _log.warning(f"[Qdrant] 文档读取失败: {e}")
            else:
                # ChromaDB: 从集合读取
                all_collections = indexer.get_all_collections()
                for col in all_collections:
                    try:
                        col_data = _batch_get_all(col)
                        for i, (doc_text, meta) in enumerate(zip(col_data.get("documents", []),
                                                                  col_data.get("metadatas", []))):
                            bm25_docs.append({
                                "doc_id": col_data.get("ids", [f"doc_{i}"])[i] if i < len(col_data.get("ids", [])) else f"doc_{i}",
                                "content": doc_text,
                                "source": meta.get("source", "unknown"),
                                "page": meta.get("page", 0),
                                "metadata": meta,
                            })
                    except Exception as e:
                        _log.warning(f"[Hybrid v2.7] 子集合 {col.name} 读取失败: {e}")
            return bm25_docs

        bm25_docs = get_cached_documents(col_name, _fetch_docs)

        if bm25_docs:
            bm25 = get_cached_bm25(col_name, lambda: BM25Retriever(bm25_docs))
            if USE_QDRANT:
                hybrid = QdrantHybridRetriever(indexer)
            else:
                hybrid = NewHybrid(bm25, indexer)
            docs = hybrid.search(query, top_k=10)

            # Rerank精排（v2.8.2: 尊重 ENABLE_RERANKER 配置, v2.8.3: 减少输入到8篇）
            if docs:
                from src.config import ENABLE_RERANKER
                if ENABLE_RERANKER:
                    try:
                        from src.retrieval.reranker import Reranker
                        reranker = Reranker()
                        docs = reranker.rerank(query, docs[:8], top_k=5)
                    except Exception:
                        docs = docs[:5]
                else:
                    docs = docs[:5]

            log.info(f"[ReAct v2.7] BM25+Hybrid+Rerank retrieved {len(docs)} docs (cached)")
        else:
            docs = []
            log.warning("[ReAct] Knowledge base is empty, 0 docs retrieved")
    except Exception as e:
        log.warning(f"[ReAct] Hybrid pipeline failed: {e}, falling back to basic retrieval")
        indexer = get_indexer(state["collection_name"])
        retriever = QdrantHybridRetriever(indexer) if USE_QDRANT else HybridRetriever(indexer)
        docs = retriever.retrieve(query, top_k=5)

    existing = state.get("retrieved_docs", [])
    used = state.get("used_tools", [])
    return {
        "retrieved_docs": existing + docs,
        "used_tools": used + ["vector_search"],
        "history": state.get("history", []) + [f"[ReAct] vector_search: {len(docs)} docs"],
    }


def node_react_exact_match(state: RAGState) -> dict:
    """ReAct: 精确查询"""
    from src.retrieval.agentic_tools import ExactMatchTool
    tool = ExactMatchTool()
    query = state.get("rewritten_query") or state["question"]
    docs = tool.search(query)

    existing = state.get("retrieved_docs", [])
    used = state.get("used_tools", [])
    return {
        "retrieved_docs": existing + docs,
        "used_tools": used + ["exact_match"],
        "history": state.get("history", []) + [f"[ReAct] exact_match: {len(docs)} docs"],
    }


def node_react_graph_search(state: RAGState) -> dict:
    """ReAct: 图谱检索"""
    from src.retrieval.agentic_tools import GraphSearchTool
    tool = GraphSearchTool()
    query = state.get("rewritten_query") or state["question"]
    docs = tool.search(query)

    existing = state.get("retrieved_docs", [])
    used = state.get("used_tools", [])
    return {
        "retrieved_docs": existing + docs,
        "used_tools": used + ["graph_search"],
        "history": state.get("history", []) + [f"[ReAct] graph_search: {len(docs)} docs"],
    }


def node_react_web_search(state: RAGState) -> dict:
    """ReAct: 网络搜索（v2.6：直接使用web_search_fallback，更稳定）"""
    query = state.get("rewritten_query") or state["question"]
    docs = web_search_fallback(query, max_results=3)

    existing = state.get("retrieved_docs", [])
    used = state.get("used_tools", [])
    return {
        "retrieved_docs": existing + docs,
        "used_tools": used + ["web_search"],
        "history": state.get("history", []) + [f"[ReAct] web_search: {len(docs)} docs"],
    }


def node_react_kb_stats(state: RAGState) -> dict:
    """ReAct: 查询系统统计信息（数据量、模型、GPU等）"""
    from src.retrieval.agentic_tools import KBStatsTool
    tool = KBStatsTool()
    raw_docs = tool.search(state["question"])
    docs = [{"doc_id": f"sys_{i}", "content": d.page_content, "source": d.metadata.get("source", "系统"), "page": 0, "metadata": d.metadata} for i, d in enumerate(raw_docs)]

    existing = state.get("retrieved_docs", [])
    used = state.get("used_tools", [])
    return {
        "retrieved_docs": existing + docs,
        "used_tools": used + ["kb_stats"],
        "history": state.get("history", []) + [f"[ReAct] kb_stats: 查询知识库统计"],
    }


def node_react_generate(state: RAGState) -> dict:
    """ReAct: 生成最终答案（v2.6：0条文档时自动Web搜索兜底）"""
    from src.graph import generate_answer  # 兼容测试 mock.patch('src.graph.generate_answer')

    docs = state.get("retrieved_docs", [])

    # v2.6: 0条文档时自动触发Web搜索
    if not docs:
        log.warning("[ReAct] 0 docs at generate — auto-triggering web search")
        try:
            web_results = web_search_fallback(state["question"])
            if web_results:
                docs = web_results
                log.info(f"[ReAct] Web search returned {len(docs)} results")
            else:
                log.error("[ReAct] Web search also returned 0 results")
        except Exception as e:
            log.error(f"[ReAct] Web search failed: {e}")

    # 转换为graded_docs格式
    graded = [
        {
            "doc_id": str(i),
            "content": d.get("content", "") if isinstance(d, dict) else getattr(d, "content", ""),
            "source": d.get("source", "") if isinstance(d, dict) else getattr(d, "source", ""),
            "page": 0,
            "grade": "relevant",
            "relevance_score": 0.8,
            "reasoning": "react_retrieved",
        }
        for i, d in enumerate(docs)
    ]
    prior_context = (state.get("prior_context") or "").strip()
    if prior_context:
        graded = [
            {
                "doc_id": "dialog_prior",
                "content": prior_context[:1200],
                "source": "会话已确认事实",
                "page": 0,
                "grade": "relevant",
                "relevance_score": 1.0,
                "reasoning": "dialog_consistency",
            }
        ] + graded
    result = generate_answer(
        state["question"], graded, prior_context=prior_context
    )

    # v2.6: 设置 relevant_count 和 hallucination_score 供UI计算可信度
    relevant_count = len(graded)
    hallucination_score = result.get("hallucination_score", 0.0)

    # 如果最终还是没有文档，标记高幻觉分数（可信度=0）
    if relevant_count == 0:
        hallucination_score = 1.0
        log.error("[ReAct] Final answer generated with 0 source docs — credibility will be 0%")

    hist = f"[ReAct] Generated {len(result['answer'])} chars from {len(docs)} docs"
    if prior_context:
        hist += " [prior_context]"
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "graded_docs": graded,
        "relevant_count": relevant_count,
        "hallucination_score": hallucination_score,
        "current_step": "react_generated",
        "history": state.get("history", []) + [hist],
    }


def build_agentic_graph() -> StateGraph:
    """构建Agentic RAG图（ReAct循环）

    与build_graph()的区别：
    - build_graph: 固定7层Pipeline（Modular RAG + CRAG）
    - build_agentic_graph: LLM驱动的ReAct循环（Agentic RAG）

    流程：Agent决策 → 选工具执行 → 回到Agent决策（循环）→ 生成
    """
    graph = StateGraph(RAGState)

    # 节点
    graph.add_node("agent_decision", node_agent_decision)
    graph.add_node("vector_search", node_react_vector_search)
    graph.add_node("exact_match", node_react_exact_match)
    graph.add_node("graph_search", node_react_graph_search)
    graph.add_node("web_search", node_react_web_search)
    graph.add_node("kb_stats", node_react_kb_stats)
    graph.add_node("generate", node_react_generate)

    # 入口
    graph.set_entry_point("agent_decision")

    # Agent决策 → 条件路由
    graph.add_conditional_edges("agent_decision", route_react_agent, {
        "vector_search": "vector_search",
        "exact_match": "exact_match",
        "graph_search": "graph_search",
        "web_search": "web_search",
        "kb_stats": "kb_stats",
        "generate": "generate",
    })

    # 每个工具执行后 → 回到Agent决策（形成ReAct循环）
    for tool in ["vector_search", "exact_match", "graph_search", "web_search", "kb_stats"]:
        graph.add_edge(tool, "agent_decision")

    # 生成 → 结束
    graph.add_edge("generate", END)

    return graph


def create_agentic_app():
    """创建Agentic RAG应用（ReAct模式）"""
    graph = build_agentic_graph()
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
