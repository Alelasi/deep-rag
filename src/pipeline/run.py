"""
DeepRAG 编排入口（从 god-module src/graph.py 抽出）

- query / stream_query / batch_query：对外主入口
- precision_query / _precision_re_search：v2.8.5 精准双Agent模式

注意：function_calling_query 对 query 的回退已在 function_calling.py 内惰性导入，
避免与本项目产生循环依赖。
"""
import logging
from typing import Optional

from src.config import RETRIEVAL_MODE
from src.pipeline.caches import (
    get_indexer,
    get_enhanced_retriever,
    USE_QDRANT,
    QdrantHybridRetriever,
    _batch_get_all,
)
from src.retrieval.hybrid import HybridRetriever  # 主 Hybrid：HybridRetriever(indexer)

from src.pipeline.build import create_app
from src.rag.react import create_agentic_app
from src.rag.function_calling import function_calling_query
from src.rag.guards import (
    _auto_route_collection,
    _query_realtime_web,
    _guard_domain_mismatch,
    _ensure_answer_or_refuse,
)

log = logging.getLogger("deeprag")


def query(question: str, collection_name: str = "default",
          max_retries: int = 2, mode: str = None,
          prior_context: str = "", dialog_turns: list = None) -> dict:
    """执行一次RAG查询，返回完整结果

    Args:
        question: 用户问题
        collection_name: 知识库名称
        max_retries: 最大重试/检索轮次
        mode: 检索模式（None=按配置 / "enhanced" / "agentic" / "hybrid" / "agentic_react"）
        prior_context: 相关前轮摘要（可空）
        dialog_turns: [{"q","a"}, ...] 若给 prior 为空则自动生成
    """
    # 多轮一致性：由 turns 生成 prior_context
    if not prior_context and dialog_turns:
        try:
            from src.agents.dialog_context import build_prior_context, consistency_hint
            prior_context = (
                build_prior_context(question, dialog_turns)
                + consistency_hint(question, dialog_turns)
            )
        except Exception as e:
            log.debug("build prior_context failed: %s", e)
            prior_context = ""
    prior_context = (prior_context or "").strip()

    route_notes: list = []
    collection_name, route_notes = _auto_route_collection(question, collection_name)
    if prior_context:
        route_notes = list(route_notes) + ["[dialog] injected prior_context for multi-turn consistency"]

    # v2.9.2: 时效/新闻问题 → 强制 Web，禁止本地库（防「今日新闻」命中工作日志）
    from src.agents.query_analyzer import needs_rag, make_refuse_answer, is_realtime_query
    is_rt, rt_why = is_realtime_query(question)
    if is_rt:
        return _query_realtime_web(question, collection_name, route_notes, rt_why)

    # v2.8.3: 常识/闲聊跳过RAG；v2.9.1: 不可答题直接拒识
    rag_needed, skip_reason = needs_rag(question)
    if not rag_needed:
        import time as _time
        _start = _time.time()
        # 荒诞/虚构/知识库探测：统一拒答，禁止 LLM 硬编
        if str(skip_reason).startswith("不可答"):
            log.info(f"[v2.9.1] 拒识（{skip_reason}）: {question[:50]}")
            _elapsed = _time.time() - _start
            return {
                "question": question,
                "collection_name": collection_name,
                "answer": make_refuse_answer(skip_reason),
                "citations": [],
                "retrieved_docs": [],
                "graded_docs": [],
                "relevant_count": 0,
                "no_knowledge": True,
                "history": route_notes + [f"[v2.9.1] {skip_reason}，拒识 ({_elapsed:.1f}s)"],
                "current_step": "done",
                "hallucination_score": 0.0,
                "fact_check_passed": True,
                "unsupported_claims": [],
                "conflicts": [],
                "web_results": [],
                "retry_count": 0,
                "need_human_review": False,
                "errors": [],
                "routed_collection": collection_name,
            }
        log.info(f"[v2.8.3] 跳过RAG（{skip_reason}），直接LLM回答: {question[:50]}")
        from src.agents.generator import generate_direct_answer
        answer = generate_direct_answer(question)
        _elapsed = _time.time() - _start
        return {
            "question": question,
            "collection_name": collection_name,
            "answer": answer,
            "citations": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "relevant_count": 0,
            "history": route_notes + [f"[v2.8.3] {skip_reason}，跳过RAG直接回答 ({_elapsed:.1f}s)"],
            "current_step": "done",
            "hallucination_score": 1.0,
            "fact_check_passed": True,
            "unsupported_claims": [],
            "conflicts": [],
            "web_results": [],
            "retry_count": 0,
            "need_human_review": False,
            "errors": [],
            "routed_collection": collection_name,
        }

    # 如果指定了mode，临时覆盖配置
    actual_mode = mode or getattr(RETRIEVAL_MODE, 'value', 'enhanced')

    # === Function Calling 模式（v2.9新增）===
    if actual_mode == "function_calling":
        return function_calling_query(question, collection_name, max_iterations=max_retries)

    # === Agentic ReAct 模式（v2.4新增）===
    if actual_mode == "agentic_react":
        app = create_agentic_app()
        config = {"configurable": {"thread_id": f"react-{hash(question) % 10000}"}}

        initial_state = {
            "question": question,
            "collection_name": collection_name,
            "prior_context": prior_context,
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
            "need_human_review": False,
            "errors": [],
            "history": [],
            "next_action": "",
            "agent_reason": "",
            "retrieval_round": 0,
            "used_tools": [],
        }

        initial_state["history"] = list(route_notes)
        for event in app.stream(initial_state, config=config):
            pass

        state = app.get_state(config)
        out = dict(state.values or {})
        out["collection_name"] = collection_name
        out["routed_collection"] = collection_name
        if route_notes:
            out["history"] = route_notes + list(out.get("history") or [])
        out = _guard_domain_mismatch(question, out)
        return _ensure_answer_or_refuse(question, out)

    # === 原有Pipeline模式 ===
    app = create_app()
    config = {"configurable": {"thread_id": f"rag-{hash(question) % 10000}"}}

    initial_state = {
        "question": question,
        "collection_name": collection_name,
        "prior_context": prior_context,
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
        "need_human_review": False,
        "errors": [],
        "history": list(route_notes),
    }

    for event in app.stream(initial_state, config=config):
        pass  # 静默执行

    state = app.get_state(config)
    out = dict(state.values or {})
    out["collection_name"] = collection_name
    out["routed_collection"] = collection_name
    if route_notes:
        out["history"] = route_notes + list(out.get("history") or [])
    out = _guard_domain_mismatch(question, out)
    return _ensure_answer_or_refuse(question, out)


def stream_query(question: str, collection_name: str = "default",
                 max_retries: int = 2, mode: str = None,
                 prior_context: str = "", dialog_turns: list = None):
    """流式执行RAG查询，生成阶段逐token yield

    Yields:
        dict: {"type": "token", "content": "..."} — 生成token
              {"type": "metadata", "state": {...}} — 最终完整状态
    """
    from src.agents.generator import generate_answer_stream, generate_direct_answer_stream
    from src.agents.query_analyzer import analyze_query_offline as analyze_query, needs_rag
    from src.agents.doc_grader import grade_documents_offline as grade_documents, check_confidence
    from src.agents.fact_checker import check_facts_offline as check_facts
    from src.agents.conflict_resolver import resolve_conflicts_offline as resolve_conflicts
    from src.retrieval.web_fallback import web_search_fallback

    if not prior_context and dialog_turns:
        try:
            from src.agents.dialog_context import build_prior_context, consistency_hint
            prior_context = (
                build_prior_context(question, dialog_turns)
                + consistency_hint(question, dialog_turns)
            )
        except Exception:
            prior_context = ""
    prior_context = (prior_context or "").strip()

    actual_mode = mode or getattr(RETRIEVAL_MODE, 'value', 'enhanced')

    # v2.9.2 时效新闻 + v2.8.3/v2.9.1 闲聊/拒识
    from src.agents.query_analyzer import make_refuse_answer, is_realtime_query
    is_rt, rt_why = is_realtime_query(question)
    if is_rt:
        out = _query_realtime_web(question, collection_name, [], rt_why)
        ans = out.get("answer") or ""
        if ans:
            yield {"type": "token", "content": ans}
        yield {"type": "metadata", "state": out}
        return

    rag_needed, skip_reason = needs_rag(question)
    if not rag_needed:
        import time as _time
        _start = _time.time()
        if str(skip_reason).startswith("不可答"):
            log.info(f"[Stream v2.9.1] 拒识（{skip_reason}）: {question[:50]}")
            ans = make_refuse_answer(skip_reason)
            yield {"type": "token", "content": ans}
            _elapsed = _time.time() - _start
            yield {
                "type": "metadata",
                "state": {
                    "question": question,
                    "collection_name": collection_name,
                    "answer": ans,
                    "citations": [],
                    "retrieved_docs": [],
                    "graded_docs": [],
                    "relevant_count": 0,
                    "no_knowledge": True,
                    "history": [f"[v2.9.1] {skip_reason}，拒识 ({_elapsed:.1f}s)"],
                    "current_step": "done",
                    "hallucination_score": 0.0,
                    "fact_check_passed": True,
                    "unsupported_claims": [],
                    "conflicts": [],
                    "web_results": [],
                    "retry_count": 0,
                    "need_human_review": False,
                    "errors": [],
                },
            }
            return
        log.info(f"[Stream v2.8.3] 跳过RAG（{skip_reason}），直接LLM流式回答: {question[:50]}")
        answer_parts = []
        for token in generate_direct_answer_stream(question):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
        _elapsed = _time.time() - _start
        yield {
            "type": "metadata",
            "state": {
                "question": question,
                "collection_name": collection_name,
                "answer": "".join(answer_parts),
                "citations": [],
                "retrieved_docs": [],
                "graded_docs": [],
                "relevant_count": 0,
                "history": [f"[v2.8.3] {skip_reason}，跳过RAG直接回答 ({_elapsed:.1f}s)"],
                "current_step": "done",
                "hallucination_score": 1.0,
                "fact_check_passed": True,
                "unsupported_claims": [],
                "conflicts": [],
                "web_results": [],
                "retry_count": 0,
                "need_human_review": False,
                "errors": [],
            },
        }
        return

    # Function Calling模式不支持流式，降级为普通query
    if actual_mode == "function_calling":
        result = function_calling_query(question, collection_name, max_iterations=max_retries)
        yield {"type": "token", "content": result.get("answer", "")}
        yield {"type": "metadata", "state": result}
        return

    # ReAct模式不支持流式，降级为普通query
    if actual_mode == "agentic_react":
        result = query(question, collection_name, max_retries, mode)
        yield {"type": "token", "content": result.get("answer", "")}
        yield {"type": "metadata", "state": result}
        return

    # 1. 查询分析
    log.info(f"[Stream] Analyzing query: {question[:50]}")
    analyze_result = analyze_query(question)
    rewritten_query = analyze_result["rewritten_query"] or question

    # 2. 检索
    query_to_retrieve = rewritten_query
    if actual_mode == "enhanced":
        retriever = get_enhanced_retriever(collection_name)
        retrieve_result = retriever.retrieve(query_to_retrieve, top_k=8, mode="enhanced")
        docs = [
            {
                "text": doc.get("text", ""),
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", 0),
                "metadata": doc.get("metadata", {}),
                "similarity": doc.get("similarity", 0.0),
                "content": doc.get("text", doc.get("content", "")),
                "doc_id": str(i),
            }
            for i, doc in enumerate(retrieve_result['results'])
        ]
    else:
        indexer = get_indexer(collection_name)
        retriever_obj = QdrantHybridRetriever(indexer) if USE_QDRANT else HybridRetriever(indexer)
        docs = retriever_obj.retrieve(query_to_retrieve, top_k=8)

    log.info(f"[Stream] Retrieved {len(docs)} docs")

    # 3. 文档评分
    graded = grade_documents(question, docs)
    relevant_count = sum(1 for d in graded if d["grade"] == "relevant")
    source_docs = [d for d in graded if d["grade"] in ("relevant", "ambiguous")]

    log.info(f"[Stream] Graded: {relevant_count} relevant")

    # 4. 如果无相关文档，尝试web兜底
    web_results = []
    if relevant_count == 0 and max_retries > 0:
        web_results = web_search_fallback(question)
        for wr in web_results:
            source_docs.append({**wr, "grade": "relevant", "relevance_score": 0.5, "reasoning": "web"})

    # 5. 流式生成（注入多轮 prior）
    if prior_context:
        source_docs = [
            {
                "doc_id": "dialog_prior",
                "content": prior_context[:1200],
                "source": "会话已确认事实",
                "page": 0,
                "grade": "relevant",
                "relevance_score": 1.0,
                "reasoning": "dialog_consistency",
            }
        ] + source_docs
    log.info(f"[Stream] Generating answer (streaming)... prior={bool(prior_context)}")
    answer_parts = []
    for token in generate_answer_stream(
        question, source_docs, prior_context=prior_context
    ):
        answer_parts.append(token)
        yield {"type": "token", "content": token}

    answer = "".join(answer_parts)
    log.info(f"[Stream] Answer generated ({len(answer)} chars)")

    # 6. 事实校验
    fact_result = check_facts(answer, source_docs)

    # 7. 冲突检测
    relevant_graded = [d for d in graded if d["grade"] == "relevant"]
    conflicts = resolve_conflicts(question, relevant_graded)

    # 8. 提取引用
    citations = []
    for doc in source_docs:
        source = doc.get("source", "")
        page = doc.get("page", 0)
        if source and (source in answer or f"第{page}块" in answer):
            citations.append({
                "text": doc.get("content", "")[:200],
                "source": source,
                "page": page,
            })

    # 9. 构造完整 state
    final_state = {
        "question": question,
        "collection_name": collection_name,
        "question_type": analyze_result.get("question_type", "factual"),
        "rewritten_query": rewritten_query,
        "search_queries": analyze_result.get("search_queries", []),
        "retrieved_docs": docs,
        "graded_docs": graded,
        "relevant_count": relevant_count,
        "irrelevant_count": sum(1 for d in graded if d["grade"] == "irrelevant"),
        "retrieval_decision": "generate" if relevant_count >= 1 else "web_search",
        "answer": answer,
        "citations": citations,
        "hallucination_score": fact_result["hallucination_score"],
        "fact_check_passed": fact_result["passed"],
        "unsupported_claims": fact_result["unsupported_claims"],
        "conflicts": conflicts,
        "web_results": web_results,
        "current_step": "completed",
        "retry_count": 0,
        "max_retries": max_retries,
        "need_human_review": check_confidence(graded),
        "errors": [],
        "history": [
            f"Query type: {analyze_result.get('question_type', 'unknown')}",
            f"Retrieved {len(docs)} docs",
            f"Graded: {relevant_count} relevant",
            f"Generated {len(answer)} chars",
            f"Fact check: {fact_result['hallucination_score']:.2f}",
        ],
        "used_tools": [],
        "agent_reason": "",
        "retrieval_round": 0,
    }

    yield {"type": "metadata", "state": final_state}


def batch_query(questions: list[str], collection_name: str = "default",
                max_retries: int = 2, mode: str = None,
                max_workers: int = 1) -> list[dict]:
    """串行批量查询 — 逐个处理，避免LLM并发导致429（v2.8更新）

    v2.6: 使用ThreadPoolExecutor并行处理（max_workers=4）
    v2.8: 改为串行处理（max_workers=1），配合全局LLM锁
          保证一次只有一个请求到LLM，彻底消除429

    Args:
        questions: 问题列表
        collection_name: 知识库名称
        max_retries: 最大重试次数
        mode: 检索模式
        max_workers: 固定为1（v2.8：串行模式，避免429）

    Returns:
        按输入顺序排列的结果列表
    """
    results = []

    for i, question in enumerate(questions):
        log.info(f"[BatchQuery v2.8] Processing {i+1}/{len(questions)}: {question[:50]}")
        try:
            result = query(question, collection_name=collection_name,
                          max_retries=max_retries, mode=mode)
            results.append(result)
        except Exception as e:
            log.error(f"[BatchQuery] Question #{i} failed: {e}")
            results.append({
                "question": question,
                "answer": f"查询失败：{e}",
                "error": str(e),
                "current_step": "error",
            })

    return results


# === v2.8.5: 精准模式（双Agent并行+矛盾检测+重新搜索）===

def _precision_re_search(question: str, conflict_points: list[str],
                         collection_name: str = "default") -> list:
    """精准模式重新搜索 — 基于矛盾点构造新查询检索

    将矛盾点作为额外关键词加入搜索，检索与矛盾相关的文档。
    """
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import ParallelHybridRetriever as NewHybrid
    from src.retrieval.cache import get_cached_documents, get_cached_bm25

    indexer = get_indexer(collection_name)

    # 将矛盾点拼接成额外搜索词
    conflict_query = f"{question} {' '.join(conflict_points[:3])}"

    try:
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
                        _log.warning(f"[Precision re-search] 子集合 {col.name} 读取失败: {e}")
            return bm25_docs

        bm25_docs = get_cached_documents(collection_name, _fetch_docs)

        if bm25_docs:
            bm25 = get_cached_bm25(collection_name, lambda: BM25Retriever(bm25_docs))
            if USE_QDRANT:
                hybrid = QdrantHybridRetriever(indexer)
            else:
                hybrid = NewHybrid(bm25, indexer)
            docs = hybrid.search(conflict_query, top_k=8)

            # Rerank
            if docs:
                from src.config import ENABLE_RERANKER
                if ENABLE_RERANKER:
                    try:
                        from src.retrieval.reranker import Reranker
                        reranker = Reranker()
                        docs = reranker.rerank(conflict_query, docs[:8], top_k=5)
                    except Exception:
                        docs = docs[:5]
                else:
                    docs = docs[:5]

            log.info(f"[Precision re-search] 检索到 {len(docs)} 篇新文档")
            return docs
    except Exception as e:
        log.error(f"[Precision re-search] 失败: {e}")

    return []


def precision_query(question: str, collection_name: str = "default",
                    max_retries: int = 2,
                    model_a: str = "THUDM/GLM-Z1-9B-0414",
                    model_b: str = "glm-4-flash",
                    compare_model: str = "glm-4-flash",
                    strategy_a: str = "socratic",
                    strategy_b: str = "concise",
                    fast_mode: bool = True) -> dict:
    """精准模式查询 — 双Agent并行回答+矛盾检测+重新搜索

    v2.8.6最优配置（Harness测试验证）:
    - 模型: Z1+Flash交叉（准确率7.93/10, 速度6.33s）
    - 策略: socratic+concise（准确率8.9/10, 速度6.04s）
    - 快速模式: 本地启发式检测（省2-3s LLM对比）
    - 矛盾时: 返回双答案让用户自行分辨

    Args:
        question: 用户问题
        collection_name: 知识库名称
        max_retries: 最大重试次数

    Returns:
        完整结果dict（含双Agent对比元数据）
    """
    from src.agents.query_analyzer import analyze_query_offline as analyze_query, needs_rag
    from src.agents.doc_grader import grade_documents_offline as grade_documents
    from src.agents.fact_checker import check_facts_offline as check_facts
    from src.agents.conflict_resolver import resolve_conflicts_offline as resolve_conflicts
    from src.agents.dual_agent import precision_generate
    from src.retrieval.web_fallback import web_search_fallback
    import time as _time

    log.info(f"[Precision v2.8.5] 开始精准模式查询: {question[:50]}")
    t0 = _time.time()

    # v2.8.3: 常识/闲聊问题跳过RAG
    rag_needed, skip_reason = needs_rag(question)
    if not rag_needed:
        log.info(f"[Precision] 跳过RAG（{skip_reason}），直接LLM回答")
        from src.agents.generator import generate_direct_answer
        answer = generate_direct_answer(question)
        return {
            "question": question,
            "collection_name": collection_name,
            "answer": answer,
            "citations": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "relevant_count": 0,
            "history": [f"[Precision] {skip_reason}，跳过RAG直接回答"],
            "current_step": "done",
            "hallucination_score": 1.0,
            "fact_check_passed": True,
            "unsupported_claims": [],
            "conflicts": [],
            "web_results": [],
            "retry_count": 0,
            "need_human_review": False,
            "errors": [],
            "answer_a": answer,
            "answer_b": answer,
            "verdict": "skipped",
            "conflict_points": [],
            "recommendation": "none",
            "re_searched": False,
        }

    # 1. 查询分析
    analyze_result = analyze_query(question)
    rewritten_query = analyze_result["rewritten_query"] or question

    # 2. 检索（复用Enhanced模式）
    query_to_retrieve = rewritten_query
    try:
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.hybrid_retriever import ParallelHybridRetriever as NewHybrid
        from src.retrieval.cache import get_cached_documents, get_cached_bm25

        indexer = get_indexer(collection_name)

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
                        _log.warning(f"[Precision] 子集合 {col.name} 读取失败: {e}")
            return bm25_docs

        bm25_docs = get_cached_documents(collection_name, _fetch_docs)

        if bm25_docs:
            bm25 = get_cached_bm25(collection_name, lambda: BM25Retriever(bm25_docs))
            if USE_QDRANT:
                hybrid = QdrantHybridRetriever(indexer)
            else:
                hybrid = NewHybrid(bm25, indexer)
            docs = hybrid.search(query_to_retrieve, top_k=15)

            if docs:
                from src.config import ENABLE_RERANKER
                if ENABLE_RERANKER:
                    try:
                        from src.retrieval.reranker import Reranker
                        reranker = Reranker()
                        docs = reranker.rerank(query_to_retrieve, docs[:8], top_k=5)
                    except Exception:
                        docs = docs[:5]
                else:
                    docs = docs[:5]

            log.info(f"[Precision] 检索到 {len(docs)} 篇文档")
        else:
            docs = []
    except Exception as e:
        log.warning(f"[Precision] 检索失败: {e}")
        docs = []

    # 3. 文档评分
    graded = grade_documents(question, docs)
    relevant_count = sum(1 for d in graded if d["grade"] == "relevant")
    source_docs = [d for d in graded if d["grade"] in ("relevant", "ambiguous")]

    log.info(f"[Precision] 评分: {relevant_count} 篇相关")

    # 4. 无文档时Web兜底
    web_results = []
    if relevant_count == 0:
        web_results = web_search_fallback(question)
        for wr in web_results:
            source_docs.append({**wr, "grade": "relevant", "relevance_score": 0.5, "reasoning": "web"})

    # 5. 双Agent精准生成
    def _re_search_fn(q, conflict_points):
        return _precision_re_search(q, conflict_points, collection_name)

    # v2.8.6: 使用Harness验证的最优配置
    precision_result = precision_generate(
        question, source_docs, re_search_fn=_re_search_fn,
        model_a=model_a,
        model_b=model_b,
        compare_model=compare_model,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        fast_mode=fast_mode,
        show_both_on_conflict=True,
    )

    # 6. 事实校验
    fact_result = check_facts(precision_result["answer"], source_docs)

    # 7. 冲突检测
    relevant_graded = [d for d in graded if d["grade"] == "relevant"]
    conflicts = resolve_conflicts(question, relevant_graded)

    elapsed = round(_time.time() - t0, 2)
    log.info(f"[Precision] 完成: verdict={precision_result['verdict']}, "
             f"re_searched={precision_result['re_searched']}, {elapsed}s")

    return {
        "question": question,
        "collection_name": collection_name,
        "question_type": analyze_result.get("question_type", "factual"),
        "rewritten_query": rewritten_query,
        "search_queries": analyze_result.get("search_queries", []),
        "retrieved_docs": docs,
        "graded_docs": graded,
        "relevant_count": relevant_count,
        "irrelevant_count": sum(1 for d in graded if d["grade"] == "irrelevant"),
        "retrieval_decision": "generate" if relevant_count >= 1 else "web_search",
        "answer": precision_result["answer"],
        "citations": precision_result["citations"],
        "hallucination_score": fact_result["hallucination_score"],
        "fact_check_passed": fact_result["passed"],
        "unsupported_claims": fact_result["unsupported_claims"],
        "conflicts": conflicts,
        "web_results": web_results,
        "current_step": "precision_completed",
        "retry_count": 0,
        "max_retries": max_retries,
        "need_human_review": False,
        "errors": [],
        "history": [
            f"Query type: {analyze_result.get('question_type', 'unknown')}",
            f"Retrieved {len(docs)} docs",
            f"Graded: {relevant_count} relevant",
        ] + precision_result["history"] + [
            f"Fact check: {fact_result['hallucination_score']:.2f}",
            f"Total: {elapsed}s",
        ],
        # 精准模式专属元数据
        "answer_a": precision_result["answer_a"],
        "answer_b": precision_result["answer_b"],
        "strategy_a": precision_result.get("strategy_a", "direct"),
        "strategy_b": precision_result.get("strategy_b", "analytical"),
        "elapsed_a": precision_result.get("elapsed_a", 0),
        "elapsed_b": precision_result.get("elapsed_b", 0),
        "verdict": precision_result["verdict"],
        "conflict_points": precision_result["conflict_points"],
        "recommendation": precision_result["recommendation"],
        "re_searched": precision_result["re_searched"],
        "show_both": precision_result.get("show_both", False),
    }
