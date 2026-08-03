"""
DeepRAG 主 Pipeline — LangGraph 状态机节点与图构建（v2.3增强版）

从 god-module src/graph.py 抽出：
- 7层 Pipeline 节点（analyze / retrieve / grade / rewrite / web / generate / fact_check / conflicts）
- Corrective RAG + Self-RAG 路由
- build_graph / create_app

注：被测试 mock.patch 的助手名（analyze_query / grade_documents / check_facts /
resolve_conflicts / get_enhanced_retriever / generate_answer）在对应节点内通过
`from src.graph import X` 解析，以保证对 src.graph 的 patch 仍然生效。
"""
import logging
import time

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# 可观测性模块（v2.3新增）
from src.observability.tracer import trace_node, performance_monitor

from src.state import RAGState
from src.config import (
    ENABLE_AGENTIC_RAG,
    ENABLE_SELF_RAG_LOOP,
    SELF_RAG_MAX_REGENERATE,
    USE_LLM_PIPELINE_NODES,
)

# 默认 offline 节点（零 Key 可跑）；USE_LLM_PIPELINE_NODES=true 时切换 LLM 版
if USE_LLM_PIPELINE_NODES:
    from src.agents.query_analyzer import analyze_query
    from src.agents.doc_grader import grade_documents
    from src.agents.fact_checker import check_facts
    from src.agents.conflict_resolver import resolve_conflicts
    _PIPELINE_NODE_MODE = "llm"
else:
    from src.agents.query_analyzer import analyze_query_offline as analyze_query
    from src.agents.doc_grader import grade_documents_offline as grade_documents
    from src.agents.fact_checker import check_facts_offline as check_facts
    from src.agents.conflict_resolver import resolve_conflicts_offline as resolve_conflicts
    _PIPELINE_NODE_MODE = "offline"

from src.agents.doc_grader import check_confidence
from src.retrieval.indexer import Indexer
from src.retrieval.hybrid import HybridRetriever  # 主 Hybrid：HybridRetriever(indexer)
from src.retrieval.web_fallback import web_search_fallback
from src.pipeline.caches import (
    USE_QDRANT,
    QdrantHybridRetriever,
    get_indexer,
    get_agentic_retriever,
    _batch_get_all,
)

log = logging.getLogger("deeprag")


# === 路由函数 ===

def route_after_grading(state: RAGState) -> str:
    """Corrective RAG 路由：委托 pipeline_routing（行为与历史一致）"""
    from src.pipeline_routing import route_after_grading as _route

    return _route(state)


def route_after_fact_check(state: RAGState) -> str:
    """Self-RAG 路由：配置开环 或 校验失败时至少重生 1 次（防错答直接上屏）"""
    from src.pipeline_routing import route_after_fact_check as _route
    from src.config import ENABLE_SELF_RAG_LOOP, SELF_RAG_MAX_REGENERATE

    nxt = _route(state)
    # 质量兜底：即便默认关环，fact_check 失败也允许 1 次 regenerate
    if (
        nxt == "check_conflicts"
        and not state.get("fact_check_passed", True)
        and not state.get("no_knowledge")
        and int(state.get("regenerate_count") or 0) < 1
    ):
        log.info(
            "Self-RAG quality gate: force 1 regenerate (score=%.2f)",
            float(state.get("hallucination_score") or 0),
        )
        return "regenerate"
    if nxt == "regenerate":
        log.info(
            "Self-RAG: regenerate (%s/%s), hallucination=%.2f",
            int(state.get("regenerate_count") or 0) + 1,
            SELF_RAG_MAX_REGENERATE,
            float(state.get("hallucination_score") or 0),
        )
    return nxt


# === 节点函数 ===

@trace_node("analyze_query")
def node_analyze_query(state: RAGState) -> dict:
    """1.查询分析+改写（v2.3：添加追踪）"""
    import time
    start_time = time.time()

    from src.graph import analyze_query  # 兼容测试 mock.patch('src.graph.analyze_query')

    log.info(f"Analyzing query: {state['question'][:50]}")
    result = analyze_query(state["question"])

    # 记录性能指标
    elapsed = (time.time() - start_time) * 1000
    performance_monitor.record("node_analyze_query", elapsed, "ms")

    return {
        "question_type": result["question_type"],
        "rewritten_query": result["rewritten_query"],
        "search_queries": result["search_queries"],
        "current_step": "query_analyzed",
        "history": state.get("history", []) + [f"Query type: {result['question_type']}"],
    }


@trace_node("retrieve")
def node_retrieve(state: RAGState) -> dict:
    """2.检索（v2.3增强版：支持 Enhanced / Agentic / Hybrid 三种模式 + 性能追踪）

    检索模式优先级：
    1. Enhanced模式（v2.2新增，推荐）：问题拒识 + 多路推理 + 重排序 + Web兜底
    2. Agentic模式（v2.1）：动态工具路由 + Agent决策
    3. Hybrid模式（v1.0基线）：BM25 + 向量检索
    """
    import time
    start_time = time.time()

    from src.graph import get_enhanced_retriever  # 兼容测试 mock.patch('src.graph.get_enhanced_retriever')

    query = state["rewritten_query"] or state["question"]
    log.info(f"Retrieving for: {query[:50]}")

    # 模式选择（从环境变量或配置读取，默认Enhanced）
    from src.config import RETRIEVAL_MODE
    mode = getattr(RETRIEVAL_MODE, 'value', 'enhanced')  # enhanced / agentic / hybrid

    # === 模式1：Enhanced检索（v2.7：缓存BM25+文档，避免每次重建）===
    if mode == "enhanced":
        # v2.7: 使用缓存的BM25+文档检索
        try:
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.hybrid_retriever import ParallelHybridRetriever as HybridRetriever
            from src.retrieval.cache import get_cached_documents, get_cached_bm25

            indexer = get_indexer(state["collection_name"])
            col_name = state["collection_name"]

            # v2.7: 缓存文档列表（避免每次HTTP传输数万文档）
            def _fetch_docs():
                import logging as _logging
                _log = _logging.getLogger("deeprag")
                bm25_docs = []

                if USE_QDRANT:
                    # Qdrant: 从检索器获取所有文档
                    try:
                        from qdrant_client.models import ScrollRequest
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
                # v2.7: 缓存BM25索引（避免每次重建）
                bm25 = get_cached_bm25(col_name, lambda: BM25Retriever(bm25_docs))
                if USE_QDRANT:
                    hybrid = QdrantHybridRetriever(indexer)
                else:
                    hybrid = HybridRetriever(bm25, indexer)

                # 如果有HyDE假设答案，也用来检索
                search_query = query
                hyde = state.get("hyde_answer", "")
                if hyde:
                    # 同时用原始query和HyDE答案检索
                    docs = hybrid.search(query, top_k=15)
                    hyde_docs = hybrid.search(hyde, top_k=10)
                    # 合并去重
                    seen_ids = {d.get("doc_id") for d in docs}
                    for d in hyde_docs:
                        if d.get("doc_id") not in seen_ids:
                            docs.append(d)
                            seen_ids.add(d.get("doc_id"))
                else:
                    docs = hybrid.search(query, top_k=15)

                log.info(f"[Hybrid v2.7] BM25+Vector retrieved {len(docs)} docs (cached)")
            else:
                # BM25无文档，降级到纯向量检索
                retriever = get_enhanced_retriever(state["collection_name"])
                result = retriever.retrieve(query, top_k=8, mode="enhanced")
                docs = [
                    {
                        "doc_id": doc.get("doc_id", f"{doc.get('source', 'unknown')}_{doc.get('page', 0)}"),
                        "content": doc.get("content", doc.get("text", "")),
                        "source": doc.get("source", "unknown"),
                        "page": doc.get("page", 0),
                        "metadata": doc.get("metadata", {}),
                        "similarity": doc.get("similarity", 0.0)
                    }
                    for doc in result['results']
                ]

            # v2.6: Rerank精排（v2.8.2: 尊重 ENABLE_RERANKER 配置, v2.8.3: 减少输入到8篇加速CPU模式）
            if docs:
                from src.config import ENABLE_RERANKER
                if ENABLE_RERANKER:
                    try:
                        from src.retrieval.reranker import Reranker
                        reranker = Reranker()
                        # v2.8.3: 只送top-8给reranker，避免CPU模式处理15篇太慢
                        docs = reranker.rerank(query, docs[:8], top_k=5)
                        log.info(f"[Rerank v2.6] Reranked to {len(docs)} docs")
                    except Exception as e:
                        log.warning(f"[Rerank v2.6] Rerank skipped: {e}, using top-5 from hybrid")
                        docs = docs[:5]
                else:
                    docs = docs[:5]
                    log.info(f"[Hybrid v2.8.2] Reranker disabled, using top-5 from RRF")

            history_entry = f"Retrieved {len(docs)} docs (Hybrid+Rerank v2.6)"

        except ImportError as e:
            log.warning(f"Hybrid modules not available: {e}, falling back to Enhanced v2.2")
            # 降级到原有Enhanced检索
            retriever = get_enhanced_retriever(state["collection_name"])
            result = retriever.retrieve(query, top_k=8, mode="enhanced")
            docs = [
                {
                    "doc_id": doc.get("doc_id", f"{doc.get('source', 'unknown')}_{doc.get('page', 0)}"),
                    "content": doc.get("content", doc.get("text", "")),
                    "source": doc.get("source", "unknown"),
                    "page": doc.get("page", 0),
                    "metadata": doc.get("metadata", {}),
                    "similarity": doc.get("similarity", 0.0)
                }
                for doc in result['results']
            ]
            history_entry = f"Retrieved {len(docs)} docs (Enhanced v2.2 fallback)"

    # === 模式2：Agentic检索（v2.1）===
    elif ENABLE_AGENTIC_RAG or mode == "agentic":
        retriever = get_agentic_retriever(state["collection_name"])
        decision = retriever.retrieve_with_decision(query, top_k=8)
        docs = decision["documents"]
        chosen_tool = decision["tool"]
        log.info(f"[Agentic v2.1] Router chose tool='{chosen_tool}', retrieved {len(docs)} docs")
        history_entry = f"Retrieved {len(docs)} docs via {chosen_tool} (Agentic v2.1)"

    # === 模式3：Hybrid检索（v1.0基线）===
    else:
        indexer = get_indexer(state["collection_name"])
        retriever = QdrantHybridRetriever(indexer) if USE_QDRANT else HybridRetriever(indexer)
        docs = retriever.retrieve(query, top_k=8)
        log.info(f"[Hybrid v1.0] Retrieved {len(docs)} documents")
        history_entry = f"Retrieved {len(docs)} docs (Hybrid v1.0)"

    # 记录性能指标（v2.3新增）
    elapsed = (time.time() - start_time) * 1000
    performance_monitor.record("node_retrieve", elapsed, "ms")
    performance_monitor.record("num_retrieved_docs", len(docs), "count")

    return {
        "retrieved_docs": docs,
        "current_step": "retrieved",
        "history": state.get("history", []) + [history_entry],
    }


@trace_node("grade_docs")
def node_grade_docs(state: RAGState) -> dict:
    """3.Corrective RAG文档评分（v2.3：添加追踪 + Human-in-the-Loop置信度判断）"""
    import time
    start_time = time.time()

    from src.graph import grade_documents  # 兼容测试 mock.patch('src.graph.grade_documents')

    log.info("Grading document relevance...")
    graded = grade_documents(state["question"], state["retrieved_docs"])

    # 记录性能指标
    elapsed = (time.time() - start_time) * 1000
    performance_monitor.record("node_grade_docs", elapsed, "ms")

    relevant = sum(1 for d in graded if d["grade"] == "relevant")
    irrelevant = sum(1 for d in graded if d["grade"] == "irrelevant")
    log.info(f"Grading: {relevant} relevant, {len(graded)-relevant-irrelevant} ambiguous, {irrelevant} irrelevant")

    # Human-in-the-Loop: 置信度判断
    need_review = check_confidence(graded)
    if need_review:
        log.warning(f"Low confidence detected (max_score < 0.5), triggering human review")

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
        "need_human_review": need_review,
        "current_step": "graded",
        "history": state.get("history", []) + [f"Graded: {relevant}R/{irrelevant}I, need_review={need_review}"],
    }


def node_rewrite_query(state: RAGState) -> dict:
    """4.查询改写（v2.6：LLM语义改写替代简单追加）"""
    from src.agents.query_analyzer import rewrite_query_for_retry

    retry = state.get("retry_count", 0) + 1
    log.info(f"Rewriting query (attempt {retry})...")

    original = state["rewritten_query"] or state["question"]
    new_query = rewrite_query_for_retry(original, retry)

    return {
        "rewritten_query": new_query,
        "retry_count": retry,
        "current_step": "query_rewritten",
        "history": state.get("history", []) + [f"Query rewritten (#{retry}): {new_query[:30]}"],
    }


def node_web_search(state: RAGState) -> dict:
    """4b.Web搜索兜底（无 API/引擎失败时可能返回 is_mock 占位结果）"""
    log.info("Knowledge base insufficient, falling back to web search...")
    results = web_search_fallback(state["question"]) or []
    used_mock = bool(results) and all(
        (r.get("metadata") or {}).get("is_mock") or (r.get("metadata") or {}).get("engine") == "mock"
        for r in results
    )
    if used_mock:
        log.warning("Web fallback returned MOCK results only (no real search engine)")
    return {
        "web_results": results,
        "retrieval_decision": "web_search",
        "used_web_fallback": True,
        "used_mock_web": used_mock,
        "current_step": "web_searched",
        "history": state.get("history", []) + [
            "Web fallback triggered" + (" [MOCK]" if used_mock else "")
        ],
    }


@trace_node("generate")
def node_generate(state: RAGState) -> dict:
    """5.带引用的答案生成（v2.3：添加追踪；Self-RAG regenerate 时跳过缓存）"""
    import time
    start_time = time.time()

    from src.graph import generate_answer  # 兼容测试 mock.patch('src.graph.generate_answer')

    # Self-RAG 回环：从 fact_check 失败再次进入 generate
    is_regenerate = state.get("current_step") == "fact_checked" and not state.get(
        "fact_check_passed", True
    )
    regenerate_count = int(state.get("regenerate_count", 0) or 0)
    if is_regenerate:
        regenerate_count += 1
        log.info("Generating answer (Self-RAG REGENERATE #%s - skipping cache)...", regenerate_count)
    else:
        log.info("Generating answer with citations...")

    # 使用 relevant+ambiguous 文档；真实 Web 结果可并入；mock Web 不作为证据
    source_docs = [
        d for d in (state.get("graded_docs") or [])
        if d.get("grade") in ("relevant", "ambiguous")
    ]
    # mock Web 不得进入证据链（统一逻辑见 pipeline_routing）
    from src.pipeline_routing import filter_real_web_results

    web_results = state.get("web_results") or []
    real_web = filter_real_web_results(web_results)
    used_mock_web = bool(state.get("used_mock_web")) or (
        bool(web_results) and not real_web
    )
    for wr in real_web:
        source_docs.append({**wr, "grade": "relevant", "relevance_score": 0.5, "reasoning": "web"})

    # 多轮一致性：把 prior_context 作为最高优先证据注入
    prior_context = (state.get("prior_context") or "").strip()
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

    no_knowledge = len(source_docs) == 0
    history_extra_note = ""
    if no_knowledge:
        log.warning("No usable evidence (KB empty + no real web); refusing to hallucinate")
        answer = (
            "【直接回答】知识库与外部检索均未找到可靠依据，无法基于证据回答该问题。\n\n"
            "【详细解释】当前没有相关文档片段"
            + ("；Web 兜底仅返回 mock 占位结果，已忽略。" if used_mock_web else "。")
            + "请补充知识库、配置真实搜索 API（Tavily/Serper）或换一个问题。\n\n"
            "【引用来源】（无）"
        )
        result = {"answer": answer, "citations": []}
        citation_validation = None
    else:
        result = generate_answer(
            state["question"],
            source_docs,
            force_regenerate=is_regenerate,
            prior_context=prior_context,
        )
        # 与前轮堆栈矛盾时强制重生一次
        try:
            from src.agents.dialog_context import find_contradiction, extract_stacks

            # 从 prior 文本构造伪 turns 做冲突检测
            fake_turns = []
            stacks = extract_stacks(prior_context)
            if stacks:
                fake_turns = [{"q": "", "a": " ".join(f"{k}: {v}" for k, v in stacks.items())}]
            contra = find_contradiction(result.get("answer") or "", fake_turns)
            if contra and not is_regenerate:
                code, old_s, new_s = contra
                log.warning(
                    "Dialog stack contradiction %s: prior=%s answer=%s → regenerate",
                    code, old_s, new_s,
                )
                hard = (
                    f"【一致性硬约束】{code} 功能堆栈必须是 {old_s}，"
                    f"禁止写成 {new_s}。请仅使用 {old_s} 回答。\n"
                )
                result = generate_answer(
                    state["question"],
                    source_docs,
                    force_regenerate=True,
                    prior_context=hard + prior_context,
                )
                regenerate_count += 1
                history_extra_note = f" [dialog_consistency_regen {code}]"
            else:
                history_extra_note = ""
        except Exception as e:
            log.debug("dialog consistency check skipped: %s", e)
            history_extra_note = ""
        log.info(
            "Answer generated (%s chars, %s citations)",
            len(result["answer"]),
            len(result["citations"]),
        )
        citation_validation = None
        try:
            from src.agents.citation_validator import get_validator
            validator = get_validator()
            citation_validation = validator.validate(result["answer"], len(source_docs))
            if citation_validation.orphan_claims and citation_validation.citation_rate < 0.5:
                warning = validator.format_warning(citation_validation)
                if warning:
                    result["answer"] = result["answer"] + warning
                    log.warning(
                        "[CitationValidator] %s 条断言未标注引用, 引用率=%.0f%%",
                        len(citation_validation.orphan_claims),
                        citation_validation.citation_rate * 100,
                    )
        except Exception as e:
            log.debug("[CitationValidator] 验证失败（不影响主流程）: %s", e)

    elapsed = (time.time() - start_time) * 1000
    performance_monitor.record("node_generate", elapsed, "ms")
    performance_monitor.record("answer_length", len(result["answer"]), "chars")

    history_extra = f"Generated {len(result['answer'])} chars"
    if no_knowledge:
        history_extra += " [no_knowledge]"
    if is_regenerate:
        history_extra += f" [regenerate#{regenerate_count}]"
    if history_extra_note:
        history_extra += history_extra_note
    if prior_context:
        history_extra += " [prior_context]"

    state_update = {
        "answer": result["answer"],
        "citations": result["citations"],
        "no_knowledge": no_knowledge,
        "used_mock_web": used_mock_web,
        "regenerate_count": regenerate_count,
        "current_step": "generated",
        "history": state.get("history", []) + [history_extra],
    }
    if citation_validation:
        state_update["citation_validation"] = citation_validation.to_dict()
    return state_update


@trace_node("fact_check")
def node_fact_check(state: RAGState) -> dict:
    """6.Self-RAG事实校验（v2.3：添加追踪）"""
    import time
    start_time = time.time()

    from src.graph import check_facts  # 兼容测试 mock.patch('src.graph.check_facts')

    # 无证据拒答：跳过幻觉重试逻辑，直接视为通过
    if state.get("no_knowledge"):
        log.info("Skip fact-check for no_knowledge refusal")
        return {
            "hallucination_score": 0.0,
            "fact_check_passed": True,
            "unsupported_claims": [],
            "current_step": "fact_checked",
            "history": state.get("history", []) + ["Fact check skipped (no_knowledge)"],
        }

    log.info("Fact-checking answer against sources...")
    source_docs = [d for d in (state.get("graded_docs") or []) if d.get("grade") in ("relevant", "ambiguous")]
    result = check_facts(state["answer"], source_docs)

    log.info(
        "Hallucination score: %.2f (%s)",
        result["hallucination_score"],
        "PASS" if result["passed"] else "FAIL",
    )

    elapsed = (time.time() - start_time) * 1000
    performance_monitor.record("node_fact_check", elapsed, "ms")
    performance_monitor.record("hallucination_score", result["hallucination_score"], "score")

    # Corrective 检索重试计数与 Self-RAG regenerate_count 分离；此处只记录校验结果
    return {
        "hallucination_score": result["hallucination_score"],
        "fact_check_passed": result["passed"],
        "unsupported_claims": result["unsupported_claims"],
        "current_step": "fact_checked",
        "history": state.get("history", [])
        + [
            f"Fact check: {result['hallucination_score']:.2f} "
            f"({'PASS' if result['passed'] else 'FAIL'})"
        ],
    }


def node_check_conflicts(state: RAGState) -> dict:
    """7.多源冲突检测"""
    from src.graph import resolve_conflicts  # 兼容测试 mock.patch('src.graph.resolve_conflicts')

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


def node_human_review(state: RAGState) -> dict:
    """Human-in-the-Loop节点 — 低置信度时暂停Pipeline等待人工审核

    使用LangGraph interrupt机制，Pipeline在此处暂停，
    人工审核完成后通过update_state恢复执行。

    人工审核可以：
    1. 补充额外检索关键词
    2. 手动指定相关文档
    3. 直接跳过继续执行
    """
    log.warning("Pipeline paused for human review (low confidence)")
    return {
        "current_step": "human_review",
        "history": state.get("history", []) + ["Paused for human review (low confidence)"],
    }


def route_after_grading_with_hitl(state: RAGState) -> str:
    """Corrective RAG路由（含Human-in-the-Loop）

    路由逻辑：
    1. need_human_review == True → 进入human_review节点（暂停等待人工审核）
    2. need_human_review == False → 进入原有的Corrective RAG路由
    """
    if state.get("need_human_review", False):
        return "human_review"
    # 否则走原有路由逻辑
    return route_after_grading(state)


# === 构建图 ===

def build_graph() -> StateGraph:
    """构建DeepRAG Pipeline（含Human-in-the-Loop）"""
    graph = StateGraph(RAGState)

    graph.add_node("analyze_query", node_analyze_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_docs", node_grade_docs)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("web_search", node_web_search)
    graph.add_node("generate", node_generate)
    graph.add_node("fact_check", node_fact_check)
    graph.add_node("check_conflicts", node_check_conflicts)
    graph.add_node("human_review", node_human_review)

    # 流转
    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_docs")

    # Corrective RAG分支（含Human-in-the-Loop）
    graph.add_conditional_edges("grade_docs", route_after_grading_with_hitl, {
        "human_review": "human_review",
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search",
    })

    # 人工审核完成后继续正常路由（进入原有的Corrective RAG逻辑）
    graph.add_conditional_edges("human_review", route_after_grading, {
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
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
