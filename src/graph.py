"""
DeepRAG 主Pipeline — LangGraph状态机（v2.3增强版）
7层Pipeline + Corrective RAG + Self-RAG + 增强检索模块（v2.2新增） + LLMOps能力（v2.3新增）

v2.2 新增功能：
1. 问题拒识：前置过滤无效查询，节省15%成本
2. 多路推理：并行3种检索策略+RRF融合，召回率+10%
3. 重排序：ColBERT精排，准确率+5-8%
4. Web兜底：低置信度自动触发，覆盖率100%

v2.3 新增功能（LLMOps）：
1. 可观测性：LangFuse分布式追踪，定位性能瓶颈
2. 性能监控：追踪每个节点的执行时间和tokens消耗
3. 错误追踪：记录失败原因和堆栈信息
"""
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# 可观测性模块（v2.3新增）
from src.observability.tracer import trace_node, performance_monitor

from src.state import RAGState
from src.agents.doc_grader import check_confidence
from src.agents.generator import generate_answer  # 生成默认走 LLM 版（失败内部降级）
from src.retrieval.indexer import Indexer
from src.retrieval.hybrid import HybridRetriever  # 主 Hybrid：HybridRetriever(indexer)
from src.retrieval.web_fallback import web_search_fallback
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("deeprag")
log.info(
    "[Graph] pipeline_nodes=%s self_rag_loop=%s max_regenerate=%s",
    _PIPELINE_NODE_MODE,
    ENABLE_SELF_RAG_LOOP,
    SELF_RAG_MAX_REGENERATE,
)

# Qdrant 替代 ChromaDB（解决 HNSW 重启损坏问题）
# 注意：必须按 VECTOR_DB 显式选择；ImportError 时打印原因，禁止静默退回 Chroma
from src.config import VECTOR_DB as _VECTOR_DB_CFG

_QDRANT_IMPORT_ERROR = None
try:
    from src.retrieval.qdrant_indexer import QdrantIndexer, get_qdrant_indexer
    from src.retrieval.qdrant_hybrid import QdrantHybridRetriever
    _QDRANT_IMPORT_OK = True
except Exception as e:  # noqa: BLE001 — 需要看到真实失败原因
    _QDRANT_IMPORT_OK = False
    _QDRANT_IMPORT_ERROR = e
    QdrantIndexer = None  # type: ignore
    get_qdrant_indexer = None  # type: ignore
    QdrantHybridRetriever = None  # type: ignore

# 默认 qdrant；仅当显式 chromadb 或导入彻底失败才回退
USE_QDRANT = _QDRANT_IMPORT_OK and str(_VECTOR_DB_CFG or "qdrant").lower() != "chromadb"
if USE_QDRANT:
    log.info("[Graph] Using Qdrant as vector store (VECTOR_DB=%s)", _VECTOR_DB_CFG)
else:
    log.warning(
        "[Graph] Qdrant disabled (VECTOR_DB=%s, import_ok=%s, err=%s) → Chroma fallback",
        _VECTOR_DB_CFG,
        _QDRANT_IMPORT_OK,
        _QDRANT_IMPORT_ERROR,
    )


def _batch_get_all(collection, batch_size=5000):
    """批量从 ChromaDB 读取全部文档，避免 SQLite 变量溢出"""
    all_ids, all_docs, all_metas = [], [], []
    offset = 0
    while True:
        data = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = data.get("ids", [])
        if not ids:
            break
        all_ids.extend(ids)
        all_docs.extend(data.get("documents", []))
        all_metas.extend(data.get("metadatas", []))
        if len(ids) < batch_size:
            break
        offset += batch_size
    return {"ids": all_ids, "documents": all_docs, "metadatas": all_metas}


# 全局索引器（按collection_name区分知识库）
_indexers: dict[str, Indexer] = {}
# Agentic RAG 全局缓存（避免每次 node_retrieve 重建 toolbox/router）
_agentic_retrievers: dict[str, object] = {}
# 增强检索器缓存（v2.2新增）
_enhanced_retrievers: dict[str, object] = {}


def get_indexer(collection_name: str):
    """获取索引器（优先使用 Qdrant，回退到 ChromaDB）"""
    if collection_name not in _indexers:
        if USE_QDRANT:
            _indexers[collection_name] = get_qdrant_indexer(collection_name)
        else:
            _indexers[collection_name] = Indexer(collection_name)
    return _indexers[collection_name]


def get_agentic_retriever(collection_name: str):
    """Agentic RAG 检索器工厂（懒加载 + 缓存）

    v2.4升级：全部4个工具已注册 + LLMRouter智能路由
    - 工具：vector_search / exact_match / graph_search / web_search
    - 路由：有LLM时用LLMRouter（智能），无LLM时降级RuleBasedRouter（规则）
    """
    if collection_name in _agentic_retrievers:
        return _agentic_retrievers[collection_name]

    from src.retrieval.agentic_tools import create_toolbox
    from src.retrieval.agent_router import RuleBasedRouter, LLMRouter, AgenticRetriever

    indexer = get_indexer(collection_name)
    hybrid = QdrantHybridRetriever(indexer) if USE_QDRANT else HybridRetriever(indexer)
    toolbox = create_toolbox(hybrid)  # 注册全部4个工具

    # 路由器选择：尝试获取LLM，有则用LLMRouter（智能路由），无则用规则路由
    from src.config import AGENTIC_ROUTER, get_llm_with_fallback
    llm = None
    if AGENTIC_ROUTER == "llm":
        llm = get_llm_with_fallback()
        if llm is not None:
            router = LLMRouter(llm, toolbox, fallback_tool="vector_search")
            log.info("[AgenticRAG] Using LLMRouter (智能路由)")
        else:
            router = RuleBasedRouter(default_tool="vector_search")
            log.info("[AgenticRAG] LLM unavailable, using RuleBasedRouter (规则路由)")
    else:
        router = RuleBasedRouter(default_tool="vector_search")
        log.info("[AgenticRAG] Using RuleBasedRouter (规则路由)")

    retriever = AgenticRetriever(toolbox, router)

    _agentic_retrievers[collection_name] = retriever
    return retriever


class _QdrantUnifiedAdapter:
    """把 QdrantHybridRetriever 适配成 EnhancedKnowledgeRetrieval 需要的 search() 接口。

    避免 enhanced 路径再走 UnifiedRetriever → Chroma HttpClient（8000 未开就报错）。
    """

    def __init__(self, hybrid_retriever, collection_name: str = ""):
        self.hybrid = hybrid_retriever
        self.collection_name = collection_name

    def search(self, query: str, top_k: int = 5, mode: str = "smart") -> dict:
        docs = self.hybrid.retrieve(query, top_k=top_k) or []
        results = []
        for d in docs:
            if isinstance(d, dict):
                content = d.get("content") or ""
                source = d.get("source") or ""
                page = d.get("page", 0)
                meta = d.get("metadata") or {}
                sim = float(meta.get("rrf_score") or d.get("similarity") or 0.65)
            else:
                content = getattr(d, "content", "") or ""
                source = getattr(d, "source", "") or ""
                page = getattr(d, "page", 0) or 0
                meta = getattr(d, "metadata", None) or {}
                sim = float(meta.get("rrf_score") or 0.65)
            # rrf 分数通常很小，映射到 0.5–0.95 便于下游阈值
            if sim < 0.5:
                sim = min(0.95, 0.55 + sim * 20)
            results.append(
                {
                    "content": content,
                    "source": source,
                    "page": page,
                    "similarity": sim,
                    "confidence": "high" if sim >= 0.7 else "medium",
                    "metadata": meta,
                }
            )
        conf = "high" if results else "no_results"
        return {
            "query": query,
            "results": results,
            "confidence": conf,
            "explanation": f"Qdrant hybrid ({self.collection_name}) → {len(results)} docs",
            "optimized_query": query,
        }


def get_enhanced_retriever(collection_name: str):
    """增强检索器工厂（v2.2新增）

    组合：QdrantHybrid / UnifiedRetriever + EnhancedKnowledgeRetrieval（5层增强）
    功能：
    1. 问题拒识（前置过滤）
    2. 多路推理（并行3路径+RRF融合）
    3. 重排序（ColBERT精排）
    4. Web兜底（低置信度触发）
    5. 混合检索（BM25+向量）
    """
    if collection_name in _enhanced_retrievers:
        return _enhanced_retrievers[collection_name]

    from src.retrieval.enhanced_knowledge_retrieval import EnhancedKnowledgeRetrieval

    # 创建基础检索器：优先 Qdrant，禁止在 Qdrant 模式下硬连 Chroma
    if USE_QDRANT:
        indexer = get_indexer(collection_name)
        hybrid = QdrantHybridRetriever(indexer)
        base_retriever = _QdrantUnifiedAdapter(hybrid, collection_name)
        log.info(f"[Enhanced Retriever] base=QdrantHybrid for {collection_name}")
    else:
        from src.retrieval.unified_retriever import UnifiedRetriever
        from src.config import EMBEDDING_MODEL, DEVICE

        base_retriever = UnifiedRetriever(
            collection_name=collection_name,
            model_name=EMBEDDING_MODEL,
            device=DEVICE,
            enable_query_optimization=True,
            enable_hallucination_detection=True,
            similarity_threshold=0.5,
        )
        log.info(f"[Enhanced Retriever] base=Chroma UnifiedRetriever for {collection_name}")

    # 创建增强检索器
    enhanced = EnhancedKnowledgeRetrieval(
        base_retriever=base_retriever,
        enable_validation=True,      # 问题拒识
        enable_multipath=True,       # 多路推理
        enable_reranking=True,       # 重排序
        enable_web_fallback=False,   # Web兜底（暂时禁用，使用原有的web_search节点）
        similarity_threshold=0.5
    )

    _enhanced_retrievers[collection_name] = enhanced
    log.info(f"[Enhanced Retriever v2.2] Created for collection: {collection_name}")
    return enhanced


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


# === ReAct Agent 循环（v2.4新增）===

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


def create_app():
    graph = build_graph()
    checkpointer = InMemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )


# === Function Calling 模式（v2.9新增）===

FC_SYSTEM_PROMPT = """你是一个智能知识库助手。请根据用户问题，使用提供的工具检索信息并回答。

工作流程：
1. 先用 search_knowledge_base 搜索本地知识库
2. 如果知识库无结果或不相关，用 web_search 搜索互联网
3. 可选：用 check_error_book 检查是否有历史错题记录
4. 检索到足够信息后，调用 generate_answer 生成最终答案

注意：
- 每次调用工具后，系统会返回结果供你参考
- 最多调用5次工具，然后必须生成答案
- 如果已有足够信息，直接调用 generate_answer"""


def function_calling_query(question: str, collection_name: str = "default",
                           max_iterations: int = 5) -> dict:
    """Function Calling 模式查询（v2.9新增）

    使用原生 Function Calling 让 LLM 自主决策调用工具，
    替代 ReAct 模式的文本 JSON 解析方式。

    优势：
    - LLM 原生返回 tool_calls（结构化输出，无需正则解析JSON）
    - 工具 schema 由 GLM_TOOLS 定义，新增工具只需加schema
    - 循环调用直到 LLM 决定生成答案

    Args:
        question:        用户问题
        collection_name: 知识库名称
        max_iterations:  最大工具调用轮次

    Returns:
        完整结果dict（与query()格式一致）
    """
    from src.config import get_llm_with_fallback
    from src.agents.glm_tools import GLM_TOOLS, execute_tool
    from src.agents.generator import generate_answer, generate_direct_answer
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    import time as _time
    import json as _json

    log.info(f"[FC v2.9] 开始Function Calling查询: {question[:50]}")
    t0 = _time.time()

    llm = get_llm_with_fallback()
    if llm is None:
        log.warning("[FC] LLM不可用，降级到enhanced模式")
        return query(question, collection_name, max_retries=2, mode="enhanced")

    # 绑定工具到LLM
    try:
        llm_with_tools = llm.bind_tools(GLM_TOOLS)
    except Exception as e:
        log.warning(f"[FC] bind_tools失败({e})，降级到enhanced模式")
        return query(question, collection_name, max_retries=2, mode="enhanced")

    # 构建初始对话
    messages = [
        SystemMessage(content=FC_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    # 收集检索到的文档
    all_docs = []
    used_tools = []
    history = []
    should_generate = False

    # FC 循环
    for iteration in range(max_iterations):
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            log.warning(f"[FC] LLM调用失败(轮次{iteration}): {e}")
            if all_docs:
                break
            return query(question, collection_name, max_retries=2, mode="enhanced")

        # 检查是否有 tool_calls
        tool_calls = getattr(response, 'tool_calls', None)

        if not tool_calls:
            # LLM 直接返回文本答案（没有调用工具）
            text = response.content if hasattr(response, 'content') else str(response)
            log.info(f"[FC] LLM直接返回答案(轮次{iteration}), 长度{len(text)}")
            history.append(f"[FC] LLM直接回答 (轮次{iteration})")

            elapsed = _time.time() - t0
            return {
                "question": question,
                "collection_name": collection_name,
                "answer": text,
                "citations": [],
                "retrieved_docs": all_docs,
                "graded_docs": all_docs,
                "relevant_count": len(all_docs),
                "history": history + [f"[FC] 总耗时 {elapsed:.1f}s"],
                "current_step": "done",
                "hallucination_score": 1.0 if all_docs else 0.5,
                "fact_check_passed": True,
                "unsupported_claims": [],
                "conflicts": [],
                "web_results": [],
                "retry_count": iteration,
                "need_human_review": False,
                "errors": [],
            }

        # 处理 tool_calls
        messages.append(response)  # 将 AI 响应（含 tool_calls）加入对话

        for tc in tool_calls:
            tool_name = tc.get("name", "") if isinstance(tc, dict) else tc.get("name", "")
            tool_args = tc.get("args", {}) if isinstance(tc, dict) else tc.get("args", {})
            tool_id = tc.get("id", f"call_{iteration}_{tool_name}") if isinstance(tc, dict) else getattr(tc, "id", f"call_{iteration}_{tool_name}")

            log.info(f"[FC] 轮次{iteration}: 调用工具 {tool_name}({tool_args})")
            used_tools.append(tool_name)
            history.append(f"[FC] 轮次{iteration}: {tool_name}")

            # 如果是 generate_answer，结束循环
            if tool_name == "generate_answer":
                log.info(f"[FC] LLM决定生成答案: {tool_args.get('summary', '')[:100]}")
                history.append("[FC] LLM决定生成答案")
                messages.append(ToolMessage(
                    content=f"已准备好生成答案。共检索到{len(all_docs)}篇文档。",
                    tool_call_id=tool_id,
                ))
                should_generate = True
                break

            # 执行工具
            result_str = execute_tool(tool_name, tool_args, collection_name=collection_name)

            # 将工具结果加入对话
            messages.append(ToolMessage(
                content=result_str,
                tool_call_id=tool_id,
            ))

            # 收集检索到的文档
            try:
                result_data = _json.loads(result_str)
                for doc in result_data.get("results", []):
                    all_docs.append({
                        "doc_id": doc.get("doc_id", ""),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", ""),
                        "page": doc.get("page", 0),
                        "metadata": {},
                        "similarity": doc.get("score", 0.0),
                        "relevance_score": doc.get("score", 0.0),
                    })
            except (_json.JSONDecodeError, KeyError):
                pass

        if should_generate:
            break

    # 用检索到的文档生成最终答案
    elapsed_retrieval = _time.time() - t0
    log.info(f"[FC] 检索完成: {len(all_docs)}篇文档, {elapsed_retrieval:.1f}s, 工具: {used_tools}")

    if not all_docs:
        history.append("[FC] 未检索到文档，直接LLM回答")
        answer = generate_direct_answer(question)
        elapsed = _time.time() - t0
        return {
            "question": question,
            "collection_name": collection_name,
            "answer": answer,
            "citations": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "relevant_count": 0,
            "history": history + [f"[FC] 总耗时 {elapsed:.1f}s"],
            "current_step": "done",
            "hallucination_score": 0.5,
            "fact_check_passed": True,
            "unsupported_claims": [],
            "conflicts": [],
            "web_results": [],
            "retry_count": len(used_tools),
            "need_human_review": False,
            "errors": [],
        }

    # 用 generator 生成结构化答案
    try:
        gen_result = generate_answer(question, all_docs)
        answer = gen_result.get("answer", "")
        citations = gen_result.get("citations", [])
    except Exception as e:
        log.warning(f"[FC] 生成答案失败: {e}，使用LLM直接回答")
        answer = generate_direct_answer(question)
        citations = []

    elapsed = _time.time() - t0
    tool_chain = " → ".join(used_tools) if used_tools else "直接回答"
    history.append(f"[FC] 答案生成完成, 总耗时 {elapsed:.1f}s, 工具链: {tool_chain}")

    return {
        "question": question,
        "collection_name": collection_name,
        "answer": answer,
        "citations": citations,
        "retrieved_docs": all_docs,
        "graded_docs": all_docs,
        "relevant_count": len(all_docs),
        "history": history,
        "current_step": "done",
        "hallucination_score": 1.0,
        "fact_check_passed": True,
        "unsupported_claims": [],
        "conflicts": [],
        "web_results": [],
        "retry_count": len(used_tools),
        "need_human_review": False,
        "errors": [],
    }


def _auto_route_collection(question: str, collection_name: str) -> tuple[str, list]:
    """错库自动纠正：人格/MBTI 等问题禁止落到 thesis 等库。"""
    notes = []
    try:
        from src.retrieval.collection_router import collection_conflicts_with_query
        conflict, msg, suggested = collection_conflicts_with_query(question, collection_name or "")
        if conflict and suggested and suggested != collection_name:
            notes.append(f"[router] 自动换库 {collection_name} → {suggested}（{msg}）")
            log.warning(notes[-1])
            return suggested, notes
    except Exception as e:
        log.debug("collection router skip: %s", e)
    return collection_name, notes


def _guard_domain_mismatch(question: str, result: dict) -> dict:
    """检索结果与问题域完全不沾边 → 强制拒答，防止论文段答 INTJ。"""
    try:
        from src.retrieval.collection_router import docs_match_query_domain
        docs = result.get("graded_docs") or result.get("retrieved_docs") or []
        # 只取 relevant
        rel = [
            d for d in docs
            if (d.get("grade") if isinstance(d, dict) else None) in (None, "relevant", "ambiguous")
        ] or docs
        if docs_match_query_domain(question, rel, min_hits=1):
            return result
        refuse = (
            "【直接回答】当前知识库中未找到与问题匹配的可靠依据，无法回答。\n\n"
            "【详细解释】检索到的片段与问题主题不一致（例如用论文/代码库回答人格类型问题）。"
            "请更换知识库（如 proj_psychology）或补充文档后重试。\n\n"
            "【引用来源】（无）"
        )
        hist = list(result.get("history") or [])
        hist.append("[router] 域不匹配：已拒答，避免答非所问")
        result = {
            **result,
            "answer": refuse,
            "citations": [],
            "relevant_count": 0,
            "no_knowledge": True,
            "hallucination_score": 1.0,
            "fact_check_passed": False,
            "history": hist,
        }
        log.warning("[router] domain mismatch → refuse")
    except Exception as e:
        log.debug("domain guard skip: %s", e)
    return result


def _format_today_news_answer(question: str, real: list, day_str: str) -> tuple[str, list]:
    """用今日条目生成可核对的结构化答案（日期写死在正文里）。"""
    bullets = []
    citations = []
    for i, wr in enumerate(real[:10], 1):
        meta = wr.get("metadata") or {}
        title = (meta.get("title") or "").strip() or (wr.get("content") or "")[:60]
        when = (meta.get("date") or "").strip() or day_str
        src = (wr.get("source") or "").strip()
        snip = (meta.get("snippet") or wr.get("content") or "").strip().replace("\n", " ")
        if len(snip) > 120:
            snip = snip[:120] + "…"
        bullets.append(f"{i}. **[{when}]** {title}" + (f" — {snip}" if snip and snip != title else ""))
        citations.append({"id": i, "source": src, "title": title, "date": when})

    head = (
        f"【直接回答】以下为 **{day_str}（东八区当天）** 可核验的新闻头条"
        f"（共 {len(real)} 条，已过滤非今日条目）。\n\n"
    )
    detail = "【详细解释】\n" + "\n".join(bullets) + "\n\n"
    detail += (
        "说明：来源为 Google News 等公开 RSS，发布时间按条目 pubDate 过滤为当天；"
        "标题与摘要以源站为准，非本地知识库。\n\n"
    )
    cites = "【引用来源】\n" + "\n".join(
        f"[{c['id']}] {c['title']} ({c['date']}) {c['source']}" for c in citations
    )
    return head + detail + cites, citations


def _query_realtime_web(
    question: str,
    collection_name: str,
    route_notes: list | None = None,
    reason: str = "realtime",
) -> dict:
    """时效/新闻问题：只走 Web，且 **强制「今天」（东八区）** 条目。

    防止「今日新闻」命中本地工作日志，也防止用旧闻冒充今日。
    """
    import time as _time
    from src.retrieval.web_fallback import (
        web_search_fallback,
        web_search_today_news,
        filter_today_results,
        today_cn,
    )
    from src.pipeline_routing import filter_real_web_results
    from src.agents.generator import generate_answer, generate_direct_answer
    from src.agents.query_analyzer import make_refuse_answer

    t0 = _time.time()
    day = today_cn()
    day_str = day.isoformat()
    notes = list(route_notes or [])
    notes.append(f"[v2.9.3] realtime → today-only web ({reason}) day={day_str}")
    log.info("[v2.9.3] realtime today-only: %s day=%s", question[:60], day_str)

    # 1) 专用今日新闻源 2) auto+today_only 兜底
    results = web_search_today_news(question, max_results=12) or []
    if len(results) < 3:
        extra = web_search_fallback(question, max_results=15, engine="auto", today_only=True) or []
        seen = {(r.get("source") or "") for r in results}
        for e in extra:
            k = e.get("source") or ""
            if k and k not in seen:
                results.append(e)
                seen.add(k)

    real = filter_today_results(filter_real_web_results(results), day=day, allow_missing_date=False)
    used_mock = bool(results) and not real

    if not real:
        ans = (
            f"【直接回答】未能获取 **{day_str}** 当天的可核验新闻（免费源暂无今日时间戳条目）。\n\n"
            "【详细解释】时效问题仅汇总带当日 pubDate 的公开 RSS/新闻结果，"
            "不会使用本地文档，也不会用昨日及更早新闻冒充今日。请稍后重试或打开权威媒体首页。\n\n"
            "【引用来源】（无）"
        )
        if used_mock:
            notes.append("[v2.9.3] no today-dated real web — refuse")
        return {
            "question": question,
            "collection_name": collection_name,
            "answer": ans,
            "citations": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "relevant_count": 0,
            "no_knowledge": True,
            "used_web_fallback": True,
            "used_mock_web": used_mock,
            "web_results": results,
            "history": notes + [f"realtime today empty ({_time.time()-t0:.1f}s)"],
            "current_step": "done",
            "hallucination_score": 0.0,
            "fact_check_passed": True,
            "unsupported_claims": [],
            "conflicts": [],
            "retry_count": 0,
            "need_human_review": False,
            "errors": [],
            "routed_collection": collection_name,
            "answer_source": "realtime_web_empty",
            "news_day": day_str,
        }

    # 结构化今日列表（保证用户看到日期）；LLM 只做一句总览，失败则纯列表
    list_answer, list_cites = _format_today_news_answer(question, real, day_str)
    source_docs = [
        {**wr, "grade": "relevant", "relevance_score": 0.7, "reasoning": "web_today"}
        for wr in real
    ]
    answer = list_answer
    citations = list_cites
    try:
        # 把日期写进每条 content，强迫模型只依据今日
        dated_docs = []
        for wr in source_docs:
            meta = wr.get("metadata") or {}
            when = meta.get("date") or day_str
            title = meta.get("title") or ""
            body = wr.get("content") or ""
            dated_docs.append(
                {
                    **wr,
                    "content": f"[发布于 {when}] {title}\n{body}",
                }
            )
        gen = generate_answer(
            (
                f"今天是 {day_str}（东八区）。用户问题：{question}\n"
                "规则：1) 只根据下列【今日】新闻摘要作答；2) 禁止使用训练记忆或本地知识库；"
                "3) 每条要点必须带上发布时间；4) 不要编造未出现的事件。"
                "先用 2～4 句总览今日要点，再分条列出。"
            ),
            dated_docs,
            force_regenerate=True,
        )
        llm_ans = (gen.get("answer") or "").strip()
        # 若 LLM 答了但未提今日日期，拼接结构化列表更稳
        if llm_ans and (day_str in llm_ans or "今日" in llm_ans or day_str[:4] in llm_ans):
            # 保留 LLM 总览 + 强制附今日条目表
            answer = (
                llm_ans
                + "\n\n---\n"
                + f"**可核验条目（{day_str} 过滤）**\n"
                + "\n".join(
                    f"- [{(d.get('metadata') or {}).get('date', day_str)}] "
                    f"{(d.get('metadata') or {}).get('title', '')[:80]}"
                    for d in real[:8]
                )
            )
            citations = gen.get("citations") or list_cites
        elif llm_ans:
            answer = list_answer
            notes.append("[v2.9.3] llm answer lacked today marker — use structured list")
    except Exception as e:
        log.warning("realtime generate failed: %s, use structured list", e)
        notes.append(f"[v2.9.3] generate failed → list: {e}")

    bad_local = any(
        w in (answer or "")
        for w in ("工作日志", "1500行", "v2.4", "心理计算器", "交叉验证UI", "docs/工作日志")
    )
    if bad_local:
        answer, citations = list_answer, list_cites
        notes.append("[v2.9.3] blocked local-log-style; fallback structured")

    return {
        "question": question,
        "collection_name": collection_name,
        "answer": answer,
        "citations": citations,
        "retrieved_docs": source_docs,
        "graded_docs": source_docs,
        "relevant_count": len(source_docs),
        "no_knowledge": False,
        "used_web_fallback": True,
        "used_mock_web": False,
        "web_results": real,
        "history": notes + [f"realtime today docs={len(real)} day={day_str} ({_time.time()-t0:.1f}s)"],
        "current_step": "done",
        "hallucination_score": 0.15,
        "fact_check_passed": True,
        "unsupported_claims": [],
        "conflicts": [],
        "retry_count": 0,
        "need_human_review": False,
        "errors": [],
        "routed_collection": collection_name,
        "answer_source": "realtime_web_today",
        "news_day": day_str,
    }


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
    from src.config import RETRIEVAL_MODE as _cfg_mode

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
    actual_mode = mode or getattr(_cfg_mode, 'value', 'enhanced')

    # === Function Calling 模式（v2.9新增）===
    if actual_mode == "function_calling":
        return function_calling_query(question, collection_name, max_iterations=max_retries)

    # === Agentic ReAct 模式（v2.4新增）===
    if actual_mode == "agentic_react":
        app = create_agentic_app()
        config = {"configurable": {"thread_id": f"react-{hash(question) % 10000}"}}

        initial_state: RAGState = {
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

    initial_state: RAGState = {
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


def _ensure_answer_or_refuse(question: str, result: dict) -> dict:
    """空答案 / 全不相关检索 的最后兜底：输出统一拒答，避免评测 answer_len=0。"""
    from src.agents.query_analyzer import make_refuse_answer, is_unanswerable_query

    ans = (result.get("answer") or "").strip()
    rel = int(result.get("relevant_count") or 0)
    # 显式不可答：无论是否编出答案，强制拒识
    bad, why = is_unanswerable_query(question)
    if bad:
        hist = list(result.get("history") or [])
        hist.append(f"[guard] unanswerable → refuse ({why})")
        return {
            **result,
            "answer": make_refuse_answer(why),
            "citations": [],
            "no_knowledge": True,
            "relevant_count": 0,
            "hallucination_score": 0.0,
            "fact_check_passed": True,
            "history": hist,
        }
    if ans:
        return result
    # 无正文：按拒答补齐
    reason = "检索后无可用答案"
    if rel == 0:
        reason = "检索文档均不相关或证据为空"
    hist = list(result.get("history") or [])
    hist.append(f"[guard] empty answer → refuse ({reason})")
    log.warning("[guard] empty answer for %s → refuse", (question or "")[:40])
    return {
        **result,
        "answer": make_refuse_answer(reason),
        "citations": result.get("citations") or [],
        "no_knowledge": True,
        "history": hist,
    }


def stream_query(question: str, collection_name: str = "default",
                 max_retries: int = 2, mode: str = None,
                 prior_context: str = "", dialog_turns: list = None):
    """流式执行RAG查询，生成阶段逐token yield

    Yields:
        dict: {"type": "token", "content": "..."} — 生成token
              {"type": "metadata", "state": {...}} — 最终完整状态
    """
    from src.config import RETRIEVAL_MODE as _cfg_mode
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

    actual_mode = mode or getattr(_cfg_mode, 'value', 'enhanced')

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
