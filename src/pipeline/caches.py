"""
DeepRAG 全局检索器缓存（从 god-module src/graph.py 抽出）。

- 全局索引器（按 collection_name 区分知识库）
- Agentic RAG 检索器缓存
- 增强检索器缓存（v2.2 新增）

所有模块级缓存只在此处定义一份，并用 threading.Lock 保护，避免并发重建。
"""
import threading
from typing import Optional

import logging

from src.config import (
    ENABLE_AGENTIC_RAG,
    ENABLE_SELF_RAG_LOOP,
    SELF_RAG_MAX_REGENERATE,
    USE_LLM_PIPELINE_NODES,
)
from src.retrieval.indexer import Indexer
from src.retrieval.hybrid import HybridRetriever  # 主 Hybrid：HybridRetriever(indexer)

log = logging.getLogger("deeprag")

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


def _batch_get_all(collection, batch_size: int = 5000) -> dict:
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

# 并发保护：仅此一份定义
_CACHE_LOCK = threading.Lock()


def get_indexer(collection_name: str) -> Indexer:
    """获取索引器（优先使用 Qdrant，回退到 ChromaDB）"""
    with _CACHE_LOCK:
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
    with _CACHE_LOCK:
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

    with _CACHE_LOCK:
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
    with _CACHE_LOCK:
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

    with _CACHE_LOCK:
        _enhanced_retrievers[collection_name] = enhanced
    log.info(f"[Enhanced Retriever v2.2] Created for collection: {collection_name}")
    return enhanced
