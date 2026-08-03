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

------------------------------------------------------------------------------
本文件为「薄转发层」：所有实现已拆分到以下子模块，公开 API 符号集合保持不变：

- src/pipeline/caches.py       全局索引器/检索器缓存（含 threading.Lock）
- src/pipeline/build.py        Pipeline 节点与图构建（build_graph 等）
- src/pipeline/run.py          编排入口（query / stream_query / batch_query / precision_query）
- src/rag/react.py             ReAct 循环
- src/rag/function_calling.py  Function-Calling 逻辑
- src/rag/guards.py            实时新闻守卫、域守卫
------------------------------------------------------------------------------
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# 可观测性模块（v2.3新增）
from src.observability.tracer import trace_node, performance_monitor

from src.state import RAGState
from src.agents.generator import generate_answer  # 生成默认走 LLM 版（失败内部降级）
from src.agents.doc_grader import check_confidence
from src.retrieval.indexer import Indexer
from src.retrieval.hybrid import HybridRetriever  # 主 Hybrid：HybridRetriever(indexer)
from src.retrieval.web_fallback import web_search_fallback
from src.config import (
    ENABLE_AGENTIC_RAG,
    ENABLE_SELF_RAG_LOOP,
    SELF_RAG_MAX_REGENERATE,
    USE_LLM_PIPELINE_NODES,
)

# === 缓存与检索器工厂（单一定义，含 threading.Lock）===
from src.pipeline.caches import (
    _batch_get_all,
    _indexers,
    _agentic_retrievers,
    _enhanced_retrievers,
    _CACHE_LOCK,
    _QdrantUnifiedAdapter,
    get_indexer,
    get_agentic_retriever,
    get_enhanced_retriever,
    USE_QDRANT,
    QdrantIndexer,
    get_qdrant_indexer,
    QdrantHybridRetriever,
    _QDRANT_IMPORT_OK,
    _QDRANT_IMPORT_ERROR,
    _VECTOR_DB_CFG,
)

# === Pipeline 节点与图构建 ===
from src.pipeline.build import (
    _PIPELINE_NODE_MODE,
    analyze_query,
    grade_documents,
    check_facts,
    resolve_conflicts,
    route_after_grading,
    route_after_fact_check,
    node_analyze_query,
    node_retrieve,
    node_grade_docs,
    node_rewrite_query,
    node_web_search,
    node_generate,
    node_fact_check,
    node_check_conflicts,
    node_human_review,
    route_after_grading_with_hitl,
    build_graph,
    create_app,
)

# === ReAct 循环 ===
from src.rag.react import (
    REACT_PROMPT,
    _summarize_docs,
    _parse_json_response,
    node_agent_decision,
    route_react_agent,
    node_react_vector_search,
    node_react_exact_match,
    node_react_graph_search,
    node_react_web_search,
    node_react_kb_stats,
    node_react_generate,
    build_agentic_graph,
    create_agentic_app,
)

# === Function Calling 模式 ===
from src.rag.function_calling import (
    FC_SYSTEM_PROMPT,
    function_calling_query,
)

# === 守卫逻辑（实时新闻 / 域守卫）===
from src.rag.guards import (
    _auto_route_collection,
    _guard_domain_mismatch,
    _format_today_news_answer,
    _query_realtime_web,
    _ensure_answer_or_refuse,
)

# === 编排入口 ===
from src.pipeline.run import (
    query,
    stream_query,
    batch_query,
    _precision_re_search,
    precision_query,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("deeprag")
log.info(
    "[Graph] pipeline_nodes=%s self_rag_loop=%s max_regenerate=%s",
    _PIPELINE_NODE_MODE,
    ENABLE_SELF_RAG_LOOP,
    SELF_RAG_MAX_REGENERATE,
)


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
