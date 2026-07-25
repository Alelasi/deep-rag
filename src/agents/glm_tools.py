"""Function Calling 工具定义与执行器（v2.9: 接入主Pipeline）

定义 OpenAI 兼容格式的 Function Calling 工具 schema，
并提供 execute_tool 函数将工具调用分发到具体实现。

v2.9 改进：
  - search_knowledge_base 支持 collection_name 参数，复用 graph.py 缓存检索器
  - 新增 generate_answer 工具，让 LLM 可主动决定生成最终答案
  - execute_tool 支持 collection_name 上下文传递

工具清单：
  1. search_knowledge_base — 本地知识库语义检索（复用缓存Pipeline）
  2. web_search            — 互联网搜索兜底
  3. check_error_book      — 错题集历史记录检查
  4. generate_answer       — 生成最终答案（检索结果足够时调用）

用法示例::

    from src.agents.glm_tools import GLM_TOOLS, execute_tool

    # 1. 将 GLM_TOOLS 传给 LLM 的 tools 参数
    response = llm.invoke(messages, tools=GLM_TOOLS)

    # 2. 解析 response 中的 tool_calls，调用 execute_tool 执行
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call["name"], tool_call["args"], collection_name="default")
"""
import json
import logging

log = logging.getLogger("deeprag.glm_tools")


# ======================================================================
#  工具 Schema 定义（OpenAI Function Calling 格式）
# ======================================================================

GLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在本地知识库中搜索与查询相关的文档片段。"
                "使用 BM25 关键词检索 + 向量语义检索的混合模式（RRF 融合），"
                "适用于知识库内已有文档的技术问题、概念解释等查询。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本，可以是关键词或自然语言问题",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的最相关文档数量",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "在互联网上搜索最新信息，作为本地知识库的兜底补充。"
                "适用于知识库无匹配、需要时效性信息、或需要交叉验证的场景。"
                "默认使用 DuckDuckGo 搜索引擎（免费，无需 API Key）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "网络搜索查询文本",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_error_book",
            "description": (
                "检查错题集历史记录，查找与当前问题相似的历史错题。"
                "返回历史错误类型和修正提示，帮助规避同类错误。"
                "适用于在检索前预判可能出错的问题类型。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检查的问题文本",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_answer",
            "description": (
                "基于已检索到的文档内容，生成最终答案。"
                "当检索结果足够回答用户问题时调用此工具。"
                "调用后系统会使用检索到的文档作为上下文，生成结构化回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "对检索结果的简要总结，说明为什么这些文档足以回答问题",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


# ======================================================================
#  工具执行器
# ======================================================================

def _execute_search_knowledge_base(query: str, top_k: int = 5,
                                   collection_name: str = "default") -> str:
    """执行知识库检索（v2.9: 复用 graph.py 缓存检索器）

    通过 graph.py 的缓存 Pipeline（get_indexer + get_cached_bm25）进行混合检索，
    避免每次创建新 Indexer 实例。

    Args:
        query:           查询文本
        top_k:           返回结果数量
        collection_name: 知识库集合名称

    Returns:
        JSON 格式的检索结果字符串
    """
    try:
        # v2.9: 复用 graph.py 的缓存检索器，避免重复创建 Indexer
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.hybrid_retriever import ParallelHybridRetriever as HybridRetriever
        from src.retrieval.cache import get_cached_documents, get_cached_bm25
        from src.retrieval.indexer import Indexer

        indexer = Indexer(collection_name)
        all_collections = indexer.get_all_collections()
        bm25_docs = []
        for col in all_collections:
            try:
                col_data = col.get(include=["documents", "metadatas"])
                for i, (doc_text, meta) in enumerate(zip(
                    col_data.get("documents", []),
                    col_data.get("metadatas", [])
                )):
                    bm25_docs.append({
                        "doc_id": col_data.get("ids", [f"doc_{i}"])[i] if i < len(col_data.get("ids", [])) else f"doc_{i}",
                        "content": doc_text,
                        "source": meta.get("source", "unknown"),
                        "page": meta.get("page", 0),
                        "metadata": meta,
                    })
            except Exception as e:
                log.warning(f"[FC] 子集合 {col.name} 读取失败: {e}")

        if not bm25_docs:
            return json.dumps({"results": [], "message": "知识库为空"}, ensure_ascii=False)

        bm25 = BM25Retriever(bm25_docs)
        hybrid = HybridRetriever(bm25, indexer)
        docs = hybrid.search(query, top_k=top_k)

        if not docs:
            log.info("知识库检索无结果: query='%s'", query[:80])
            return json.dumps({"results": [], "message": "知识库未找到相关文档"}, ensure_ascii=False)

        results = []
        for doc in docs:
            results.append({
                "doc_id": doc.get("doc_id", ""),
                "content": doc.get("content", "")[:500],
                "source": doc.get("source", ""),
                "page": doc.get("page", 0),
                "score": doc.get("metadata", {}).get("rrf_score", 0),
            })

        log.info("知识库检索成功: query='%s...' → %d 条结果", query[:40], len(results))
        return json.dumps({"results": results}, ensure_ascii=False)

    except Exception as e:
        log.error("知识库检索失败: %s", e)
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)


def _execute_web_search(query: str) -> str:
    """执行网络搜索

    调用 web_search_fallback（位于 src.retrieval.web_fallback）进行互联网搜索，
    默认使用 DuckDuckGo 引擎。

    Args:
        query: 搜索查询文本

    Returns:
        JSON 格式的搜索结果字符串
    """
    try:
        # 注: web_search_fallback 定义在 src.retrieval.web_fallback 模块
        from src.retrieval.web_fallback import web_search_fallback

        results = web_search_fallback(query, max_results=3, engine="duckduckgo")

        if not results:
            log.info("网络搜索无结果: query='%s'", query[:80])
            return json.dumps({"results": [], "message": "网络搜索未找到结果"}, ensure_ascii=False)

        formatted = []
        for r in results:
            formatted.append({
                "doc_id": r.get("doc_id", ""),
                "content": r.get("content", "")[:500],
                "source": r.get("source", ""),
                "title": r.get("metadata", {}).get("title", ""),
            })

        log.info("网络搜索成功: query='%s...' → %d 条结果", query[:40], len(formatted))
        return json.dumps({"results": formatted}, ensure_ascii=False)

    except Exception as e:
        log.error("网络搜索失败: %s", e)
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)


def _execute_check_error_book(query: str) -> str:
    """检查错题集历史记录

    调用 ErrorBook.get_correction_hint 获取与当前问题相似的历史错题及修正提示。

    Args:
        query: 要检查的问题文本

    Returns:
        JSON 格式的错题集检查结果字符串
    """
    try:
        from src.agents.error_book import ErrorBook

        book = ErrorBook()
        hint = book.get_correction_hint(query)

        if not hint:
            log.info("错题集无匹配: query='%s'", query[:80])
            return json.dumps(
                {"has_history": False, "hint": "", "message": "无历史错题记录"},
                ensure_ascii=False,
            )

        log.info("错题集匹配成功: query='%s...'", query[:40])
        return json.dumps(
            {"has_history": True, "hint": hint},
            ensure_ascii=False,
        )

    except Exception as e:
        log.error("错题集检查失败: %s", e)
        return json.dumps({"error": str(e), "has_history": False}, ensure_ascii=False)


def execute_tool(tool_name: str, args: dict, collection_name: str = "default") -> str:
    """根据工具名执行对应的工具逻辑（v2.9: 优先通过注册中心执行）

    这是 Function Calling 的统一入口。LLM 返回 tool_call 后，
    调用此函数执行实际操作并获取结果字符串。

    v2.9 改进：
    - 优先通过 ToolRegistry.execute() 执行（带安全校验）
    - 如果注册中心未注册该工具，降级为直接执行（向后兼容）

    Args:
        tool_name:        工具名称
        args:             工具参数字典
        collection_name:  知识库集合名称（用于 search_knowledge_base）

    Returns:
        工具执行结果（JSON 格式字符串）
    """
    log.info("执行工具: %s, 参数: %s, 集合: %s", tool_name, args, collection_name)

    # v2.9: 优先通过注册中心执行（带安全校验）
    try:
        from src.tools.tool_registry import get_registry
        registry = get_registry()
        # 检查工具是否已注册
        try:
            registry.get(tool_name)
            # 工具已注册，通过注册中心执行（带安全校验）
            result = registry.execute(tool_name, collection_name=collection_name, **args)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except KeyError:
            # 工具未注册，降级为直接执行
            pass
    except ImportError:
        pass

    # 降级：直接执行（向后兼容）
    if tool_name == "search_knowledge_base":
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        return _execute_search_knowledge_base(query, top_k, collection_name=collection_name)

    elif tool_name == "web_search":
        query = args.get("query", "")
        return _execute_web_search(query)

    elif tool_name == "check_error_book":
        query = args.get("query", "")
        return _execute_check_error_book(query)

    elif tool_name == "generate_answer":
        summary = args.get("summary", "")
        log.info("[FC] LLM决定生成答案: %s", summary[:100])
        return json.dumps({"action": "generate", "summary": summary}, ensure_ascii=False)

    else:
        error_msg = f"未知工具: {tool_name}，可用工具: search_knowledge_base, web_search, check_error_book, generate_answer"
        log.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)
