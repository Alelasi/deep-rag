"""Web Fallback — 知识库无答案时搜索外部兜底"""


def web_search_fallback(query: str, max_results: int = 3) -> list[dict]:
    """
    外部搜索兜底（当知识库检索结果全部irrelevant时触发）
    生产环境接Tavily/Serper API，当前用mock
    """
    # TODO: 接真实搜索API
    return [{
        "doc_id": f"web_{i}",
        "content": f"[Web搜索结果占位] 查询: {query}",
        "source": "web_search",
        "page": 0,
        "metadata": {"is_web": True},
    } for i in range(max_results)]
