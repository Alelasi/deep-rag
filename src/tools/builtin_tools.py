"""
内置工具集（白名单）— v2.9: 统一工具注册

注册的工具：
  1. search_database        — 只读SQL查询（原有）
  2. search_knowledge_base  — 知识库语义检索（从 glm_tools.py 迁移）
  3. web_search             — 互联网搜索（从 glm_tools.py 迁移）
  4. check_error_book       — 错题集检查（从 glm_tools.py 迁移）
  5. generate_answer        — 生成最终答案（v2.9新增）

安全机制：
  - 所有工具必须通过 ToolRisk.SAFE 白名单检查
  - SQL查询有三层防护（语法白名单 + 关键词黑名单 + 自动限制）
  - 知识库检索为只读操作
"""

from typing import List, Dict, Any
import re
import logging
from .tool_registry import Tool, ToolCategory, ToolRisk, get_registry

log = logging.getLogger("deeprag")


# ============================================================================
# 工具1: 只读SQL查询（三层防护）
# ============================================================================

def search_database(sql_query: str) -> List[Dict]:
    """
    搜索数据库（只读SQL查询）

    三层安全防护：
    1. 语法白名单：只允许SELECT + 简单WHERE/ORDER/LIMIT
    2. 关键词黑名单：禁止任何写操作
    3. 自动限制：强制LIMIT 100，超时5秒
    """
    query_lower = sql_query.lower().strip()

    if not query_lower.startswith("select"):
        raise ValueError("❌ 只允许SELECT查询")

    allowed_keywords = ["select", "from", "where", "and", "or", "order", "by", "limit", "as", "join", "on", "in", "like", "between"]
    tokens = re.findall(r'\b\w+\b', query_lower)
    sql_keywords = [t for t in tokens if t in ["select", "from", "where", "insert", "update", "delete", "drop", "alter", "create", "truncate", "replace", "merge"]]

    if not all(kw in allowed_keywords for kw in sql_keywords):
        raise ValueError("❌ 发现非法SQL关键词")

    dangerous = ["insert", "update", "delete", "drop", "truncate", "alter", "create", "replace", "merge", "exec", "execute", "call", "grant", "revoke", "into", "set"]
    for kw in dangerous:
        if re.search(rf'\b{kw}\b', query_lower):
            raise ValueError(f"❌ 禁止使用 {kw.upper()} 操作")

    if "limit" not in query_lower:
        sql_query = sql_query.rstrip(";") + " LIMIT 100"
    else:
        limit_match = re.search(r'limit\s+(\d+)', query_lower)
        if limit_match and int(limit_match.group(1)) > 100:
            sql_query = re.sub(r'limit\s+\d+', 'LIMIT 100', sql_query, flags=re.IGNORECASE)

    # 实际执行SQL（只读连接 + 5秒超时）
    import sqlite3
    import json
    import os

    # 安全：限制数据库路径为项目内或指定路径
    db_path = os.environ.get("DEEPRAG_DB_PATH", "data/deeprag.db")
    if not os.path.exists(db_path):
        return [{"error": f"数据库文件不存在: {db_path}"}]

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA query_only = ON")  # 只读模式
        cursor = conn.execute(sql_query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()[:100]  # 限制100行
        conn.close()

        result = [dict(zip(columns, row)) for row in rows]
        return result if result else [{"info": "查询结果为空"}]
    except sqlite3.Error as e:
        return [{"error": f"SQL执行错误: {str(e)}"}]
    except Exception as e:
        return [{"error": f"查询异常: {str(e)}"}]


# ============================================================================
# 工具2-5: 知识库工具（从 glm_tools.py 迁移，v2.9统一注册）
# ============================================================================

def _search_knowledge_base(query: str, top_k: int = 5, collection_name: str = "default") -> str:
    """知识库语义检索（只读操作）"""
    from src.agents.glm_tools import _execute_search_knowledge_base
    return _execute_search_knowledge_base(query, top_k, collection_name=collection_name)


def _web_search(query: str) -> str:
    """互联网搜索（只读操作）"""
    from src.agents.glm_tools import _execute_web_search
    return _execute_web_search(query)


def _check_error_book(query: str) -> str:
    """错题集检查（只读操作）"""
    from src.agents.glm_tools import _execute_check_error_book
    return _execute_check_error_book(query)


def _generate_answer(summary: str) -> str:
    """生成最终答案"""
    import json
    return json.dumps({"action": "generate", "summary": summary}, ensure_ascii=False)


# ============================================================================
# 工具注册（白名单 - 统一注册所有工具）
# ============================================================================

def register_builtin_tools():
    """
    注册所有内置工具到白名单注册中心（v2.9: 统一注册）

    工具清单：
      1. search_database        — 只读SQL查询
      2. search_knowledge_base  — 知识库语义检索
      3. web_search             — 互联网搜索
      4. check_error_book       — 错题集检查
      5. generate_answer        — 生成最终答案
    """
    registry = get_registry()

    # --- 工具1: 只读SQL查询 ---
    tool_db = Tool(
        name="search_database",
        description=(
            "搜索数据库（只读SQL查询）。"
            "⚠️ 只允许SELECT查询，禁止INSERT/UPDATE/DELETE。"
            "⚠️ 自动限制最多100条结果。"
        ),
        category=ToolCategory.SEARCH,
        risk=ToolRisk.SAFE,
        function=search_database,
        parameters={
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "SQL查询语句（只允许SELECT）。示例：'SELECT * FROM users WHERE age > 18'"
                }
            },
            "required": ["sql_query"]
        },
        examples=[
            "search_database('SELECT * FROM users LIMIT 10')",
        ]
    )

    # --- 工具2: 知识库语义检索 ---
    tool_kb = Tool(
        name="search_knowledge_base",
        description=(
            "搜索本地知识库（只读语义检索）。"
            "使用 BM25 + 向量混合检索，返回最相关的文档片段。"
            "适用于回答知识库中已有的问题。"
        ),
        category=ToolCategory.RETRIEVAL,
        risk=ToolRisk.SAFE,
        function=_search_knowledge_base,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量（默认5）",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        examples=[
            "search_knowledge_base(query='INTJ的主导功能', top_k=5)",
        ]
    )

    # --- 工具3: 互联网搜索 ---
    tool_web = Tool(
        name="web_search",
        description=(
            "搜索互联网获取最新信息（只读操作）。"
            "当知识库中没有相关内容时使用。"
            "返回搜索结果摘要。"
        ),
        category=ToolCategory.SEARCH,
        risk=ToolRisk.SAFE,
        function=_web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                }
            },
            "required": ["query"]
        },
        examples=[
            "web_search(query='2024年最新大模型排行榜')",
        ]
    )

    # --- 工具4: 错题集检查 ---
    tool_error = Tool(
        name="check_error_book",
        description=(
            "检查错题集历史记录（只读查询）。"
            "查找与当前问题相似的历史错题，返回错误类型和修正提示。"
        ),
        category=ToolCategory.READ_ONLY,
        risk=ToolRisk.SAFE,
        function=_check_error_book,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检查的问题文本"
                }
            },
            "required": ["query"]
        },
        examples=[
            "check_error_book(query='RAG和微调的区别')",
        ]
    )

    # --- 工具5: 生成最终答案 ---
    tool_gen = Tool(
        name="generate_answer",
        description=(
            "基于已检索到的文档内容，生成最终答案。"
            "当检索结果足够回答用户问题时调用此工具。"
        ),
        category=ToolCategory.GENERATION,
        risk=ToolRisk.SAFE,
        function=_generate_answer,
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "对检索结果的简要总结，说明为什么这些文档足以回答问题"
                }
            },
            "required": ["summary"]
        },
        examples=[
            "generate_answer(summary='已找到3篇关于INTJ的文档，足以回答主导功能问题')",
        ]
    )

    # 批量注册
    tools = [tool_db, tool_kb, tool_web, tool_error, tool_gen]
    registered = 0
    for tool in tools:
        try:
            registry.register(tool)
            registered += 1
        except Exception as e:
            log.warning(f"注册工具 [{tool.name}] 失败: {e}")

    log.info(f"[ToolRegistry v2.9] 注册 {registered}/{len(tools)} 个工具")
