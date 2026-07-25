#!/usr/bin/env python3
"""[LEGACY] DeepRAG MCP Server - 手写 JSON-RPC stdio（协议 2024-11-05）

⚠️ 已弃用作为主入口。请改用：
    python start_mcp_server.py
    # 或
    python -m src.tools.mcp_server

本文件仅保留兼容旧客户端；新集成请使用 src.tools.mcp_server（2025-03-26）。
"""
import sys
import json
import logging
from typing import Any

logging.basicConfig(level=logging.WARNING, filename="mcp_server.log")
log = logging.getLogger("deeprag.mcp")
log.warning(
    "scripts/mcp_server.py is LEGACY; prefer: python start_mcp_server.py "
    "or python -m src.tools.mcp_server"
)

from src.graph import get_indexer
from src.retrieval.hybrid import HybridRetriever


class MCPServer:
    """MCP Server - JSON-RPC 2.0 over stdio"""

    def __init__(self):
        self.tools = {
            "vector_search": {
                "name": "vector_search",
                "description": "向量语义检索工具 - 基于语义相似度检索知识库文档",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "查询文本"},
                        "collection": {"type": "string", "description": "知识库集合名", "default": "default"},
                        "top_k": {"type": "integer", "description": "返回结果数", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            "exact_match": {
                "name": "exact_match",
                "description": "精确匹配工具 - 通过SQLite查询特定 ID/订单号/版本号",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "查询文本"},
                        "filters": {"type": "object", "description": "过滤条件"},
                    },
                    "required": ["query"],
                },
            },
            "graph_search": {
                "name": "graph_search",
                "description": "图检索工具 - 通过NetworkX知识图谱查询实体关系",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "实体名称"},
                        "relation": {"type": "string", "description": "关系类型"},
                    },
                    "required": ["entity"],
                },
            },
            "web_search": {
                "name": "web_search",
                "description": "网络搜索工具 - 通过DuckDuckGo搜索最新信息（知识库兜底）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "查询文本"},
                        "max_results": {"type": "integer", "description": "最大结果数", "default": 3},
                    },
                    "required": ["query"],
                },
            },
        }

    def handle_request(self, request: dict) -> dict:
        """处理 JSON-RPC 请求"""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "initialize":
                return self._initialize(req_id)
            elif method == "tools/list":
                return self._list_tools(req_id)
            elif method == "tools/call":
                return self._call_tool(req_id, params)
            else:
                return self._error(req_id, -32601, f"Method not found: {method}")
        except Exception as e:
            log.exception("Request handling failed")
            return self._error(req_id, -32603, str(e))

    def _initialize(self, req_id) -> dict:
        """初始化响应"""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "deeprag-mcp-server",
                    "version": "2.0.0",
                },
            },
        }

    def _list_tools(self, req_id) -> dict:
        """列出所有工具"""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": list(self.tools.values())},
        }

    def _call_tool(self, req_id, params: dict) -> dict:
        """调用工具"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            return self._error(req_id, -32602, f"Unknown tool: {tool_name}")

        # 执行工具
        if tool_name == "vector_search":
            result = self._vector_search(arguments)
        elif tool_name == "exact_match":
            result = self._exact_match(arguments)
        elif tool_name == "graph_search":
            result = self._graph_search(arguments)
        elif tool_name == "web_search":
            result = self._web_search(arguments)
        else:
            result = {"content": [{"type": "text", "text": f"工具 {tool_name} 未实现"}]}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    def _vector_search(self, args: dict) -> dict:
        """向量检索实现"""
        query = args.get("query", "")
        collection = args.get("collection", "default")
        top_k = args.get("top_k", 5)

        indexer = get_indexer(collection)
        retriever = HybridRetriever(indexer)
        docs = retriever.retrieve(query, top_k=top_k)

        # 格式化结果
        if not docs:
            text = f"未找到相关文档（集合: {collection}）"
        else:
            lines = [f"检索到 {len(docs)} 个相关文档：\n"]
            for i, doc in enumerate(docs, 1):
                content = doc.get("content", "")[:200]
                source = doc.get("metadata", {}).get("source", "unknown")
                lines.append(f"{i}. [{source}] {content}...")
            text = "\n".join(lines)

        return {"content": [{"type": "text", "text": text}]}

    def _exact_match(self, args: dict) -> dict:
        """精确匹配实现 - 通过SQLite查询特定ID/编号/版本号"""
        query = args.get("query", "")
        filters = args.get("filters")

        try:
            # 懒导入避免循环依赖
            from src.retrieval.agentic_tools import ExactMatchTool
            tool = ExactMatchTool()
            docs = tool.search(query, filters=filters)
        except Exception as e:
            log.exception("exact_match failed")
            return {"content": [{"type": "text", "text": f"精确匹配查询失败: {e}"}]}

        if not docs:
            text = f"未找到精确匹配的文档（查询: {query}）"
        else:
            lines = [f"精确匹配到 {len(docs)} 个文档：\n"]
            for i, doc in enumerate(docs, 1):
                content = doc.get("content", "")[:200]
                source = doc.get("source", "unknown")
                lines.append(f"{i}. [{source}] {content}")
            text = "\n".join(lines)

        return {"content": [{"type": "text", "text": text}]}

    def _graph_search(self, args: dict) -> dict:
        """图检索实现 - 通过NetworkX知识图谱查询实体关系"""
        entity = args.get("entity", "")
        relation = args.get("relation")

        try:
            # 懒导入避免循环依赖
            from src.retrieval.agentic_tools import GraphSearchTool
            tool = GraphSearchTool()
            # GraphSearchTool.search() 从query中提取实体，这里直接传entity作为query
            docs = tool.search(entity, relation=relation)
            node_count = len(tool.graph.nodes)
        except Exception as e:
            log.exception("graph_search failed")
            return {"content": [{"type": "text", "text": f"图检索查询失败: {e}"}]}

        if not docs:
            if node_count == 0:
                text = (
                    f"知识图谱为空，暂无实体关系数据。\n"
                    f"实体 '{entity}' 未找到关联关系。\n"
                    f"提示：请先通过 build_all_kb_v2.py 构建知识图谱。"
                )
            else:
                text = (
                    f"实体 '{entity}' 在知识图谱中未找到关联关系\n"
                    f"（当前图谱共 {node_count} 个节点）。"
                )
        else:
            lines = [f"图检索到 {len(docs)} 条关系：\n"]
            for i, doc in enumerate(docs, 1):
                content = doc.get("content", "")
                lines.append(f"{i}. {content}")
            text = "\n".join(lines)

        return {"content": [{"type": "text", "text": text}]}

    def _web_search(self, args: dict) -> dict:
        """网络搜索实现 - 通过DuckDuckGo搜索外部信息"""
        query = args.get("query", "")
        max_results = args.get("max_results", 3)

        try:
            # 懒导入避免循环依赖
            from src.retrieval.web_fallback import web_search_fallback
            results = web_search_fallback(query, max_results=max_results)
        except Exception as e:
            log.exception("web_search failed")
            return {"content": [{"type": "text", "text": f"网络搜索失败: {e}"}]}

        if not results:
            text = f"网络搜索未返回结果（查询: {query}）"
        else:
            lines = [f"网络搜索到 {len(results)} 条结果：\n"]
            for i, result in enumerate(results, 1):
                content = result.get("content", "")[:200]
                source = result.get("source", "unknown")
                metadata = result.get("metadata", {})
                title = metadata.get("title", "")
                engine = metadata.get("engine", "unknown")
                is_mock = metadata.get("is_mock", False)
                mock_tag = " [Mock]" if is_mock else ""
                lines.append(f"{i}.{mock_tag} [{title}] ({engine})")
                lines.append(f"   URL: {source}")
                lines.append(f"   摘要: {content}")
            text = "\n".join(lines)

        return {"content": [{"type": "text", "text": text}]}

    def _error(self, req_id, code: int, message: str) -> dict:
        """错误响应"""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    def run(self):
        """主循环 - 从 stdin 读取请求，向 stdout 写响应"""
        log.info("MCP Server started")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON: {e}")
                error_resp = self._error(None, -32700, "Parse error")
                print(json.dumps(error_resp), flush=True)


if __name__ == "__main__":
    server = MCPServer()
    server.run()
