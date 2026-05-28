#!/usr/bin/env python3
"""DeepRAG MCP Server - 手写 JSON-RPC stdio 实现

MCP (Model Context Protocol) Server，暴露 deep-rag 的 4 种检索工具：
- vector_search: 向量语义检索
- exact_match: 精确 ID 匹配（接口预留）
- graph_search: 知识图谱关系查询（接口预留）
- web_search: 网络搜索兜底（接口预留）

协议：JSON-RPC 2.0 over stdio
启动：python mcp_server.py
客户端：Claude Desktop / 其他 MCP 客户端
"""
import sys
import json
import logging
from typing import Any

from src.graph import get_indexer
from src.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.WARNING, filename="mcp_server.log")
log = logging.getLogger("deeprag.mcp")


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
                "description": "精确匹配工具 - 查询特定 ID/订单号/版本号（接口预留）",
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
                "description": "图检索工具 - 查询实体关系（接口预留）",
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
                "description": "网络搜索工具 - 搜索最新信息（接口预留）",
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
            result = {"content": [{"type": "text", "text": "exact_match 接口预留，未实现"}]}
        elif tool_name == "graph_search":
            result = {"content": [{"type": "text", "text": "graph_search 接口预留，未实现"}]}
        elif tool_name == "web_search":
            result = {"content": [{"type": "text", "text": "web_search 接口预留，未实现"}]}
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
