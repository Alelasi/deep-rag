"""自定义MCP Server — v2.9.2新增

将DeepRAG项目工具暴露为标准MCP服务。

核心能力：
1. Tools — 有副作用的操作（向量检索、Web搜索）
2. Resources — 只读数据（知识库文档、配置信息）
3. Prompts — 提示词模板（RAG问答模板）

通信方式：
- stdio: 标准输入输出（本地模式，Claude Desktop默认）
- Streamable HTTP: HTTP + SSE（远程模式，2025-03-26规范新增）

面试要点：
- MCP是协议，不是框架
- Host/Client/Server三层架构
- Tools/Resources/Prompts三类能力
- 底层用JSON-RPC 2.0通信
- 2025-03-26规范：Streamable HTTP替代旧的HTTP+SSE双端点方案

用法：
    # 启动Server（stdio模式）
    python -m src.tools.mcp_server

    # 启动Server（Streamable HTTP模式）
    python -m src.tools.mcp_server --transport http --port 8080

    # 或在Claude Desktop配置中添加：
    {
      "mcpServers": {
        "deeprag": {
          "command": "python",
          "args": ["-m", "src.tools.mcp_server"]
        }
      }
    }
"""
import sys
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

log = logging.getLogger("deeprag")

# ============================================================
# MCP协议常量
# ============================================================

MCP_PROTOCOL_VERSION = "2025-03-26"  # 2025年3月更新：Streamable HTTP
SERVER_NAME = "deeprag-mcp-server"
SERVER_VERSION = "1.0.0"


# ============================================================
# Tools定义 — 有副作用的操作
# ============================================================

TOOLS = [
    {
        "name": "vector_search",
        "description": "向量语义检索，从知识库中搜索与查询最相关的文档块。适用于查找特定知识、概念解释、技术文档。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                },
                "collection_name": {
                    "type": "string",
                    "description": "知识库集合名（默认: default）",
                    "default": "default"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量（默认: 5）",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "exact_match",
        "description": "精确匹配查询，从SQLite检索完全匹配的文档。适用于查找特定条目、定义、术语。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "精确匹配关键词"
                },
                "collection_name": {
                    "type": "string",
                    "description": "知识库集合名",
                    "default": "default"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "graph_search",
        "description": "知识图谱查询，检索实体间关系。适用于查找人物关系、概念关联、技术栈依赖。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "查询实体名"
                },
                "relation": {
                    "type": "string",
                    "description": "关系类型（可选）",
                    "default": ""
                }
            },
            "required": ["entity"]
        }
    },
    {
        "name": "web_search",
        "description": "网络搜索，从DuckDuckGo获取实时信息。适用于查找最新资讯、实时数据、网络信息。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数（默认: 3）",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "rag_query",
        "description": "完整的RAG问答，执行问题分析→知识检索→深度研究→交叉验证→生成报告的完整流程。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户问题"
                },
                "collection_name": {
                    "type": "string",
                    "description": "知识库集合名",
                    "default": "default"
                },
                "mode": {
                    "type": "string",
                    "description": "查询模式: simple/smart/expanded/precision",
                    "enum": ["simple", "smart", "expanded", "precision"],
                    "default": "smart"
                }
            },
            "required": ["question"]
        }
    }
]


# ============================================================
# Resources定义 — 只读数据
# ============================================================

RESOURCES = [
    {
        "uri": "deeprag://collections",
        "name": "知识库集合列表",
        "description": "列出所有可用的知识库集合及其元数据",
        "mimeType": "application/json"
    },
    {
        "uri": "deeprag://config",
        "name": "系统配置",
        "description": "当前DeepRAG系统的配置信息（不含敏感密钥）",
        "mimeType": "application/json"
    },
    {
        "uri": "deeprag://stats",
        "name": "系统统计",
        "description": "系统运行统计：调用次数、缓存命中率、延迟等",
        "mimeType": "application/json"
    }
]


# ============================================================
# Prompts定义 — 提示词模板
# ============================================================

PROMPTS = [
    {
        "name": "rag_answer",
        "description": "RAG问答的标准提示词模板，包含角色设定、格式要求、引用规范",
        "arguments": [
            {
                "name": "question",
                "description": "用户问题",
                "required": True
            },
            {
                "name": "context",
                "description": "检索到的文档上下文",
                "required": True
            },
            {
                "name": "style",
                "description": "回答风格: concise/detailed/analytical",
                "required": False
            }
        ]
    },
    {
        "name": "fact_check",
        "description": "事实核查提示词模板，用于验证答案准确性",
        "arguments": [
            {
                "name": "claim",
                "description": "待验证的声明",
                "required": True
            },
            {
                "name": "evidence",
                "description": "参考证据",
                "required": True
            }
        ]
    },
    {
        "name": "code_review",
        "description": "代码审查提示词模板，检查bug、安全漏洞和性能问题",
        "arguments": [
            {
                "name": "code",
                "description": "待审查的代码",
                "required": True
            },
            {
                "name": "language",
                "description": "编程语言",
                "required": False
            }
        ]
    }
]


# ============================================================
# 工具执行器
# ============================================================

class ToolExecutor:
    """工具执行器 — 调用DeepRAG的实际功能"""

    def __init__(self):
        self._initialized = False

    def _ensure_init(self):
        """延迟初始化（避免导入循环）"""
        if self._initialized:
            return

        try:
            # 添加项目根目录到path
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            self._initialized = True
        except Exception as e:
            log.error(f"[MCP Server] 初始化失败: {e}")

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            执行结果文本
        """
        self._ensure_init()

        try:
            if tool_name == "vector_search":
                return self._vector_search(**arguments)
            elif tool_name == "exact_match":
                return self._exact_match(**arguments)
            elif tool_name == "graph_search":
                return self._graph_search(**arguments)
            elif tool_name == "web_search":
                return self._web_search(**arguments)
            elif tool_name == "rag_query":
                return self._rag_query(**arguments)
            else:
                return f"未知工具: {tool_name}"
        except Exception as e:
            log.error(f"[MCP Server] 工具执行失败: {tool_name}, 错误: {e}")
            return f"执行失败: {e}"

    def _vector_search(self, query: str, collection_name: str = "default", top_k: int = 5) -> str:
        """向量检索"""
        try:
            from src.retrieval.agentic_tools import VectorSearchTool
            tool = VectorSearchTool()
            results = tool.search(query, collection_name=collection_name, top_k=top_k)
            return json.dumps(results, ensure_ascii=False, indent=2) if results else "未找到相关文档"
        except Exception as e:
            return f"向量检索失败: {e}"

    def _exact_match(self, query: str, collection_name: str = "default") -> str:
        """精确匹配"""
        try:
            from src.retrieval.agentic_tools import ExactMatchTool
            tool = ExactMatchTool()
            results = tool.search(query, collection_name=collection_name)
            return json.dumps(results, ensure_ascii=False, indent=2) if results else "未找到精确匹配结果"
        except Exception as e:
            return f"精确匹配失败: {e}"

    def _graph_search(self, entity: str, relation: str = "") -> str:
        """图谱查询"""
        try:
            from src.retrieval.agentic_tools import GraphSearchTool
            tool = GraphSearchTool()
            results = tool.search(entity, relation=relation)
            return json.dumps(results, ensure_ascii=False, indent=2) if results else "未找到图谱关系"
        except Exception as e:
            return f"图谱查询失败: {e}"

    def _web_search(self, query: str, max_results: int = 3) -> str:
        """网络搜索"""
        try:
            from src.retrieval.agentic_tools import WebSearchTool
            tool = WebSearchTool()
            results = tool.search(query, max_results=max_results)
            return json.dumps(results, ensure_ascii=False, indent=2) if results else "未找到网络搜索结果"
        except Exception as e:
            return f"网络搜索失败: {e}"

    def _rag_query(self, question: str, collection_name: str = "default", mode: str = "smart") -> str:
        """完整RAG查询"""
        try:
            from src.graph import query as rag_query
            result = rag_query(question, collection_name=collection_name, mode=mode)
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:
            return f"RAG查询失败: {e}"


# ============================================================
# Resource提供器
# ============================================================

class ResourceProvider:
    """资源提供器 — 返回只读数据"""

    def read(self, uri: str) -> str:
        """读取资源

        Args:
            uri: 资源URI

        Returns:
            资源内容JSON
        """
        try:
            if uri == "deeprag://collections":
                return self._get_collections()
            elif uri == "deeprag://config":
                return self._get_config()
            elif uri == "deeprag://stats":
                return self._get_stats()
            else:
                return json.dumps({"error": f"未知资源: {uri}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_collections(self) -> str:
        """获取知识库集合列表"""
        try:
            import chromadb
            from src.config import get_chroma_client
            client = get_chroma_client()
            collections = client.list_collections()
            data = [
                {
                    "name": c.name,
                    "count": c.count(),
                }
                for c in collections
            ]
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"无法获取集合列表: {e}"})

    def _get_config(self) -> str:
        """获取系统配置（不含敏感信息）"""
        try:
            from src.config import (
                LLM_BACKEND, LLM_MODEL, EMBEDDING_MODEL,
                ENABLE_RERANKER, ENABLE_WEB_FALLBACK,
            )
            config = {
                "llm_backend": LLM_BACKEND,
                "llm_model": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "enable_reranker": ENABLE_RERANKER,
                "enable_web_fallback": ENABLE_WEB_FALLBACK,
            }
            return json.dumps(config, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"无法获取配置: {e}"})

    def _get_stats(self) -> str:
        """获取系统统计"""
        try:
            from src.llm.gateway import get_gateway
            gateway = get_gateway()
            stats = gateway.get_metrics()
            return json.dumps(stats, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"无法获取统计: {e}"})


# ============================================================
# Prompt模板提供器
# ============================================================

class PromptProvider:
    """提示词模板提供器"""

    def get(self, name: str, arguments: Dict[str, str]) -> str:
        """获取提示词模板

        Args:
            name: 模板名称
            arguments: 模板参数

        Returns:
            渲染后的提示词
        """
        if name == "rag_answer":
            return self._rag_answer(**arguments)
        elif name == "fact_check":
            return self._fact_check(**arguments)
        elif name == "code_review":
            return self._code_review(**arguments)
        else:
            return f"未知提示词模板: {name}"

    def _rag_answer(self, question: str, context: str, style: str = "detailed") -> str:
        """RAG问答模板"""
        style_instructions = {
            "concise": "用1-3句话简洁回答，只给结论。",
            "detailed": "详细回答，包含背景、分析和结论，标注引用来源。",
            "analytical": "分析式回答，先分析问题核心，再给出证据和结论。",
        }
        style_text = style_instructions.get(style, style_instructions["detailed"])

        return f"""你是知识库问答专家。根据提供的文档回答问题。

## 回答风格
{style_text}

## 引用规范
- 每个关键事实后标注来源：[来源: 文档名, 第X块]
- 如果文档中没有相关信息，明确说明"文档中未找到相关信息"

## 参考文档
{context}

## 用户问题
{question}

请根据以上文档回答问题。"""

    def _fact_check(self, claim: str, evidence: str) -> str:
        """事实核查模板"""
        return f"""你是事实核查专家。验证以下声明是否准确。

## 待验证声明
{claim}

## 参考证据
{evidence}

## 核查要求
1. 逐条验证声明中的事实点
2. 标注哪些是正确的、哪些是错误的、哪些无法确认
3. 给出整体可信度评分（1-10分）
4. 列出需要进一步核实的点

请输出结构化的核查报告。"""

    def _code_review(self, code: str, language: str = "python") -> str:
        """代码审查模板"""
        return f"""你是资深{language}工程师，负责代码审查。

## 待审查代码
```{language}
{code}
```

## 审查维度
1. **功能正确性**: 逻辑是否有bug，边界条件是否处理
2. **安全性**: 是否有注入、XSS、权限绕过等漏洞
3. **性能**: 是否有N+1查询、不必要的循环、内存泄漏风险
4. **可读性**: 命名是否清晰，关键逻辑是否有注释
5. **最佳实践**: 是否遵循{language}的惯用写法

## 输出格式
按维度逐项检查，给出具体问题和修改建议。"""


# ============================================================
# MCP Server核心 — JSON-RPC 2.0处理
# ============================================================

class MCPServer:
    """MCP Server核心实现

    处理JSON-RPC 2.0消息，管理Tools/Resources/Prompts。
    """

    def __init__(self):
        self.tool_executor = ToolExecutor()
        self.resource_provider = ResourceProvider()
        self.prompt_provider = PromptProvider()

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理JSON-RPC 2.0请求

        Args:
            request: JSON-RPC请求对象

        Returns:
            JSON-RPC响应对象
        """
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = self._handle_tools_call(params)
            elif method == "resources/list":
                result = self._handle_resources_list()
            elif method == "resources/read":
                result = self._handle_resources_read(params)
            elif method == "prompts/list":
                result = self._handle_prompts_list()
            elif method == "prompts/get":
                result = self._handle_prompts_get(params)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(req_id, -32601, f"方法不存在: {method}")

            return self._success_response(req_id, result)

        except Exception as e:
            log.error(f"[MCP Server] 处理请求失败: {method}, 错误: {e}")
            return self._error_response(req_id, -32000, str(e))

    def _handle_initialize(self, params: Dict) -> Dict:
        """处理初始化请求"""
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_tools_list(self) -> Dict:
        """列出所有工具"""
        return {"tools": TOOLS}

    def _handle_tools_call(self, params: Dict) -> Dict:
        """执行工具调用"""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        result_text = self.tool_executor.execute(tool_name, arguments)

        return {
            "content": [
                {
                    "type": "text",
                    "text": result_text,
                }
            ]
        }

    def _handle_resources_list(self) -> Dict:
        """列出所有资源"""
        return {"resources": RESOURCES}

    def _handle_resources_read(self, params: Dict) -> Dict:
        """读取资源"""
        uri = params.get("uri", "")
        content = self.resource_provider.read(uri)

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content,
                }
            ]
        }

    def _handle_prompts_list(self) -> Dict:
        """列出所有提示词模板"""
        return {"prompts": PROMPTS}

    def _handle_prompts_get(self, params: Dict) -> Dict:
        """获取提示词模板"""
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        text = self.prompt_provider.get(name, arguments)

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": text,
                    },
                }
            ]
        }

    def _success_response(self, req_id: Any, result: Dict) -> Dict:
        """成功响应"""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    def _error_response(self, req_id: Any, code: int, message: str) -> Dict:
        """错误响应"""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message,
            },
        }


# ============================================================
# stdio模式入口
# ============================================================

def run_stdio():
    """以stdio模式运行MCP Server

    从stdin读取JSON-RPC请求，写入stdout响应。
    """
    server = MCPServer()

    log.info(f"[MCP Server] 启动 {SERVER_NAME} v{SERVER_VERSION}")
    log.info(f"[MCP Server] 协议版本: {MCP_PROTOCOL_VERSION}")
    log.info(f"[MCP Server] 工具: {[t['name'] for t in TOOLS]}")
    log.info(f"[MCP Server] 资源: {[r['uri'] for r in RESOURCES]}")
    log.info(f"[MCP Server] 模板: {[p['name'] for p in PROMPTS]}")

    # 设置stdin/stdout为UTF-8
    import io
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = server.handle_request(request)
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except json.JSONDecodeError as e:
            error_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"JSON解析失败: {e}"}
            }
            print(json.dumps(error_resp), flush=True)
        except Exception as e:
            log.error(f"[MCP Server] 处理失败: {e}")


# ============================================================
# Streamable HTTP模式（2025-03-26规范新增）
# ============================================================

def run_streamable_http(host: str = "0.0.0.0", port: int = 8080):
    """以Streamable HTTP模式运行MCP Server

    2025-03-26规范：单端点/mcp，支持普通JSON响应和SSE流式响应。

    Args:
        host: 监听地址
        port: 监听端口
    """
    try:
        from flask import Flask, request, Response
        import queue
    except ImportError:
        log.error("[MCP Server] Streamable HTTP模式需要安装Flask: pip install flask")
        return

    app = Flask(__name__)
    server = MCPServer()

    @app.route("/mcp", methods=["POST"])
    def handle_mcp():
        """处理MCP请求 — Streamable HTTP单端点"""
        try:
            req_data = request.get_json(force=True)
            response = server.handle_request(req_data)

            # 检查是否需要SSE流式响应
            # 简化实现：直接返回JSON响应
            # 完整实现应支持SSE流式推送
            return Response(
                json.dumps(response, ensure_ascii=False),
                content_type="application/json",
                headers={
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                },
            )
        except Exception as e:
            error_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(e)},
            }
            return Response(
                json.dumps(error_resp),
                content_type="application/json",
                status=500,
            )

    @app.route("/mcp", methods=["GET"])
    def handle_sse():
        """SSE端点 — 用于Server向Client推送事件"""
        def event_stream():
            # 简化实现：发送心跳
            yield "event: ping\ndata: {}\n\n"

        return Response(
            event_stream(),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            },
        )

    @app.route("/.well-known/agent-card.json")
    def agent_card():
        """导出Agent Card — A2A协议兼容"""
        # 这里可以返回MCP Server的能力声明
        return Response(
            json.dumps({
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "protocol": "mcp",
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": [t["name"] for t in TOOLS],
                    "resources": [r["uri"] for r in RESOURCES],
                    "prompts": [p["name"] for p in PROMPTS],
                },
            }, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

    log.info(f"[MCP Server] Streamable HTTP模式启动: http://{host}:{port}/mcp")
    log.info(f"[MCP Server] Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    app.run(host=host, port=port, debug=False)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepRAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="传输方式: stdio(本地) 或 http(Streamable HTTP远程)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP模式监听地址")
    parser.add_argument("--port", type=int, default=8080, help="HTTP模式监听端口")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=open("mcp_server.log", "a", encoding="utf-8") if args.transport == "stdio" else sys.stderr,
    )

    if args.transport == "http":
        run_streamable_http(host=args.host, port=args.port)
    else:
        run_stdio()
