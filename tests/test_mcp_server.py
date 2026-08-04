"""MCP Server 测试 - 验证 JSON-RPC 协议实现"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
# 入口脚本：start_mcp_server.py（旧 mcp_server.py 已迁入 src/tools/）
MCP_SCRIPT = PROJECT_ROOT / "start_mcp_server.py"


def call_mcp(request: dict) -> dict:
    """调用 MCP Server 并返回响应"""
    proc = subprocess.run(
        [sys.executable, str(MCP_SCRIPT)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        timeout=90,
    )
    # stderr 包含 import warnings，忽略；stdout 应只含 JSON 响应
    stdout_lines = [l for l in proc.stdout.strip().split("\n") if l.strip().startswith("{")]
    if not stdout_lines:
        raise RuntimeError(f"No JSON response. stderr: {proc.stderr[:500]}")
    return json.loads(stdout_lines[-1])


def test_mcp_initialize():
    """initialize 方法应返回 protocol version 和 server info"""
    response = call_mcp({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert response["result"]["protocolVersion"] == "2025-03-26"
    assert response["result"]["serverInfo"]["name"] == "deeprag-mcp-server"


def test_mcp_tools_list():
    """tools/list 应返回 5 个工具"""
    response = call_mcp({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert "result" in response
    tools = response["result"]["tools"]
    assert len(tools) == 5
    tool_names = {t["name"] for t in tools}
    assert tool_names == {"vector_search", "exact_match", "graph_search", "web_search", "rag_query"}


def test_mcp_tools_list_schema():
    """每个工具应有完整的 inputSchema"""
    response = call_mcp({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
    tools = response["result"]["tools"]
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"
        assert "properties" in tool["inputSchema"]
        assert "required" in tool["inputSchema"]


def test_mcp_call_vector_search():
    """vector_search 工具调用"""
    response = call_mcp({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "vector_search",
            "arguments": {"query": "test", "collection": "default", "top_k": 3},
        },
    })
    assert "result" in response
    assert "content" in response["result"]
    assert response["result"]["content"][0]["type"] == "text"


def test_mcp_call_exact_match():
    """exact_match 工具应返回文本结果（已实现）"""
    response = call_mcp({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "exact_match", "arguments": {"query": "id:123"}},
    })
    assert "result" in response
    text = response["result"]["content"][0]["text"]
    assert isinstance(text, str) and len(text) > 0


def test_mcp_unknown_method():
    """未知方法应返回 -32601 错误"""
    response = call_mcp({"jsonrpc": "2.0", "id": 6, "method": "unknown/method", "params": {}})
    assert "error" in response
    assert response["error"]["code"] == -32601


def test_mcp_unknown_tool():
    """未知工具应返回文本提示（执行器不抛协议错误）"""
    response = call_mcp({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "nonexistent_tool", "arguments": {}},
    })
    assert "result" in response
    text = response["result"]["content"][0]["text"]
    assert "未知工具" in text
