"""FastAPI 接口测试 - 用 TestClient 验证端点"""
import pytest
from fastapi.testclient import TestClient

from api import app
from src.graph import get_indexer

client = TestClient(app)


def test_health():
    """健康检查端点"""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "agentic_rag_enabled" in data


def test_collections():
    """列出集合端点"""
    r = client.get("/collections")
    assert r.status_code == 200
    assert "collections" in r.json()


def test_query_endpoint():
    """单次查询端点 - 完整流程"""
    # 准备数据
    indexer = get_indexer("api_test")
    indexer.index_texts([
        {"content": "FastAPI 是一个现代化的 Python Web 框架", "metadata": {"source": "doc1"}},
    ])

    r = client.post("/query", json={
        "question": "什么是 FastAPI",
        "collection_name": "api_test",
        "max_retries": 1,
    })

    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "citations" in data
    assert "hallucination_score" in data
    assert "history" in data
    assert data["mode"] in ("hybrid", "agentic")


def test_query_invalid_input():
    """空问题应被校验拒绝"""
    r = client.post("/query", json={"question": "", "collection_name": "test"})
    assert r.status_code == 422  # Pydantic 校验失败


def test_index_invalid_path():
    """不存在的目录应返回 400"""
    r = client.post("/index", json={
        "collection_name": "test",
        "docs_dir": "Z:/nonexistent_path_12345",
    })
    assert r.status_code == 400


def test_query_stream():
    """流式接口返回 SSE 格式"""
    r = client.post("/query/stream", json={
        "question": "test stream",
        "collection_name": "api_test",
        "max_retries": 0,
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "data:" in body
    assert "done" in body or "error" in body
