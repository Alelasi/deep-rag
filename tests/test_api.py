"""FastAPI 接口测试 — 使用根 api:app，Mock RAG 避免依赖向量库"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime_seconds" in data


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert "package_version" in body
    assert "capability_version" in body


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "deeprag_uptime_seconds" in r.text


def test_query_endpoint_mocked():
    fake = {
        "answer": "【直接回答】测试答案",
        "citations": [{"source": "doc1", "page": 0, "text": "x"}],
        "hallucination_score": 0.1,
        "fact_check_passed": True,
        "relevant_count": 1,
        "conflicts": [],
        "history": ["ok"],
        "no_knowledge": False,
        "used_mock_web": False,
    }
    with patch("scripts.api.rag_query", return_value=fake):
        r = client.post(
            "/query",
            json={"question": "什么是测试", "collection_name": "default", "max_retries": 1},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["answer"]
    assert data["request_id"]
    assert data["fact_check_passed"] is True


def test_query_empty_rejected():
    r = client.post("/query", json={"question": "  ", "collection_name": "default"})
    # pydantic min_length or sanitize
    assert r.status_code in (400, 422)


def test_index_path_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_ALLOWED_ROOTS", str(tmp_path / "allowed"))
    (tmp_path / "allowed").mkdir()
    outside = tmp_path / "hack"
    outside.mkdir()
    # reload validation uses env at call time for roots — ok
    r = client.post(
        "/index",
        json={"collection_name": "x", "docs_dir": str(outside)},
    )
    assert r.status_code in (403, 400)


def test_auth_when_key_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "prod-key-xyz")
    # force re-read
    r = client.get("/collections")
    # without key should 401 when auth enabled
    assert r.status_code == 401
    r2 = client.get("/collections", headers={"X-API-Key": "prod-key-xyz"})
    # may 200 with empty collections
    assert r2.status_code == 200
