"""Reranker 单元测试（对齐当前 API：Reranker 三模式 api/cpu/none）

当前实现（reranker.py v2.8.3）：
- 有 SILICONFLOW_API_KEY → api 模式
- 否则本地 CrossEncoder → cpu 模式
- 都失败 → none 降级模式（跳过重排）
- rerank(query, documents: list[dict], top_k) -> list[dict]，结果附加 "rerank_score"
"""
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import pytest

import src.retrieval.reranker as reranker_mod
from src.retrieval.reranker import Reranker


def _make_instance(mode="none", **attrs):
    """绕过 __init__ 直接构造实例，避免加载模型/网络"""
    r = object.__new__(Reranker)
    r.model_name = "mock-model"
    r.api_key = ""
    r.api_url = "https://api.siliconflow.cn/v1/rerank"
    r.mode = mode
    r.model = None
    r.device = "cpu"
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


def _docs(n=3, prefix="doc"):
    return [{"doc_id": f"{prefix}{i}", "content": f"内容{i} 关键词"} for i in range(n)]


# ===== 模式选择 =====

def test_mode_api_when_key_present(monkeypatch):
    """有 API key → api 模式，不加载本地模型"""
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    r = Reranker()
    assert r.mode == "api"


def test_mode_none_when_model_unavailable(monkeypatch):
    """无 API key 且本地模型加载失败 → none 降级"""
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    def _boom(*a, **k):
        raise ImportError("sentence-transformers not installed")

    monkeypatch.setattr(reranker_mod, "_get_cross_encoder", _boom)
    r = Reranker()
    assert r.mode == "none"


# ===== rerank 基础行为 =====

def test_rerank_empty_input():
    """空文档列表返回空列表"""
    r = _make_instance(mode="none")
    assert r.rerank("query", []) == []


def test_rerank_none_mode_truncates():
    """none 模式：跳过重排，仅截断 top_k"""
    r = _make_instance(mode="none")
    results = r.rerank("query", _docs(5), top_k=3)
    assert len(results) == 3


def test_rerank_failure_falls_back_to_original():
    """cpu 模式模型异常 → 返回原始 top_k"""
    r = _make_instance(mode="cpu")

    class BoomModel:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("inference failed")

    r.model = BoomModel()
    results = r.rerank("query", _docs(5), top_k=2)
    assert len(results) == 2


# ===== api 模式排序 =====

def test_rerank_api_ordering(monkeypatch):
    """api 模式按 relevance_score 降序重排，写入 rerank_score"""
    r = _make_instance(mode="api", api_key="sk-test")
    docs = _docs(3)

    def fake_api(query, documents, top_k):
        order = [2, 0, 1]
        out = []
        for rank, idx in enumerate(order[:top_k]):
            d = documents[idx].copy()
            d["rerank_score"] = 1.0 - rank * 0.1
            out.append(d)
        return out

    monkeypatch.setattr(r, "_rerank_api", fake_api)
    results = r.rerank("query", docs, top_k=3)

    assert results[0]["doc_id"] == "doc2"
    assert results[0]["rerank_score"] == pytest.approx(1.0)
    assert results[2]["rerank_score"] == pytest.approx(0.8)


def test_rerank_preserves_doc_fields():
    """重排不破坏原字段，仅新增 rerank_score"""
    r = _make_instance(mode="none")
    docs = [{"doc_id": "d1", "content": "content", "source": "src.md", "page": 42}]
    results = r.rerank("content", docs, top_k=1)
    assert results[0]["doc_id"] == "d1"
    assert results[0]["source"] == "src.md"
    assert results[0]["page"] == 42


def test_get_text_content_or_text():
    """_get_text 兼容 content / text 两种字段"""
    r = _make_instance()
    assert r._get_text({"content": "A", "text": "B"}) == "A"
    assert r._get_text({"text": "B"}) == "B"
    assert r._get_text({}) == ""
