"""Qdrant 向量检索器 — 多 collection + local/server

替代 Chroma 作为主向量后端（重启不烂 HNSW）。
集合名请用 ``project_collections.resolve_collection``。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional

from qdrant_client.models import Distance, PointStruct, VectorParams

from src.retrieval.project_collections import DEFAULT_COLLECTION, resolve_collection
from src.retrieval.qdrant_client_factory import get_qdrant_client

log = logging.getLogger(__name__)

# 与 bge-base-zh-v1.5 一致
VECTOR_SIZE = 768

# qdrant-client 是否可用（测试与降级逻辑用）
try:
    import qdrant_client  # noqa: F401

    QDRANT_AVAILABLE = True
except Exception:  # pragma: no cover - 环境缺依赖
    QDRANT_AVAILABLE = False


class QdrantRetriever:
    """单 collection 检索器；客户端由工厂共享。"""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION, client=None):
        # 支持 work / proj_work / default 等别名
        self.collection_name = resolve_collection(collection_name)
        self.name = self.collection_name  # 兼容旧 Chroma 接口字段
        self.client = client or get_qdrant_client()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """集合不存在则创建（768 维 cosine）。"""
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            log.info("[Qdrant] Created collection: %s", self.collection_name)

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        """批量 upsert 文档。"""
        points = []
        for doc, emb, meta, doc_id in zip(documents, embeddings, metadatas, ids):
            # 稳定 int id，避免字符串 id 兼容问题
            point_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:15], 16)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={
                        "content": doc,
                        "source": meta.get("source", ""),
                        "page": meta.get("page", 0),
                        "doc_id": doc_id,
                        **{k: v for k, v in meta.items() if k not in ("source", "page")},
                    },
                )
            )
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )
        log.info("[Qdrant] Added %s docs → %s", len(points), self.collection_name)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """向量检索，返回统一文档字典列表。"""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        docs = []
        for r in results.points:
            payload = r.payload or {}
            score = r.score if r.score is not None else 0.0
            docs.append(
                {
                    "doc_id": payload.get("doc_id", str(r.id)),
                    "content": payload.get("content", ""),
                    "source": payload.get("source", ""),
                    "page": payload.get("page", 0),
                    "metadata": payload,
                    "_vector_distance": 1.0 - score,
                    "_score": score,
                }
            )
        try:
            from src.retrieval.source_filter import filter_docs
            docs = filter_docs(docs)
        except Exception:
            pass
        return docs

    def count(self) -> int:
        """集合内点数。"""
        info = self.client.get_collection(self.collection_name)
        return int(info.points_count or 0)

    def is_ready(self) -> bool:
        """是否已有数据可检索。"""
        try:
            return self.count() > 0
        except Exception:
            return False

    def clear(self) -> None:
        """删除并重建空集合。"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()


# collection_name → 实例（修复旧版「忽略后续 collection」单例 bug）
_retrievers: Dict[str, QdrantRetriever] = {}


def get_qdrant_retriever(collection_name: str = DEFAULT_COLLECTION) -> QdrantRetriever:
    """按 collection 缓存检索器。"""
    name = resolve_collection(collection_name)
    if name not in _retrievers:
        _retrievers[name] = QdrantRetriever(name)
    return _retrievers[name]


def list_collection_stats() -> List[Dict]:
    """列出中心库全部 collection 及点数（验收用）。"""
    client = get_qdrant_client()
    out = []
    for c in client.get_collections().collections:
        try:
            n = client.get_collection(c.name).points_count or 0
        except Exception:
            n = -1
        out.append({"name": c.name, "points": int(n)})
    return out
