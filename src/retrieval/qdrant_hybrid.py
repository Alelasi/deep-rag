"""Qdrant 混合检索器 — BM25(关键词) + Qdrant向量(语义) 融合

替代 ChromaDB 版 HybridRetriever，解决 HNSW 重启损坏问题。
"""
import logging
import jieba
from src.retrieval.qdrant_indexer import QdrantIndexer
from src.state import Document

log = logging.getLogger(__name__)


class QdrantHybridRetriever:
    """BM25 + Qdrant 向量检索，Reciprocal Rank Fusion 合并"""

    def __init__(self, indexer: QdrantIndexer):
        self.indexer = indexer

    def search(self, query: str, top_k: int = 8,
               bm25_weight: float = 0.4, vector_weight: float = 0.6) -> list[Document]:
        """混合检索（兼容旧接口）"""
        return self.retrieve(query, top_k, bm25_weight, vector_weight)

    def retrieve(self, query: str, top_k: int = 8,
                 bm25_weight: float = 0.4, vector_weight: float = 0.6) -> list[Document]:
        """混合检索：BM25 + Qdrant 向量 + RRF 融合"""
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        vector_results = self._vector_search(query, top_k=top_k * 2)

        # RRF 合并
        k = 60
        rrf_scores = {}

        for rank, doc in enumerate(bm25_results):
            did = doc["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0) + bm25_weight / (k + rank + 1)

        for rank, doc in enumerate(vector_results):
            did = doc["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0) + vector_weight / (k + rank + 1)

        # 合并文档信息
        all_docs = {}
        for doc in bm25_results + vector_results:
            all_docs[doc["doc_id"]] = doc

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
        results = []
        for did in sorted_ids[: top_k * 3]:  # 多取再滤脏源
            if did in all_docs:
                doc = all_docs[did]
                results.append(Document(
                    doc_id=did,
                    content=doc["content"],
                    source=doc.get("source", ""),
                    page=doc.get("page", 0),
                    metadata={"rrf_score": round(rrf_scores[did], 6)},
                ))
        try:
            from src.retrieval.source_filter import filter_docs, prefer_exact_type_stack
            results = filter_docs(results)
            # query 从调用方传入：在 retrieve() 内可用
            results = prefer_exact_type_stack(query, results)[:top_k]
        except Exception:
            results = results[:top_k]
        return results

    def _bm25_search(self, query: str, top_k: int = 16) -> list[dict]:
        """BM25 关键词检索"""
        bm25, docs = self.indexer.get_bm25()
        if not bm25 or not docs:
            return []

        tokens = list(jieba.cut(query))
        scores = bm25.get_scores(tokens)

        indexed = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in indexed:
            if score > 0:
                doc = docs[idx].copy()
                doc["_bm25_score"] = float(score)
                results.append(doc)
        return results

    def _vector_search(self, query: str, top_k: int = 16) -> list[dict]:
        """Qdrant 向量检索"""
        embedder = self.indexer._get_embedder()
        query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()[0]

        try:
            results = self.indexer.retriever.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            return results
        except Exception as e:
            log.warning(f"Qdrant 查询失败: {e}")
            return []
