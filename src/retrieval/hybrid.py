"""混合检索器 — BM25(关键词) + 向量(语义) 融合

主路径（graph / MCP / 多数测试）使用本模块：
    HybridRetriever(indexer: Indexer)

并行版见 hybrid_retriever.ParallelHybridRetriever(bm25, vector)，
构造签名不同，勿与本类混用同名导入。
"""
import logging
import jieba
from src.retrieval.indexer import Indexer
from src.state import Document

log = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + 向量检索，RRF 合并（主路径：传入 Indexer）"""

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 8,
                 bm25_weight: float = 0.4, vector_weight: float = 0.6) -> list[Document]:
        """
        混合检索
        用RRF(Reciprocal Rank Fusion)合并两路结果，比简单加权更鲁棒
        """
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        vector_results = self._vector_search(query, top_k=top_k * 2)

        # RRF合并：score = sum(1 / (k + rank)) across all lists
        # k=60是经验值（原始论文推荐）
        k = 60
        rrf_scores = {}  # doc_id → score

        for rank, doc in enumerate(bm25_results):
            did = doc["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0) + bm25_weight / (k + rank + 1)

        for rank, doc in enumerate(vector_results):
            did = doc["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0) + vector_weight / (k + rank + 1)

        # 合并文档信息，按RRF分数排序
        all_docs = {}
        for doc in bm25_results + vector_results:
            all_docs[doc["doc_id"]] = doc

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
        results = []
        for did in sorted_ids[:top_k]:
            if did in all_docs:
                doc = all_docs[did]
                results.append(Document(
                    doc_id=did,
                    content=doc["content"],
                    source=doc.get("source", ""),
                    page=doc.get("page", 0),
                    metadata={"rrf_score": round(rrf_scores[did], 6)},
                ))

        return results

    def _bm25_search(self, query: str, top_k: int = 16) -> list[dict]:
        """BM25关键词检索"""
        bm25, docs = self.indexer.get_bm25()
        if not bm25 or not docs:
            return []

        tokens = list(jieba.cut(query))
        scores = bm25.get_scores(tokens)

        # 排序
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in indexed:
            if score > 0:
                doc = docs[idx].copy()
                doc["_bm25_score"] = float(score)
                results.append(doc)
        return results

    def _vector_search(self, query: str, top_k: int = 16) -> list[dict]:
        """ChromaDB向量检索（v2.6：支持子集合 + 手动嵌入避免维度不匹配）"""
        # v2.6: 获取所有子集合
        collections = self.indexer.get_all_collections()
        if not collections:
            return []

        # 用Indexer的embedder手动嵌入查询文本
        embedder = self.indexer._get_embedder()
        query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

        all_docs = []
        for col in collections:
            try:
                count = col.count()
            except Exception:
                continue
            if count == 0:
                continue
            try:
                results = col.query(
                    query_embeddings=query_embedding,
                    n_results=min(top_k, count),
                )
                if results and results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        meta = results["metadatas"][0][i] if results["metadatas"] else {}
                        content = results["documents"][0][i] if results["documents"] else ""
                        distance = results["distances"][0][i] if results["distances"] else 1.0
                        all_docs.append({
                            "doc_id": doc_id,
                            "content": content,
                            "source": meta.get("source", ""),
                            "page": meta.get("page", 0),
                            "_vector_distance": float(distance),
                        })
            except Exception as e:
                log.warning(f"子集合 {col.name} 查询失败: {e}")
                continue

        # 按距离排序，取top_k
        all_docs.sort(key=lambda d: d.get("_vector_distance", 1.0))
        return all_docs[:top_k]
