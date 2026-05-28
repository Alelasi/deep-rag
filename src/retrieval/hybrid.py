"""混合检索器 — BM25(关键词) + 向量(语义) 融合"""
import jieba
from src.retrieval.indexer import Indexer
from src.state import Document


class HybridRetriever:
    """BM25 + ChromaDB向量检索，Reciprocal Rank Fusion合并"""

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
        """ChromaDB向量检索"""
        collection = self.indexer.get_collection()
        if collection is None or collection.count() == 0:
            return []

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
            )
        except Exception:
            return []

        docs = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 1.0
                docs.append({
                    "doc_id": doc_id,
                    "content": content,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "_vector_distance": float(distance),
                })
        return docs
