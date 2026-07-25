"""
幻觉检测增强版检索器
相似度阈值 + 置信度评分 + 来源验证

默认优先 Qdrant（VECTOR_DB=qdrant），仅在 chromadb 模式才连 Chroma HttpClient。
"""
from typing import List, Dict, Tuple
from ..config import EMBEDDING_MODEL, DEVICE, VECTOR_DB


class HallucinationAwareRetriever:
    """幻觉检测增强版检索器"""

    def __init__(
        self,
        collection_name: str = "full_docs",
        model_name: str = None,
        device: str = None,
        similarity_threshold: float = 0.5
    ):
        name = model_name or EMBEDDING_MODEL
        from src.ui.model_cache import get_embedding_model
        self.model = get_embedding_model(name, device or DEVICE)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.backend = (VECTOR_DB or "qdrant").lower()
        self.collection = None
        self._qdrant = None

        if self.backend == "qdrant":
            try:
                from src.retrieval.qdrant_retriever import get_qdrant_retriever
                self._qdrant = get_qdrant_retriever(collection_name)
            except Exception:
                self._qdrant = None
        else:
            try:
                from ..config import get_chroma_client
                client = get_chroma_client()
                try:
                    self.collection = client.get_collection(collection_name)
                except Exception:
                    self.collection = None
            except Exception:
                self.collection = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_confidence: bool = True
    ) -> Tuple[List[Dict], str]:
        """
        检索并检测幻觉

        Args:
            query: 查询文本
            top_k: 返回文档数
            return_confidence: 是否返回置信度

        Returns:
            (结果列表, 置信度等级)
        """
        # ---- Qdrant 路径 ----
        if self._qdrant is not None:
            try:
                emb = self.model.encode([query])[0]
                hits = self._qdrant.search(emb.tolist() if hasattr(emb, "tolist") else list(emb), top_k=top_k * 2)
            except Exception:
                hits = []
            if not hits:
                return [], "no_results"
            formatted = []
            for h in hits:
                # Qdrant 分数多为 cosine 相似度或 distance，统一成 similarity
                score = float(
                    h.get("score")
                    or h.get("_score")
                    or h.get("similarity")
                    or 0.0
                )
                if score > 1.0:
                    score = 1.0 / (1.0 + score)
                payload = h.get("payload") or h
                formatted.append({
                    "content": payload.get("content") or h.get("content") or "",
                    "source": payload.get("source") or h.get("source") or "unknown",
                    "page": payload.get("page") or h.get("page") or 1,
                    "similarity": score if score > 0 else 0.6,
                    "confidence": self._classify_confidence(score if score > 0 else 0.6),
                })
            filtered = [r for r in formatted if r["similarity"] >= self.similarity_threshold]
            if not filtered and formatted:
                filtered = formatted[:top_k]
            if not filtered:
                return [], "low_confidence"
            filtered.sort(key=lambda x: x["similarity"], reverse=True)
            final_results = filtered[:top_k]
            return final_results, self._compute_overall_confidence(final_results)

        # ---- Chroma 路径 ----
        if not self.collection:
            return [], "no_data"

        # 向量化查询
        query_embedding = self.model.encode([query])[0]

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2  # 多取一些，过滤后返回top_k
        )

        if not results['documents'][0]:
            return [], "no_results"

        # 格式化结果
        formatted = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - distance
            formatted.append({
                'content': doc,
                'source': metadata.get('source', 'unknown'),
                'page': metadata.get('page', 1),
                'similarity': similarity,
                'confidence': self._classify_confidence(similarity),
            })

        # 幻觉检测（过滤低相似度）
        filtered = [r for r in formatted if r['similarity'] >= self.similarity_threshold]

        if not filtered:
            return [], "low_confidence"

        # 排序并返回top_k
        filtered.sort(key=lambda x: x['similarity'], reverse=True)
        final_results = filtered[:top_k]

        # 计算整体置信度
        overall_confidence = self._compute_overall_confidence(final_results)

        return final_results, overall_confidence

    def _classify_confidence(self, similarity: float) -> str:
        """分类置信度"""
        if similarity >= 0.8:
            return "very_high"
        elif similarity >= 0.7:
            return "high"
        elif similarity >= 0.6:
            return "medium"
        elif similarity >= 0.5:
            return "low"
        else:
            return "very_low"

    def _compute_overall_confidence(self, results: List[Dict]) -> str:
        """计算整体置信度"""
        if not results:
            return "no_results"

        avg_similarity = sum(r['similarity'] for r in results) / len(results)

        if avg_similarity >= 0.7:
            return "high"
        elif avg_similarity >= 0.6:
            return "medium"
        elif avg_similarity >= 0.5:
            return "low"
        else:
            return "very_low"

    def retrieve_with_explanation(self, query: str, top_k: int = 3) -> Dict:
        """检索并返回详细解释"""
        results, confidence = self.retrieve(query, top_k)

        explanation = self._generate_explanation(query, results, confidence)

        return {
            'query': query,
            'results': results,
            'confidence': confidence,
            'explanation': explanation,
            'should_trust': confidence in ['high', 'medium'],
        }

    def _generate_explanation(
        self,
        query: str,
        results: List[Dict],
        confidence: str
    ) -> str:
        """生成解释"""
        if confidence == "no_data":
            return "❌ 知识库为空，无法检索"

        if confidence == "no_results":
            return "❌ 未找到任何相关内容"

        if confidence == "low_confidence":
            return "⚠️ 找到结果但相似度过低（<50%），可能是幻觉，建议换个问法"

        if confidence == "very_low":
            return "⚠️ 相似度很低（50-60%），结果可能不准确"

        if confidence == "low":
            return "⚠️ 相似度较低（50-60%），建议谨慎参考"

        if confidence == "medium":
            return "✅ 相似度中等（60-70%），结果可能相关"

        if confidence == "high":
            return "✅ 相似度高（70%+），结果高度相关"

        return "未知"

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 3
    ) -> List[Dict]:
        """批量检索"""
        results = []
        for query in queries:
            result = self.retrieve_with_explanation(query, top_k)
            results.append(result)
        return results
