"""混合检索器 — BM25 + 向量检索 并行融合模块

功能：
- 并行调用BM25检索器和向量检索器（ThreadPoolExecutor, max_workers=2）
- 使用RRF（Reciprocal Rank Fusion）融合两路检索结果
- 文档按doc_id去重
- 返回融合排序后的top_k结果

RRF公式：score(d) = Sigma 1/(k + rank)，k=60（经验值，来自原始论文）
- BM25返回Top-20，向量返回Top-20，融合后返回Top-15
- 每篇文档附加"bm25_rank"和"vector_rank"字段
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

# RRF常数（原始论文推荐值k=60）
RRF_K = 60


class ParallelHybridRetriever:
    """BM25 + 向量检索 并行混合检索器（非 graph 主路径）

    并行调用 BM25Retriever 与向量检索器，RRF 融合。
    构造：ParallelHybridRetriever(bm25_retriever, vector_retriever)
    主路径请用 hybrid.HybridRetriever(indexer)。

    Attributes:
        bm25_retriever: BM25检索器实例
        vector_retriever: 向量检索器实例（Indexer或具有search方法的对象）
    """

    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever):
        """初始化并行混合检索器

        Args:
            bm25_retriever: BM25检索器实例
            vector_retriever: 向量检索器实例（Indexer或具有search方法的对象）
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever

    # ------------------------------------------------------------------ #
    #  内部工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_text(doc: dict) -> str:
        """获取文档文本，兼容content/text两种字段"""
        return doc.get("content") or doc.get("text") or ""

    def _bm25_search(self, query: str, top_k: int = 20) -> list[dict]:
        """调用BM25检索"""
        try:
            return self.bm25_retriever.search(query, top_k=top_k)
        except Exception as e:
            logger.error("BM25检索异常: %s", e, exc_info=True)
            return []

    def _vector_search(self, query: str, top_k: int = 20) -> list[dict]:
        """调用向量检索（v2.6：支持子集合 + 手动嵌入避免维度不匹配）"""
        try:
            # 优先使用search方法（如果存在）
            if hasattr(self.vector_retriever, "search"):
                return self.vector_retriever.search(query, top_k=top_k)

            # v2.6: 获取所有子集合
            if hasattr(self.vector_retriever, "get_all_collections"):
                collections = self.vector_retriever.get_all_collections()
            else:
                collection = self.vector_retriever._get_collection()
                collections = [collection] if collection else []

            if not collections:
                logger.warning("向量索引为空，跳过向量检索")
                return []

            # v2.6: 用Indexer的embedder手动嵌入查询文本
            embedder = self.vector_retriever._get_embedder()
            query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

            all_docs = []
            for col in collections:
                try:
                    count = col.count()
                except Exception:
                    continue
                if count == 0:
                    continue
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
                            "metadata": meta,
                            "_vector_distance": float(distance),
                        })
            # 按距离排序取top_k
            all_docs.sort(key=lambda d: d.get("_vector_distance", 1.0))
            return all_docs[:top_k]

        except Exception as e:
            logger.error("向量检索异常: %s", e, exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    #  核心方法
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """混合检索 — 并行调用两路检索 + RRF融合

        BM25返回Top-20，向量返回Top-20，融合后返回Top-15。

        Args:
            query: 查询文本
            top_k: 最终返回的文档数，默认15

        Returns:
            融合排序后的文档列表，每个文档附加"bm25_rank"和"vector_rank"字段
        """
        recall_k = 20  # 每路召回数

        bm25_results: list[dict] = []
        vector_results: list[dict] = []

        # 并行调用BM25和向量检索
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_bm25 = executor.submit(self._bm25_search, query, recall_k)
                future_vector = executor.submit(
                    self._vector_search, query, recall_k
                )

                for future in as_completed([future_bm25, future_vector]):
                    try:
                        result = future.result()
                        # 通过future标识区分两路结果
                        if future == future_bm25:
                            bm25_results = result
                        else:
                            vector_results = result
                    except Exception as e:
                        logger.error(
                            "检索子任务异常: %s", e, exc_info=True
                        )
        except Exception as e:
            logger.error("并行检索失败，回退到串行: %s", e, exc_info=True)
            # 降级：串行调用
            bm25_results = self._bm25_search(query, recall_k)
            vector_results = self._vector_search(query, recall_k)

        logger.info(
            "混合检索召回: BM25=%d篇, 向量=%d篇",
            len(bm25_results), len(vector_results)
        )

        # RRF融合
        return self._rrf_fusion(bm25_results, vector_results, top_k)

    # ------------------------------------------------------------------ #
    #  RRF融合
    # ------------------------------------------------------------------ #

    def _rrf_fusion(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """RRF（Reciprocal Rank Fusion）融合两路检索结果

        公式: score(d) = Sigma 1/(k + rank)
        其中k=60（经验值），rank从1开始计数

        Args:
            bm25_results: BM25检索结果
            vector_results: 向量检索结果
            top_k: 返回的top文档数

        Returns:
            融合后的文档列表，附加bm25_rank、vector_rank和rrf_score字段
        """
        # 记录每篇文档在各路检索中的排名（rank从1开始）
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}

        for rank, doc in enumerate(bm25_results, start=1):
            bm25_ranks[doc["doc_id"]] = rank

        for rank, doc in enumerate(vector_results, start=1):
            vector_ranks[doc["doc_id"]] = rank

        # 计算RRF分数: score(d) = Sigma 1/(k + rank)
        rrf_scores: dict[str, float] = {}

        for doc_id, rank in bm25_ranks.items():
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            )

        for doc_id, rank in vector_ranks.items():
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            )

        # 合并文档信息（按doc_id去重）
        all_docs: dict[str, dict] = {}
        for doc in bm25_results + vector_results:
            did = doc["doc_id"]
            if did not in all_docs:
                all_docs[did] = doc.copy()

        # 按RRF分数降序排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])

        results = []
        for did in sorted_ids[:top_k]:
            if did in all_docs:
                doc = all_docs[did]
                doc["bm25_rank"] = bm25_ranks.get(did)   # 未出现在该路则为None
                doc["vector_rank"] = vector_ranks.get(did)
                doc["rrf_score"] = round(rrf_scores[did], 6)
                results.append(doc)

        logger.info("RRF融合完成，返回%d篇文档", len(results))
        return results


# 向后兼容：历史代码 import HybridRetriever 时仍指向并行实现
HybridRetriever = ParallelHybridRetriever
