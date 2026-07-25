"""
统一检索接口
整合: 分层RAG + 幻觉检测 + 查询优化 + 自适应策略
"""
from typing import List, Dict, Optional
from ..config import EMBEDDING_MODEL, DEVICE
from .hallucination_aware_retriever import HallucinationAwareRetriever
from .query_optimizer import QueryOptimizer


class UnifiedRetriever:
    """统一检索接口"""

    def __init__(
        self,
        collection_name: str = "full_docs",
        model_name: str = None,
        device: str = None,
        enable_query_optimization: bool = True,
        enable_hallucination_detection: bool = True,
        similarity_threshold: float = 0.5
    ):
        """
        初始化统一检索器

        Args:
            collection_name: 集合名称
            model_name: 模型名称
            device: 设备（None 则自动检测）
            enable_query_optimization: 启用查询优化
            enable_hallucination_detection: 启用幻觉检测
            similarity_threshold: 相似度阈值
        """
        # 核心检索器
        model = model_name or EMBEDDING_MODEL
        self.retriever = HallucinationAwareRetriever(
            collection_name=collection_name,
            model_name=model,
            device=device or DEVICE,
            similarity_threshold=similarity_threshold
        )

        # 查询优化器
        self.query_optimizer = QueryOptimizer() if enable_query_optimization else None

        # 配置
        self.enable_query_optimization = enable_query_optimization
        self.enable_hallucination_detection = enable_hallucination_detection

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "smart"
    ) -> Dict:
        """
        智能检索

        Args:
            query: 查询文本
            top_k: 返回文档数
            mode: 检索模式
                - "simple": 简单检索
                - "smart": 智能检索（查询优化+幻觉检测）
                - "expanded": 扩展检索（多查询融合）

        Returns:
            {
                'query': 原始查询,
                'results': 结果列表,
                'confidence': 置信度,
                'explanation': 解释,
                'optimized_query': 优化后的查询（如果启用）,
            }
        """
        result = {
            'query': query,
            'results': [],
            'confidence': 'unknown',
            'explanation': '',
        }

        # 查询优化
        if mode in ["smart", "expanded"] and self.enable_query_optimization:
            optimized = self.query_optimizer.optimize(query)
            result['optimized_query'] = optimized['rewritten']
            query_to_use = optimized['rewritten']
        else:
            query_to_use = query

        # 检索
        if mode == "expanded" and self.enable_query_optimization:
            # 扩展检索（多查询融合）
            results = self._expanded_retrieve(optimized['expanded'], top_k)
            result['results'] = results
            result['confidence'] = self._compute_confidence(results)
        else:
            # 普通检索
            retrieval_result = self.retriever.retrieve_with_explanation(
                query_to_use, top_k
            )
            result['results'] = retrieval_result['results']
            result['confidence'] = retrieval_result['confidence']
            result['explanation'] = retrieval_result['explanation']

        return result

    def _expanded_retrieve(self, queries: List[str], top_k: int) -> List[Dict]:
        """扩展检索（多查询融合）"""
        all_results = {}

        # 对每个查询检索
        for query in queries:
            results, _ = self.retriever.retrieve(query, top_k=top_k)
            for r in results:
                key = (r['source'], r['page'])
                if key not in all_results:
                    all_results[key] = r
                else:
                    # 取最高相似度
                    if r['similarity'] > all_results[key]['similarity']:
                        all_results[key] = r

        # 排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['similarity'],
            reverse=True
        )

        return sorted_results[:top_k]

    def _compute_confidence(self, results: List[Dict]) -> str:
        """计算置信度"""
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

    def batch_search(self, queries: List[str], top_k: int = 3) -> List[Dict]:
        """批量检索"""
        return [self.search(q, top_k) for q in queries]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if self.retriever.collection:
            return {
                'total_docs': self.retriever.collection.count(),
                'query_optimization': self.enable_query_optimization,
                'hallucination_detection': self.enable_hallucination_detection,
            }
        return {'total_docs': 0}
