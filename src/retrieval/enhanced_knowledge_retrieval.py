"""
增强版知识检索模块 - 对标截图架构
解决5个核心问题：
1. 添加问题拒识（Query Validation）- 前置过滤
2. 添加多路推理（Multi-Path Reasoning）- 并行检索
3. 集成重排序（Reranking）- 提升准确率
4. 优化混合检索（Hybrid Search）- RRF融合
5. 添加Web兜底（Fallback）- 提升覆盖率

架构对标：
┌──────────────┐
│ 问题拒识     │ ← 新增（前置过滤）
└──────┬───────┘
       ↓
┌──────────────┐
│ 规则引擎     │ ← 关键词匹配
│ 语义匹配     │ ← 向量检索
│ 多路推理     │ ← 新增（并行检索）
└──────┬───────┘
       ↓
┌──────────────┐
│ RRF融合      │ ← 优化（倒数排序融合）
│ Reranking    │ ← 集成（ColBERT/CrossEncoder）
└──────┬───────┘
       ↓
┌──────────────┐
│ Web兜底      │ ← 集成（低置信度触发）
└──────────────┘
"""
import logging
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


# ========== 1. 问题拒识模块（Query Validation）==========

class QueryIntentType(Enum):
    """查询意图类型"""
    KNOWLEDGE = "knowledge"      # 知识查询（正常处理）
    REALTIME = "realtime"        # 实时查询（需要Web检索）
    CHITCHAT = "chitchat"        # 闲聊（拒识）
    MALICIOUS = "malicious"      # 恶意查询（拒识）
    UNCLEAR = "unclear"          # 不清晰（需要澄清）


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool               # 是否有效
    intent: QueryIntentType      # 意图类型
    confidence: float            # 置信度（0-1）
    reason: str                  # 原因
    suggestion: Optional[str]    # 建议（如何改写）


class QueryValidator:
    """问题拒识器 - 前置过滤无效查询"""

    def __init__(self):
        # 拒识规则：闲聊关键词
        self.chitchat_keywords = [
            "你好", "谢谢", "再见", "天气", "你是谁",
            "几点了", "吃饭", "怎么样", "好的", "嗯"
        ]

        # 拒识规则：恶意关键词（prompt injection）
        self.malicious_keywords = [
            "忽略上述指令", "forget all", "ignore previous",
            "system prompt", "泄露密码", "获取密码"
        ]

        # 有效查询关键词（技术文档相关）
        self.valid_keywords = [
            "如何", "怎么", "什么是", "为什么", "能否",
            "配置", "安装", "使用", "报错", "问题"
        ]

    def validate(self, query: str) -> ValidationResult:
        """
        验证查询有效性

        Args:
            query: 查询文本

        Returns:
            ValidationResult: 验证结果
        """
        query_lower = query.lower()

        # 规则1：长度检查（过短/过长拒识）
        if len(query) < 3:
            return ValidationResult(
                is_valid=False,
                intent=QueryIntentType.UNCLEAR,
                confidence=1.0,
                reason="查询过短（<3字符）",
                suggestion="请提供更详细的问题描述"
            )

        if len(query) > 500:
            return ValidationResult(
                is_valid=False,
                intent=QueryIntentType.UNCLEAR,
                confidence=1.0,
                reason="查询过长（>500字符）",
                suggestion="请精简问题，聚焦核心需求"
            )

        # 规则2：恶意查询检测
        for keyword in self.malicious_keywords:
            if keyword in query_lower:
                return ValidationResult(
                    is_valid=False,
                    intent=QueryIntentType.MALICIOUS,
                    confidence=1.0,
                    reason=f"检测到恶意关键词：{keyword}",
                    suggestion=None
                )

        # 规则3：闲聊检测
        chitchat_count = sum(1 for kw in self.chitchat_keywords if kw in query)
        if chitchat_count >= 2 or query in self.chitchat_keywords:
            return ValidationResult(
                is_valid=False,
                intent=QueryIntentType.CHITCHAT,
                confidence=0.8,
                reason="检测到闲聊意图",
                suggestion="请提出技术相关的问题"
            )

        # 规则4：有效查询检测
        valid_count = sum(1 for kw in self.valid_keywords if kw in query)
        if valid_count > 0:
            return ValidationResult(
                is_valid=True,
                intent=QueryIntentType.KNOWLEDGE,
                confidence=0.9,
                reason="检测到有效技术查询",
                suggestion=None
            )

        # 默认：通过（置信度较低）
        return ValidationResult(
            is_valid=True,
            intent=QueryIntentType.KNOWLEDGE,
            confidence=0.6,
            reason="未检测到明确特征，允许通过",
            suggestion=None
        )


# ========== 2. 多路推理模块（Multi-Path Reasoning）==========

@dataclass
class RetrievalPath:
    """检索路径"""
    name: str                    # 路径名称
    method: str                  # 检索方法
    results: List[Dict]          # 检索结果
    score: float                 # 路径得分
    metadata: Dict               # 元数据


class MultiPathRetriever:
    """多路推理检索器 - 并行多种检索策略"""

    def __init__(self, base_retriever):
        """
        初始化多路检索器

        Args:
            base_retriever: 基础检索器（unified_retriever.UnifiedRetriever）
        """
        self.base_retriever = base_retriever

    def retrieve_multipath(
        self,
        query: str,
        top_k: int = 5,
        paths: List[str] = None
    ) -> Dict:
        """
        多路并行检索

        Args:
            query: 查询文本
            top_k: 每条路径返回文档数
            paths: 检索路径列表（默认：["simple", "smart", "expanded"]）

        Returns:
            {
                'query': 原始查询,
                'paths': 各路径结果,
                'merged_results': 融合后的结果,
                'best_path': 最佳路径
            }
        """
        if paths is None:
            paths = ["simple", "smart", "expanded"]

        log.info(f"[MultiPath] Retrieving with {len(paths)} paths: {paths}")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _retrieve_path(path_name):
            try:
                result = self.base_retriever.search(query, top_k=top_k, mode=path_name)
                path_score = self._compute_path_score(result)
                rp = RetrievalPath(
                    name=path_name,
                    method=f"{path_name} retrieval",
                    results=result['results'],
                    score=path_score,
                    metadata={
                        'confidence': result.get('confidence', 'unknown'),
                        'optimized_query': result.get('optimized_query', query)
                    }
                )
                log.info(f"[MultiPath] Path '{path_name}': {len(result['results'])} results, score={path_score:.2f}")
                return rp
            except Exception as e:
                log.error(f"[MultiPath] Path '{path_name}' failed: {e}")
                return None

        # 并行执行各路径检索
        path_results = [None] * len(paths)
        with ThreadPoolExecutor(max_workers=len(paths)) as executor:
            future_to_idx = {
                executor.submit(_retrieve_path, p): i
                for i, p in enumerate(paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                path_results[idx] = future.result()

        # 过滤掉失败的路径
        path_results = [p for p in path_results if p is not None]

        # 选择最佳路径
        best_path = max(path_results, key=lambda p: p.score) if path_results else None

        # 融合结果（RRF - Reciprocal Rank Fusion）
        merged_results = self._merge_results_rrf(path_results, top_k=top_k)

        return {
            'query': query,
            'paths': [
                {
                    'name': p.name,
                    'method': p.method,
                    'score': p.score,
                    'num_results': len(p.results),
                    'metadata': p.metadata
                }
                for p in path_results
            ],
            'merged_results': merged_results,
            'best_path': best_path.name if best_path else None
        }

    def _compute_path_score(self, result: Dict) -> float:
        """计算路径得分"""
        if not result['results']:
            return 0.0

        # 平均相似度
        avg_similarity = sum(r['similarity'] for r in result['results']) / len(result['results'])

        # 置信度加权
        confidence_weight = {
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8,
            'very_low': 0.6,
            'no_results': 0.0,
            'unknown': 1.0
        }
        weight = confidence_weight.get(result.get('confidence', 'unknown'), 1.0)

        return avg_similarity * weight

    def _merge_results_rrf(self, paths: List[RetrievalPath], top_k: int, k: int = 60) -> List[Dict]:
        """
        RRF融合（Reciprocal Rank Fusion）

        公式：RRF_score(doc) = Σ 1 / (k + rank_i)
        其中 rank_i 是文档在第i条路径中的排名

        Args:
            paths: 检索路径列表
            top_k: 返回文档数
            k: RRF常数（通常=60）

        Returns:
            融合后的结果列表
        """
        # 收集所有文档的RRF得分
        doc_scores = {}

        for path in paths:
            for rank, doc in enumerate(path.results, start=1):
                # 使用 (source, page) 作为唯一标识
                doc_id = (doc['source'], doc['page'])

                # RRF得分
                rrf_score = 1.0 / (k + rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'rrf_score': 0.0,
                        'paths': []
                    }

                doc_scores[doc_id]['rrf_score'] += rrf_score
                doc_scores[doc_id]['paths'].append(path.name)

        # 排序
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )

        # 构造结果
        merged = []
        for item in sorted_docs[:top_k]:
            doc = item['doc'].copy()
            doc['rrf_score'] = item['rrf_score']
            doc['merged_from_paths'] = item['paths']
            merged.append(doc)

        log.info(f"[RRF] Merged {len(merged)} documents from {len(paths)} paths")
        return merged


# ========== 3. 增强版知识检索器（主接口）==========

class EnhancedKnowledgeRetrieval:
    """
    增强版知识检索器
    整合：问题拒识 + 混合检索 + 多路推理 + 重排序 + Web兜底
    """

    def __init__(
        self,
        base_retriever,
        enable_validation: bool = True,
        enable_multipath: bool = True,
        enable_reranking: bool = True,
        enable_web_fallback: bool = True,
        similarity_threshold: float = 0.5
    ):
        """
        初始化增强检索器

        Args:
            base_retriever: 基础检索器（unified_retriever.UnifiedRetriever）
            enable_validation: 启用问题拒识
            enable_multipath: 启用多路推理
            enable_reranking: 启用重排序
            enable_web_fallback: 启用Web兜底
            similarity_threshold: 相似度阈值（低于此值触发Web兜底）
        """
        self.base_retriever = base_retriever
        self.validator = QueryValidator() if enable_validation else None
        self.multipath = MultiPathRetriever(base_retriever) if enable_multipath else None

        self.enable_validation = enable_validation
        self.enable_multipath = enable_multipath
        self.enable_reranking = enable_reranking
        self.enable_web_fallback = enable_web_fallback
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["simple", "enhanced", "multipath"] = "enhanced"
    ) -> Dict:
        """
        增强检索（完整流水线）

        Args:
            query: 查询文本
            top_k: 返回文档数
            mode: 检索模式
                - "simple": 简单检索（跳过所有增强）
                - "enhanced": 增强检索（问题拒识 + 重排序 + Web兜底）
                - "multipath": 多路推理检索（最强）

        Returns:
            {
                'query': 原始查询,
                'validation': 验证结果（如果启用）,
                'results': 检索结果,
                'confidence': 置信度,
                'used_web_fallback': 是否使用了Web兜底,
                'metadata': 元数据
            }
        """
        result = {
            'query': query,
            'validation': None,
            'results': [],
            'confidence': 'unknown',
            'used_web_fallback': False,
            'metadata': {}
        }

        # Step 1: 问题拒识（前置过滤）
        if mode != "simple" and self.enable_validation:
            validation = self.validator.validate(query)
            result['validation'] = {
                'is_valid': validation.is_valid,
                'intent': validation.intent.value,
                'confidence': validation.confidence,
                'reason': validation.reason,
                'suggestion': validation.suggestion
            }

            # 拒识：无效查询直接返回
            if not validation.is_valid:
                log.warning(f"[Validation] Query rejected: {validation.reason}")
                result['confidence'] = 'rejected'
                result['metadata']['rejection_reason'] = validation.reason
                return result

        # Step 2: 检索
        if mode == "multipath" and self.enable_multipath:
            # 多路推理检索
            multipath_result = self.multipath.retrieve_multipath(query, top_k=top_k)
            result['results'] = multipath_result['merged_results']
            result['metadata']['multipath'] = multipath_result['paths']
            result['metadata']['best_path'] = multipath_result['best_path']
        else:
            # 普通检索
            retrieval_result = self.base_retriever.search(
                query,
                top_k=top_k,
                mode="smart" if mode == "enhanced" else "simple"
            )
            result['results'] = retrieval_result['results']
            result['confidence'] = retrieval_result.get('confidence', 'unknown')

        # Step 3: 重排序（如果启用）
        if mode != "simple" and self.enable_reranking and result['results']:
            result['results'] = self._rerank(query, result['results'], top_k)

        # Step 4: Web兜底（低置信度时触发）
        if mode != "simple" and self.enable_web_fallback:
            need_fallback = (
                not result['results'] or
                (result['results'] and result['results'][0]['similarity'] < self.similarity_threshold)
            )

            if need_fallback:
                log.info(f"[Web Fallback] Triggered (low confidence)")
                web_results = self._web_fallback(query)
                result['results'].extend(web_results)
                result['used_web_fallback'] = True
                result['metadata']['fallback_reason'] = 'low_confidence'

        # 计算最终置信度
        if result['results']:
            result['confidence'] = self._compute_final_confidence(result['results'])

        return result

    def _rerank(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """重排序（ColBERT）"""
        try:
            from src.retrieval.reranker import colbert_rerank
            reranked = colbert_rerank(query, results, top_k=top_k)
            log.info(f"[Reranking] ColBERT reranked {len(results)} -> {len(reranked)} results")
            return reranked
        except ImportError:
            log.warning("[Reranking] ColBERT not available, skip reranking")
            return results[:top_k]

    def _web_fallback(self, query: str) -> List[Dict]:
        """Web兜底检索"""
        try:
            from src.retrieval.web_fallback import WebSearchFallback
            web_search = WebSearchFallback()
            web_results = web_search.search(query, top_k=3)
            log.info(f"[Web Fallback] Retrieved {len(web_results)} results")
            return web_results
        except Exception as e:
            log.error(f"[Web Fallback] Failed: {e}")
            return []

    def _compute_final_confidence(self, results: List[Dict]) -> str:
        """计算最终置信度"""
        if not results:
            return "no_results"

        avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results)

        if avg_similarity >= 0.7:
            return "high"
        elif avg_similarity >= 0.6:
            return "medium"
        elif avg_similarity >= self.similarity_threshold:
            return "low"
        else:
            return "very_low"

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'validation_enabled': self.enable_validation,
            'multipath_enabled': self.enable_multipath,
            'reranking_enabled': self.enable_reranking,
            'web_fallback_enabled': self.enable_web_fallback,
            'similarity_threshold': self.similarity_threshold,
        }


# ========== 便捷工厂函数 ==========

def create_enhanced_retriever(
    collection_name: str = "full_docs",
    model_name: str = None,
    device: str = None
) -> EnhancedKnowledgeRetrieval:
    """创建增强检索器"""
    from src.config import EMBEDDING_MODEL, DEVICE
    from src.retrieval.unified_retriever import UnifiedRetriever

    base_retriever = UnifiedRetriever(
        collection_name=collection_name,
        model_name=model_name or EMBEDDING_MODEL,
        device=device or DEVICE
    )

    return EnhancedKnowledgeRetrieval(
        base_retriever=base_retriever,
        enable_validation=True,
        enable_multipath=True,
        enable_reranking=True,
        enable_web_fallback=True,
        similarity_threshold=0.5
    )


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例：如何使用增强检索器

    # 1. 创建增强检索器
    retriever = create_enhanced_retriever()

    # 2. 简单检索
    query1 = "什么是LangChain的LCEL？"
    result1 = retriever.retrieve(query1, top_k=5, mode="simple")
    print(f"Simple: {len(result1['results'])} results, confidence={result1['confidence']}")

    # 3. 增强检索（问题拒识 + 重排序 + Web兜底）
    query2 = "LangChain 0.3有什么新特性？"
    result2 = retriever.retrieve(query2, top_k=5, mode="enhanced")
    print(f"Enhanced: {len(result2['results'])} results, confidence={result2['confidence']}")

    # 4. 多路推理检索（最强）
    query3 = "如何配置LangChain的API Key？"
    result3 = retriever.retrieve(query3, top_k=5, mode="multipath")
    print(f"Multipath: {len(result3['results'])} results, best_path={result3['metadata'].get('best_path')}")

    # 5. 问题拒识测试
    query4 = "你好，今天天气怎么样？"
    result4 = retriever.retrieve(query4, top_k=5, mode="enhanced")
    print(f"Chitchat: validation={result4['validation']}")
