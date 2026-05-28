"""Reranker重排序模块
实现两阶段检索（召回+精排）的精排阶段

理论依据（《AI Agent与RAG完整技术指南》）：
- 第一阶段：混合检索召回大量候选文档（top_k=20）
- 第二阶段：用Reranker模型对候选精排（top_k=5）
- 提升检索精度约10%

支持两种Reranker：
1. CrossEncoderReranker：基于sentence-transformers的cross-encoder（默认）
2. KeywordReranker：基于关键词重叠的简单重排序（无模型依赖，用于fallback）
"""
from typing import List, Tuple
from src.state import Document
import jieba
from collections import Counter

# 延迟导入：未安装sentence-transformers时仍可import本模块
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None  # type: ignore


class BaseReranker:
    """Reranker基类"""

    def rerank(self, query: str, documents: List[Document],
               top_k: int = 5) -> List[Document]:
        """对候选文档重新排序，返回top_k

        Args:
            query: 查询文本
            documents: 待重排的候选文档（来自第一阶段召回）
            top_k: 返回的top文档数

        Returns:
            重排后的top_k文档（按相关性降序）
        """
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """基于Cross-Encoder的精排器

    Cross-Encoder vs Bi-Encoder：
    - Bi-Encoder（用于召回）：分别编码query和doc，计算余弦相似度。快但精度一般。
    - Cross-Encoder（用于精排）：将[query, doc]拼接后编码。慢但精度高。

    推荐模型：
    - BAAI/bge-reranker-base（中文友好，238MB）
    - BAAI/bge-reranker-large（中文最强，1.3GB）
    - cross-encoder/ms-marco-MiniLM-L-6-v2（英文小模型，22MB）
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", model=None):
        """初始化Cross-Encoder

        Args:
            model_name: HuggingFace模型ID
            model: 可选，注入已加载的模型（便于测试时注入mock）
        """
        if not CROSS_ENCODER_AVAILABLE and model is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        if model is not None:
            self.model = model
        else:
            self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document],
               top_k: int = 5) -> List[Document]:
        if not documents:
            return []

        # 构造[query, doc_content]对
        pairs = [(query, doc["content"]) for doc in documents]

        # Cross-encoder预测相关性分数（数值越大越相关）
        scores = self.model.predict(pairs)

        # 排序并返回top_k
        scored_docs: List[Tuple[float, Document]] = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: -float(x[0]))

        results = []
        for score, doc in scored_docs[:top_k]:
            new_doc = dict(doc)
            metadata = dict(new_doc.get("metadata") or {})
            metadata["rerank_score"] = float(score)
            new_doc["metadata"] = metadata
            results.append(new_doc)
        return results


class KeywordReranker(BaseReranker):
    """基于关键词重叠的简单重排序器（无模型依赖）

    用作CrossEncoder不可用时的fallback。
    评分规则：query与doc的关键词重叠数 / query关键词数（Jaccard变种）
    """

    def rerank(self, query: str, documents: List[Document],
               top_k: int = 5) -> List[Document]:
        if not documents:
            return []

        query_tokens = set(jieba.cut(query))
        # 过滤单字（噪音）和空token
        query_tokens = {t for t in query_tokens if len(t.strip()) >= 2}

        scored_docs: List[Tuple[float, Document]] = []
        for doc in documents:
            doc_tokens = set(jieba.cut(doc["content"]))
            doc_tokens = {t for t in doc_tokens if len(t.strip()) >= 2}

            # 计算重叠率
            if not query_tokens:
                score = 0.0
            else:
                overlap = len(query_tokens & doc_tokens)
                score = overlap / len(query_tokens)

            scored_docs.append((score, doc))

        # 排序
        scored_docs.sort(key=lambda x: -x[0])

        results = []
        for score, doc in scored_docs[:top_k]:
            new_doc = dict(doc)
            metadata = dict(new_doc.get("metadata") or {})
            metadata["rerank_score"] = score
            new_doc["metadata"] = metadata
            results.append(new_doc)
        return results


def get_reranker(use_cross_encoder: bool = False,
                 model_name: str = "BAAI/bge-reranker-base") -> BaseReranker:
    """工厂函数：根据可用性自动选择Reranker

    Args:
        use_cross_encoder: 是否优先使用CrossEncoder（需要安装sentence-transformers）
        model_name: CrossEncoder模型名

    Returns:
        Reranker实例。CrossEncoder不可用时降级为KeywordReranker
    """
    if use_cross_encoder and CROSS_ENCODER_AVAILABLE:
        return CrossEncoderReranker(model_name=model_name)
    return KeywordReranker()


def two_stage_retrieve(retriever, query: str,
                        recall_k: int = 20, rerank_k: int = 5,
                        reranker: BaseReranker = None) -> List[Document]:
    """两阶段检索：召回 + 精排

    Args:
        retriever: 第一阶段检索器（如HybridRetriever）
        query: 查询文本
        recall_k: 召回阶段返回数（应较大）
        rerank_k: 精排阶段返回数（应较小）
        reranker: 精排器，None时自动选择

    Returns:
        精排后的top rerank_k文档
    """
    # 第一阶段：召回
    candidates = retriever.retrieve(query, top_k=recall_k)
    if not candidates:
        return []

    # 第二阶段：精排
    if reranker is None:
        reranker = get_reranker(use_cross_encoder=False)
    return reranker.rerank(query, candidates, top_k=rerank_k)
