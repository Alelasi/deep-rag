"""BM25检索器 — 基于rank_bm25 + jieba中文分词的稀疏检索模块

功能：
- 使用BM25Okapi算法进行关键词检索
- jieba中文分词 + 停用词过滤
- 索引存储在内存中，支持通过pickle持久化
- 支持动态更新（update/remove）文档

文档格式：
    {
        "doc_id": str,       # 文档唯一ID
        "content": str,      # 文档内容（也兼容"text"字段）
        "source": str,       # 来源
        "page": int,         # 页码
        "metadata": dict     # 元数据
    }
"""
import logging
import pickle
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 常用中文停用词表（覆盖高频虚词、代词、连词等）
DEFAULT_STOPWORDS = set(
    "的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 "
    "着 没有 看 好 自己 这 那 他 她 它 们 这个 那个 什么 怎么 为什么 "
    "哪里 哪个 可以 可 把 让 被 从 向 给 对 跟 与 及 或 但 如果 因为 "
    "所以 虽然 但是 而且 然后 就是 还是 已经 将要 正在 多少 几 "
    "这个 那个 这些 那些 的话 的话 是不是 不会 不能 不要 不是 没有"
    .split()
)


class BM25Retriever:
    """基于BM25Okapi的中文检索器

    使用jieba进行中文分词，支持停用词过滤。
    索引保存在内存中，可通过pickle序列化持久化。

    Attributes:
        documents: 文档列表
        bm25: BM25Okapi索引实例
        stopwords: 停用词集合
        _doc_id_index: doc_id到documents列表位置的映射
    """

    def __init__(self, documents: list[dict]):
        """初始化BM25检索器

        Args:
            documents: 文档列表，每个文档包含doc_id, content, source, page, metadata
        """
        self.documents: list[dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.stopwords = DEFAULT_STOPWORDS.copy()
        self._doc_id_index: dict[str, int] = {}  # doc_id -> 在documents中的位置

        if documents:
            self.update(documents)

    # ------------------------------------------------------------------ #
    #  内部工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_text(doc: dict) -> str:
        """获取文档文本，兼容content/text两种字段"""
        return doc.get("content") or doc.get("text") or ""

    def _tokenize(self, text: str) -> list[str]:
        """jieba中文分词 + 停用词过滤

        Args:
            text: 待分词文本

        Returns:
            分词后的token列表（已过滤停用词和空白字符）
        """
        tokens = jieba.cut(text)
        return [
            t.strip() for t in tokens
            if t.strip() and t.strip() not in self.stopwords
        ]

    def _build_index(self):
        """构建BM25索引"""
        if not self.documents:
            self.bm25 = None
            return

        corpus = [self._tokenize(self._get_text(doc)) for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)

        # 重建doc_id到列表位置的映射
        self._doc_id_index = {
            doc["doc_id"]: i for i, doc in enumerate(self.documents)
        }

    # ------------------------------------------------------------------ #
    #  核心方法
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25检索

        Args:
            query: 查询文本
            top_k: 返回的top文档数，默认20

        Returns:
            排序后的文档列表，每个文档附加"similarity"字段（BM25分数归一化到0-1）
        """
        if not self.bm25 or not self.documents:
            logger.warning("BM25索引为空，无法检索")
            return []

        try:
            tokens = self._tokenize(query)
            if not tokens:
                logger.warning("查询分词后为空: %s", query)
                return []

            scores = self.bm25.get_scores(tokens)

            # 按BM25分数降序排序
            indexed_scores = sorted(enumerate(scores), key=lambda x: -x[1])

            # 取top_k中分数大于阈值的结果
            # 注意：BM25Okapi在小语料上可能返回负分数（术语出现在所有文档中时IDF为负）
            # 所以阈值设为负值，只过滤极低分结果
            results = []
            top_scores = []
            for idx, score in indexed_scores[:top_k]:
                if score > -1.0:  # 过滤极低分（而非仅>0）
                    doc = self.documents[idx].copy()
                    doc["similarity"] = float(score)
                    results.append(doc)
                    top_scores.append(float(score))

            # 归一化到0-1（使用最大分数作为分母，最小分数作为偏移）
            if top_scores:
                max_score = max(top_scores)
                min_score = min(top_scores)
                if max_score > min_score:
                    for doc in results:
                        # 线性归一化：将分数映射到0-1区间
                        doc["similarity"] = round(
                            (doc["similarity"] - min_score) / (max_score - min_score), 6
                        )
                else:
                    # 所有分数相同，设为1.0
                    for doc in results:
                        doc["similarity"] = 1.0

            logger.info(
                "BM25检索完成: query='%s', 召回%d篇", query[:50], len(results)
            )
            return results

        except Exception as e:
            logger.error("BM25检索失败: %s", e, exc_info=True)
            return []

    def update(self, new_docs: list[dict]):
        """增量添加文档到索引

        已存在的doc_id会被更新，新doc_id会被追加。

        Args:
            new_docs: 新文档列表
        """
        try:
            for doc in new_docs:
                doc_id = doc.get("doc_id")
                if doc_id and doc_id in self._doc_id_index:
                    # 已存在则替换
                    idx = self._doc_id_index[doc_id]
                    self.documents[idx] = doc
                else:
                    self.documents.append(doc)

            self._build_index()
            logger.info("BM25索引更新完成，当前文档数: %d", len(self.documents))
        except Exception as e:
            logger.error("BM25索引更新失败: %s", e, exc_info=True)

    def remove(self, doc_ids: list[str]):
        """从索引中移除文档

        Args:
            doc_ids: 要移除的文档ID列表
        """
        try:
            remove_set = set(doc_ids)
            self.documents = [
                doc for doc in self.documents
                if doc.get("doc_id") not in remove_set
            ]
            self._build_index()
            logger.info("BM25索引移除完成，当前文档数: %d", len(self.documents))
        except Exception as e:
            logger.error("BM25索引移除失败: %s", e, exc_info=True)

    # ------------------------------------------------------------------ #
    #  持久化方法
    # ------------------------------------------------------------------ #

    def save(self, filepath: str):
        """通过pickle持久化索引到磁盘

        Args:
            filepath: 保存路径
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "stopwords": self.stopwords,
                }, f)
            logger.info("BM25索引已保存到: %s", filepath)
        except Exception as e:
            logger.error("BM25索引保存失败: %s", e, exc_info=True)

    def load(self, filepath: str):
        """从磁盘加载pickle索引

        Args:
            filepath: 加载路径
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.stopwords = data.get("stopwords", DEFAULT_STOPWORDS.copy())
            self._build_index()
            logger.info(
                "BM25索引已从 %s 加载，文档数: %d", filepath, len(self.documents)
            )
        except Exception as e:
            logger.error("BM25索引加载失败: %s", e, exc_info=True)
