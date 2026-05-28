"""Qdrant向量数据库检索器
基于理论文档中的Qdrant最佳实践实现
- 支持稠密向量（Dense Vector）检索
- 支持稀疏向量（Sparse Vector）用于BM25
- 支持混合检索（Hybrid Search）
- 支持元数据过滤（Metadata Filtering）

依赖：pip install qdrant-client
"""
from typing import List, Optional, Dict, Any
from collections import Counter
from src.state import Document

# 延迟导入：未安装qdrant-client时仍可import本模块（跳过）
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
        SparseVector, SparseVectorParams
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore

import jieba


class QdrantRetriever:
    """Qdrant向量检索器 - 支持混合检索"""

    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection_name: str = "deep_rag_docs", client=None):
        """初始化Qdrant客户端

        Args:
            host: Qdrant服务地址
            port: Qdrant服务端口
            collection_name: 集合名称
            client: 可选，注入已有客户端（便于测试时注入mock）
        """
        if not QDRANT_AVAILABLE and client is None:
            raise ImportError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            )
        self.client = client if client is not None else QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = 768  # 默认使用768维向量

    def create_collection(self, embedding_dim: int = 768):
        """创建集合（支持稠密+稀疏向量）"""
        self.embedding_dim = embedding_dim

        # 检查集合是否存在
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            print(f"Collection {self.collection_name} already exists")
            return

        if QDRANT_AVAILABLE:
            # 创建集合：稠密向量 + 稀疏向量（用于BM25）
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )
        else:
            # 测试场景：用dict代替
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": {"size": embedding_dim, "distance": "cosine"}},
                sparse_vectors_config={"sparse": {}},
            )
        print(f"Created collection: {self.collection_name}")

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """添加文档到Qdrant

        Args:
            documents: 文档列表（TypedDict）
            embeddings: 对应的稠密向量列表
        """
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # 生成稀疏向量（BM25风格）
            sparse_vector = self._generate_sparse_vector(doc["content"])

            payload = {
                "doc_id": doc["doc_id"],
                "content": doc["content"],
                "source": doc["source"],
                "page": doc["page"],
                "metadata": doc.get("metadata") or {},
            }

            if QDRANT_AVAILABLE:
                point = PointStruct(
                    id=i,
                    vector={
                        "dense": embedding,
                        "sparse": sparse_vector,
                    },
                    payload=payload,
                )
            else:
                # 测试场景：用普通dict代替PointStruct
                point = {
                    "id": i,
                    "vector": {"dense": embedding, "sparse": sparse_vector},
                    "payload": payload,
                }
            points.append(point)

        # 批量插入
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Added {len(points)} documents to Qdrant")

    def _generate_sparse_vector(self, text: str):
        """生成稀疏向量（基于词频的BM25风格）

        Returns:
            SparseVector对象，或在qdrant未安装时返回dict（便于测试）
        """
        # 分词
        tokens = list(jieba.cut(text))
        # 词频统计
        word_counts = Counter(tokens)

        # 转换为稀疏向量格式
        indices = []
        values = []
        for word, count in word_counts.items():
            # 使用词的hash作为索引（简化版）
            idx = hash(word) % 10000  # 限制在10000维内
            indices.append(idx)
            values.append(float(count))

        if QDRANT_AVAILABLE:
            return SparseVector(indices=indices, values=values)
        return {"indices": indices, "values": values}

    def search(self, query: str, query_embedding: List[float],
               top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """向量检索（仅稠密向量）

        使用 Qdrant Query API (v1.10+) 替代已废弃的 search() 方法。

        Args:
            query: 查询文本
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件，如 {"source": "mbti_theory.md"}
        """
        # 构建过滤器
        query_filter = None
        if filters:
            if QDRANT_AVAILABLE:
                conditions = [
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filters.items()
                ]
                query_filter = Filter(must=conditions)
            else:
                # 测试场景：直接传dict，由mock client自行处理
                query_filter = filters

        # 搜索：优先使用 Query API，降级到 search()（兼容 mock）
        if QDRANT_AVAILABLE:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="dense",
                query_filter=query_filter,
                limit=top_k,
            )
            results = response.points
        else:
            # 测试场景：mock client 仍用旧的 search() 接口
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_embedding),
                query_filter=query_filter,
                limit=top_k,
            )

        # 转换为Document对象
        documents = []
        for result in results:
            doc = Document(
                doc_id=result.payload["doc_id"],
                content=result.payload["content"],
                source=result.payload["source"],
                page=result.payload["page"],
                metadata={
                    **result.payload.get("metadata", {}),
                    "score": result.score
                }
            )
            documents.append(doc)

        return documents

    def hybrid_search(self, query: str, query_embedding: List[float],
                      top_k: int = 5, dense_weight: float = 0.6,
                      sparse_weight: float = 0.4) -> List[Document]:
        """混合检索（稠密向量 + 稀疏向量 + RRF融合）

        使用 Qdrant Query API (v1.10+) 实现真正的混合检索：
        1. Prefetch 阶段：并行检索 dense 和 sparse 向量
        2. Fusion 阶段：RRF (Reciprocal Rank Fusion) 融合结果

        Args:
            query: 查询文本
            query_embedding: 查询的稠密向量
            top_k: 返回结果数量
            dense_weight: 稠密向量权重（保留参数，RRF 自动平衡）
            sparse_weight: 稀疏向量权重（保留参数，RRF 自动平衡）

        Returns:
            融合后的文档列表（按 RRF 分数排序）
        """
        # 生成查询的稀疏向量
        query_sparse = self._generate_sparse_vector(query)

        if QDRANT_AVAILABLE:
            from qdrant_client.models import Prefetch, FusionQuery, Fusion

            # Query API: Prefetch + RRF 融合
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # 阶段1：稠密向量检索（语义相似）
                    Prefetch(query=query_embedding, using="dense", limit=top_k * 2),
                    # 阶段2：稀疏向量检索（关键词匹配）
                    Prefetch(query=query_sparse, using="sparse", limit=top_k * 2),
                ],
                # 阶段3：RRF 融合（自动平衡两种检索结果）
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
            )
            search_results = results.points
        else:
            # 测试模式：降级为单一 dense 检索（mock client 不支持 query_points）
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_embedding),
                query_filter=None,
                limit=top_k,
            )
            search_results = results

        # 转换为Document对象
        documents = []
        for result in search_results:
            doc = Document(
                doc_id=result.payload["doc_id"],
                content=result.payload["content"],
                source=result.payload["source"],
                page=result.payload["page"],
                metadata={
                    **result.payload.get("metadata", {}),
                    "score": result.score
                }
            )
            documents.append(doc)

        return documents

    def delete_collection(self):
        """删除集合"""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
