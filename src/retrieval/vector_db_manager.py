"""双向量数据库管理器 — 同时管理 ChromaDB 和 Qdrant

实现「双写单读双删」策略：
  - 写入：同时写入 ChromaDB 和 Qdrant（双写），保证数据一致
  - 读取：仅从 active_backend 读取（单读），通过环境变量 VECTOR_DB 切换
  - 删除：同时从两个后端删除（双删）

后端说明：
  - ChromaDB：通过 src.retrieval.indexer.Indexer 管理（本地持久化，零部署成本）
  - Qdrant  ：通过 scripts.experiments.qdrant_retriever.QdrantRetriever 管理
              （需安装 qdrant-client 并运行 Qdrant 服务，import 时 try-except，不可用则跳过）
"""
import os
import hashlib
import logging
from typing import Optional

from src.config import (
    VECTOR_DB, QDRANT_HOST, QDRANT_PORT,
    EMBEDDING_MODEL, DEVICE, CHROMA_DB_PATH,
    get_embedding_dim, get_chroma_client,
)
from src.retrieval.indexer import Indexer
from src.state import Document

log = logging.getLogger("deeprag.vector_db_manager")

# 延迟导入 QdrantRetriever（不可用则跳过，不影响 ChromaDB 正常使用）
try:
    from scripts.experiments.qdrant_retriever import QdrantRetriever
    _QDRANT_RETRIEVER_AVAILABLE = True
except ImportError:
    QdrantRetriever = None
    _QDRANT_RETRIEVER_AVAILABLE = False
    log.info("QdrantRetriever 不可用，将仅使用 ChromaDB 后端")


class VectorDBManager:
    """双向量数据库管理器 — ChromaDB + Qdrant 双写单读

    用法示例::

        manager = VectorDBManager()

        # 双写
        manager.add_documents(
            [{"content": "...", "source": "doc.md", "page": 1}],
            collection="knowledge_base",
        )

        # 单读（从 active_backend）
        results = manager.search(query_embedding, collection="knowledge_base", top_k=5)

        # 双删
        manager.delete(["doc_id_1", "doc_id_2"], collection="knowledge_base")
    """

    def __init__(self):
        """初始化两个后端，通过环境变量 VECTOR_DB 选择主读后端

        环境变量：
          VECTOR_DB — 主读后端（chromadb / qdrant），默认 chromadb
        """
        # ---- ChromaDB 后端（通过 Indexer）----
        self._chroma_indexers: dict[str, Indexer] = {}
        self._chroma_client = None
        try:
            self._chroma_client = get_chroma_client()
        except Exception as e:
            log.error("ChromaDB 客户端初始化失败: %s", e)

        # ---- Qdrant 后端（通过 QdrantRetriever）----
        self._qdrant_retrievers: dict = {}  # collection_name -> QdrantRetriever
        self._qdrant_client = None
        self._qdrant_available = _QDRANT_RETRIEVER_AVAILABLE

        if self._qdrant_available:
            try:
                from qdrant_client import QdrantClient
                self._qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                log.info("Qdrant 客户端连接成功: %s:%s", QDRANT_HOST, QDRANT_PORT)
            except Exception as e:
                log.warning("Qdrant 客户端连接失败，Qdrant 后端不可用: %s", e)
                self._qdrant_available = False

        # ---- 活跃读后端 ----
        self.active_backend = VECTOR_DB.lower() if VECTOR_DB else "chromadb"

        # 如果 active_backend 是 qdrant 但 qdrant 不可用，降级到 chromadb
        if self.active_backend == "qdrant" and not self._qdrant_available:
            log.warning("active_backend=qdrant 但 Qdrant 不可用，降级到 chromadb")
            self.active_backend = "chromadb"

        # ---- 共享嵌入模型 ----
        self._embedder = None

        log.info(
            "VectorDBManager 初始化完成: active_backend='%s', qdrant_available=%s",
            self.active_backend, self._qdrant_available,
        )

    # ------------------------------------------------------------------
    #  内部工具方法
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """延迟加载 SentenceTransformer 嵌入模型（与 Indexer 共享缓存）"""
        if self._embedder is None:
            from src.ui.model_cache import get_embedding_model
            self._embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)
        return self._embedder

    def _get_chroma_indexer(self, collection: str) -> Indexer:
        """获取或创建指定集合的 ChromaDB Indexer 实例"""
        if collection not in self._chroma_indexers:
            self._chroma_indexers[collection] = Indexer(collection_name=collection)
        return self._chroma_indexers[collection]

    def _get_qdrant_retriever(self, collection: str):
        """获取或创建指定集合的 QdrantRetriever 实例

        Args:
            collection: 集合名称

        Returns:
            QdrantRetriever 实例，Qdrant 不可用时返回 None
        """
        if not self._qdrant_available:
            return None

        if collection not in self._qdrant_retrievers:
            try:
                retriever = QdrantRetriever(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    collection_name=collection,
                    client=self._qdrant_client,
                )
                # 创建集合（如果不存在）
                retriever.create_collection(embedding_dim=get_embedding_dim())
                self._qdrant_retrievers[collection] = retriever
            except Exception as e:
                log.error("QdrantRetriever 创建失败 [collection=%s]: %s", collection, e)
                return None

        return self._qdrant_retrievers[collection]

    @staticmethod
    def _generate_doc_id(doc: dict, index: int) -> str:
        """为文档生成唯一 ID（优先使用文档自带的 doc_id）"""
        if doc.get("doc_id"):
            return doc["doc_id"]
        content = doc.get("content", "")
        source = doc.get("source", f"doc_{index}")
        return hashlib.md5(f"{source}:{index}:{content[:50]}".encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    #  核心方法：双写 / 单读 / 双删
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[dict], collection: str):
        """双写：同时将文档添加到 ChromaDB 和 Qdrant

        Args:
            docs:       文档列表，每个 dict 至少包含 content 字段，
                        可选 doc_id / source / page / metadata
            collection: 集合名称
        """
        if not docs:
            log.warning("add_documents: docs 为空，跳过 [collection=%s]", collection)
            return

        # 生成 doc_ids 和 embeddings（只生成一次，双写复用）
        doc_ids = [self._generate_doc_id(d, i) for i, d in enumerate(docs)]
        try:
            embedder = self._get_embedder()
            embeddings = embedder.encode([d["content"] for d in docs]).tolist()
        except Exception as e:
            log.error("嵌入向量生成失败: %s", e)
            return

        # ---- 写入 ChromaDB ----
        chroma_success = self._add_to_chroma(docs, doc_ids, embeddings, collection)

        # ---- 写入 Qdrant ----
        qdrant_success = self._add_to_qdrant(docs, doc_ids, embeddings, collection)

        log.info(
            "双写完成 [collection=%s]: %d 条文档 → ChromaDB=%s, Qdrant=%s",
            collection, len(docs), chroma_success, qdrant_success,
        )

    def _add_to_chroma(self, docs, doc_ids, embeddings, collection: str) -> bool:
        """写入 ChromaDB（通过 Indexer 的 collection）"""
        try:
            indexer = self._get_chroma_indexer(collection)
            col = indexer.get_collection()
            col.add(
                documents=[d["content"] for d in docs],
                ids=doc_ids,
                metadatas=[
                    {"source": d.get("source", "unknown"), "page": d.get("page", 0)}
                    for d in docs
                ],
                embeddings=embeddings,
            )
            log.info("ChromaDB 写入成功: %d 条 [collection=%s]", len(docs), collection)
            return True
        except Exception as e:
            log.error("ChromaDB 写入失败 [collection=%s]: %s", collection, e)
            return False

    def _add_to_qdrant(self, docs, doc_ids, embeddings, collection: str) -> bool:
        """写入 Qdrant（通过 QdrantRetriever）"""
        if not self._qdrant_available:
            log.debug("Qdrant 不可用，跳过写入 [collection=%s]", collection)
            return False

        retriever = self._get_qdrant_retriever(collection)
        if retriever is None:
            return False

        try:
            # 转换为 Document TypedDict 格式
            documents = [
                Document(
                    doc_id=doc_ids[i],
                    content=d["content"],
                    source=d.get("source", "unknown"),
                    page=d.get("page", 0),
                    metadata=d.get("metadata", {}),
                )
                for i, d in enumerate(docs)
            ]
            retriever.add_documents(documents, embeddings)
            log.info("Qdrant 写入成功: %d 条 [collection=%s]", len(docs), collection)
            return True
        except Exception as e:
            log.error("Qdrant 写入失败 [collection=%s]: %s", collection, e)
            return False

    def search(self, query_embedding: list[float], collection: str,
               top_k: int = 5) -> list[Document]:
        """单读：从 active_backend 检索相似文档

        Args:
            query_embedding: 查询向量
            collection:      集合名称
            top_k:           返回结果数量

        Returns:
            Document 列表（按相似度降序）
        """
        if self.active_backend == "qdrant" and self._qdrant_available:
            return self._search_qdrant(query_embedding, collection, top_k)
        else:
            return self._search_chroma(query_embedding, collection, top_k)

    def _search_chroma(self, query_embedding, collection: str, top_k: int) -> list[Document]:
        """从 ChromaDB 检索"""
        try:
            indexer = self._get_chroma_indexer(collection)
            col = indexer.get_collection()

            if col.count() == 0:
                return []

            results = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, col.count()),
            )

            docs = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    content = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    # ChromaDB cosine distance → similarity score（越高越相似）
                    similarity = round(1.0 - float(distance), 4)
                    docs.append(Document(
                        doc_id=doc_id,
                        content=content,
                        source=meta.get("source", ""),
                        page=meta.get("page", 0),
                        metadata={"score": similarity, "backend": "chromadb"},
                    ))

            log.info("ChromaDB 检索: %d 条结果 [collection=%s]", len(docs), collection)
            return docs

        except Exception as e:
            log.error("ChromaDB 检索失败 [collection=%s]: %s", collection, e)
            return []

    def _search_qdrant(self, query_embedding, collection: str, top_k: int) -> list[Document]:
        """从 Qdrant 检索"""
        retriever = self._get_qdrant_retriever(collection)
        if retriever is None:
            log.warning("Qdrant 不可用，降级到 ChromaDB [collection=%s]", collection)
            return self._search_chroma(query_embedding, collection, top_k)

        try:
            # QdrantRetriever.search 需要 query 参数（仅用于接口一致性，dense 检索不使用）
            docs = retriever.search(
                query="",
                query_embedding=query_embedding,
                top_k=top_k,
            )
            # 标注来源后端
            for doc in docs:
                doc["metadata"]["backend"] = "qdrant"

            log.info("Qdrant 检索: %d 条结果 [collection=%s]", len(docs), collection)
            return docs

        except Exception as e:
            log.error("Qdrant 检索失败 [collection=%s]: %s，降级到 ChromaDB", collection, e)
            return self._search_chroma(query_embedding, collection, top_k)

    def delete(self, doc_ids: list[str], collection: str):
        """双删：同时从 ChromaDB 和 Qdrant 删除文档

        Args:
            doc_ids:   要删除的文档 ID 列表
            collection: 集合名称
        """
        if not doc_ids:
            log.warning("delete: doc_ids 为空，跳过 [collection=%s]", collection)
            return

        chroma_success = self._delete_from_chroma(doc_ids, collection)
        qdrant_success = self._delete_from_qdrant(doc_ids, collection)

        log.info(
            "双删完成 [collection=%s]: %d 条 → ChromaDB=%s, Qdrant=%s",
            collection, len(doc_ids), chroma_success, qdrant_success,
        )

    def _delete_from_chroma(self, doc_ids: list[str], collection: str) -> bool:
        """从 ChromaDB 删除文档"""
        try:
            indexer = self._get_chroma_indexer(collection)
            col = indexer.get_collection()
            col.delete(ids=doc_ids)
            log.info("ChromaDB 删除成功: %d 条 [collection=%s]", len(doc_ids), collection)
            return True
        except Exception as e:
            log.error("ChromaDB 删除失败 [collection=%s]: %s", collection, e)
            return False

    def _delete_from_qdrant(self, doc_ids: list[str], collection: str) -> bool:
        """从 Qdrant 删除文档（按 payload 中的 doc_id 过滤）"""
        if not self._qdrant_available or self._qdrant_client is None:
            log.debug("Qdrant 不可用，跳过删除 [collection=%s]", collection)
            return False

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # 构建 should 过滤条件：doc_id 匹配任意一个
            conditions = [
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                for doc_id in doc_ids
            ]
            self._qdrant_client.delete(
                collection_name=collection,
                points_selector=Filter(should=conditions),
            )
            log.info("Qdrant 删除成功: %d 条 [collection=%s]", len(doc_ids), collection)
            return True

        except Exception as e:
            log.error("Qdrant 删除失败 [collection=%s]: %s", collection, e)
            return False

    # ------------------------------------------------------------------
    #  集合管理
    # ------------------------------------------------------------------

    def update_collection(self, collection: str, additions: list, deletions: list):
        """动态更新集合：先删除旧文档，再添加新文档

        Args:
            collection: 集合名称
            additions:  要添加的文档列表 [{content, source, page, ...}]
            deletions:  要删除的文档 ID 列表 [doc_id, ...]
        """
        log.info(
            "更新集合 [collection=%s]: additions=%d, deletions=%d",
            collection, len(additions), len(deletions),
        )

        # 先删除
        if deletions:
            self.delete(deletions, collection)

        # 再添加
        if additions:
            self.add_documents(additions, collection)

        log.info("集合更新完成 [collection=%s]", collection)

    def list_collections(self) -> dict:
        """列出所有集合（ChromaDB + Qdrant）

        Returns:
            {
                "chromadb": [{"name": ..., "count": ...}, ...],
                "qdrant": [{"name": ..., "count": ...}, ...],
            }
        """
        result = {"chromadb": [], "qdrant": []}

        # ChromaDB 集合列表
        if self._chroma_client is not None:
            try:
                collections = self._chroma_client.list_collections()
                for col in collections:
                    info = {"name": col.name, "count": 0}
                    try:
                        info["count"] = col.count()
                    except Exception:
                        pass
                    result["chromadb"].append(info)
            except Exception as e:
                log.error("ChromaDB 列出集合失败: %s", e)

        # Qdrant 集合列表
        if self._qdrant_client is not None:
            try:
                response = self._qdrant_client.get_collections()
                for col_info in response.collections:
                    count = 0
                    try:
                        count_info = self._qdrant_client.count(
                            collection_name=col_info.name, exact=True
                        )
                        count = count_info.count
                    except Exception:
                        pass
                    result["qdrant"].append({"name": col_info.name, "count": count})
            except Exception as e:
                log.error("Qdrant 列出集合失败: %s", e)

        log.info(
            "集合列表: ChromaDB=%d, Qdrant=%d",
            len(result["chromadb"]), len(result["qdrant"]),
        )
        return result

    def get_stats(self) -> dict:
        """获取各 DB 统计信息

        Returns:
            {
                "active_backend": "chromadb" | "qdrant",
                "chromadb": {"collections": N, "total_docs": N, "db_path": ...},
                "qdrant": {"collections": N, "total_docs": N, "host": ..., "available": bool},
            }
        """
        stats = {
            "active_backend": self.active_backend,
            "chromadb": {"collections": 0, "total_docs": 0, "db_path": CHROMA_DB_PATH},
            "qdrant": {
                "collections": 0,
                "total_docs": 0,
                "host": f"{QDRANT_HOST}:{QDRANT_PORT}",
                "available": self._qdrant_available,
            },
        }

        # ChromaDB 统计
        if self._chroma_client is not None:
            try:
                collections = self._chroma_client.list_collections()
                stats["chromadb"]["collections"] = len(collections)
                total = 0
                for col in collections:
                    try:
                        total += col.count()
                    except Exception:
                        pass
                stats["chromadb"]["total_docs"] = total
            except Exception as e:
                log.error("ChromaDB 统计失败: %s", e)

        # Qdrant 统计
        if self._qdrant_client is not None:
            try:
                response = self._qdrant_client.get_collections()
                stats["qdrant"]["collections"] = len(response.collections)
                total = 0
                for col_info in response.collections:
                    try:
                        count_info = self._qdrant_client.count(
                            collection_name=col_info.name, exact=True
                        )
                        total += count_info.count
                    except Exception:
                        pass
                stats["qdrant"]["total_docs"] = total
            except Exception as e:
                log.error("Qdrant 统计失败: %s", e)

        log.info(
            "DB统计: ChromaDB(%d集合/%d文档), Qdrant(%d集合/%d文档)",
            stats["chromadb"]["collections"], stats["chromadb"]["total_docs"],
            stats["qdrant"]["collections"], stats["qdrant"]["total_docs"],
        )
        return stats
