"""向量数据库可视化管理 — 查看 Collection、文档预览、删除操作

支持 Qdrant（优先）和 ChromaDB（回退）。
"""
import os
from pathlib import Path
from typing import List, Dict, Optional


def _get_qdrant_client():
    """获取 Qdrant 客户端"""
    from src.retrieval.qdrant_retriever import get_qdrant_retriever
    return get_qdrant_retriever()


def _get_chroma_client():
    """获取 ChromaDB 客户端"""
    from ..config import get_chroma_client
    return get_chroma_client()


def _get_client():
    """获取向量数据库客户端（优先 Qdrant，回退 ChromaDB）"""
    try:
        return _get_qdrant_client()
    except Exception:
        return _get_chroma_client()


def get_db_info() -> dict:
    """获取数据库概要信息

    Returns:
        {db_path, total_collections, total_docs, db_size_mb}
    """
    try:
        # 尝试 Qdrant
        retriever = _get_qdrant_client()
        count = retriever.count()
        return {
            "db_path": "Qdrant (Docker)",
            "total_collections": 1,
            "total_docs": count,
            "db_size_mb": 0,  # Qdrant 不直接暴露大小
        }
    except Exception:
        # 回退到 ChromaDB
        try:
            from ..config import CHROMA_DB_PATH
            client = _get_chroma_client()
            collections = client.list_collections()
            total_docs = 0
            for col in collections:
                try:
                    total_docs += col.count()
                except Exception:
                    pass
            db_size_mb = 0
            if os.path.exists(CHROMA_DB_PATH):
                for root, dirs, files in os.walk(CHROMA_DB_PATH):
                    for f in files:
                        db_size_mb += os.path.getsize(os.path.join(root, f))
                db_size_mb = round(db_size_mb / (1024 * 1024), 2)
            return {
                "db_path": CHROMA_DB_PATH,
                "total_collections": len(collections),
                "total_docs": total_docs,
                "db_size_mb": db_size_mb,
            }
        except Exception:
            return {
                "db_path": "N/A",
                "total_collections": 0,
                "total_docs": 0,
                "db_size_mb": 0,
            }


def list_collections() -> List[Dict]:
    """列出所有 Collection

    Returns:
        [{name, count, sample_doc}]
    """
    try:
        # 尝试 Qdrant
        retriever = _get_qdrant_client()
        collections = retriever.client.get_collections().collections
        result = []
        for col in collections:
            info = {"name": col.name, "count": 0}
            try:
                col_info = retriever.client.get_collection(col.name)
                info["count"] = col_info.points_count
                # 获取一个样本文档
                scroll = retriever.client.scroll(
                    collection_name=col.name,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if scroll[0]:
                    payload = scroll[0][0].payload or {}
                    content = payload.get("content", "")
                    info["sample_doc"] = content[:200] + "..." if len(content) > 200 else content
                    info["sample_meta"] = {k: v for k, v in payload.items() if k != "content"}
            except Exception as e:
                info["error"] = str(e)
            result.append(info)
        return result
    except Exception:
        # 回退到 ChromaDB
        try:
            client = _get_chroma_client()
            collections = client.list_collections()
            result = []
            for col in collections:
                info = {"name": col.name, "count": 0}
                try:
                    info["count"] = col.count()
                    sample = col.get(limit=1, include=["documents", "metadatas"])
                    if sample["documents"]:
                        info["sample_doc"] = sample["documents"][0][:200] + "..." if len(sample["documents"][0]) > 200 else sample["documents"][0]
                    if sample["metadatas"]:
                        info["sample_meta"] = sample["metadatas"][0]
                except Exception as e:
                    info["error"] = str(e)
                result.append(info)
            return result
        except Exception:
            return []


def get_collection_docs(name: str, limit: int = 10) -> List[Dict]:
    """获取 Collection 中的文档预览

    Args:
        name: Collection 名称
        limit: 返回文档数

    Returns:
        [{id, document, metadata}]
    """
    try:
        # 尝试 Qdrant
        retriever = _get_qdrant_client()
        scroll = retriever.client.scroll(
            collection_name=name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        result = []
        for point in scroll[0]:
            payload = point.payload or {}
            result.append({
                "id": payload.get("doc_id", str(point.id)),
                "document": payload.get("content", "")[:500],
                "metadata": {k: v for k, v in payload.items() if k != "content"},
            })
        return result
    except Exception:
        # 回退到 ChromaDB
        try:
            client = _get_chroma_client()
            col = client.get_collection(name)
            data = col.get(limit=limit, include=["documents", "metadatas"])
            result = []
            for i, doc_id in enumerate(data["ids"]):
                result.append({
                    "id": doc_id,
                    "document": data["documents"][i][:500] if i < len(data["documents"]) and data["documents"][i] else "",
                    "metadata": data["metadatas"][i] if i < len(data["metadatas"]) else {},
                })
            return result
        except Exception as e:
            return [{"error": str(e)}]


def delete_collection(name: str) -> bool:
    """删除 Collection

    Args:
        name: Collection 名称

    Returns:
        是否成功
    """
    try:
        # 尝试 Qdrant
        retriever = _get_qdrant_client()
        retriever.client.delete_collection(name)
        return True
    except Exception:
        try:
            # 回退到 ChromaDB
            client = _get_chroma_client()
            client.delete_collection(name)
            return True
        except Exception:
            return False


def delete_document(collection_name: str, doc_id: str) -> bool:
    """删除单个文档

    Args:
        collection_name: Collection 名称
        doc_id: 文档 ID

    Returns:
        是否成功
    """
    client = _get_client()
    try:
        col = client.get_collection(collection_name)
        col.delete(ids=[doc_id])
        return True
    except Exception:
        return False


def reindex_collection(collection_name: str = "knowledge_base"):
    """重建 Collection（清空后重新索引）

    Args:
        collection_name: Collection 名称

    Returns:
        是否成功
    """
    client = _get_client()
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False
