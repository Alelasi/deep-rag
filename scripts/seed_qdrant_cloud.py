"""seed_qdrant_cloud.py — 将本地向量数据上传到 Qdrant Cloud

支持两种数据源：
  1. Qdrant 本地数据（DATA_SOURCE=qdrant）
  2. ChromaDB 服务器数据（DATA_SOURCE=chromadb）

使用方式：
    python scripts/seed_qdrant_cloud.py

环境变量：
    DATA_SOURCE        — 数据源：qdrant 或 chromadb（默认 qdrant）
    QDRANT_CLOUD_URL   — Qdrant Cloud URL（必填）
    QDRANT_CLOUD_KEY   — Qdrant Cloud API Key（必填）
    QDRANT_COLLECTION  — Cloud 目标 collection 名（默认 deep_rag_docs）
    SOURCE_COLLECTION  — 本地源 collection 名（默认同 QDRANT_COLLECTION）

    Qdrant 数据源额外：
    QDRANT_LOCAL_PATH  — 本地 Qdrant 数据路径

    ChromaDB 数据源额外：
    CHROMA_SERVER_HOST — ChromaDB 服务器地址（默认 localhost）
    CHROMA_SERVER_PORT — ChromaDB 服务器端口（默认 8000）
"""
import os
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _read_from_qdrant(local_path, source_collection):
    """从本地 Qdrant 读取数据，返回 (points_list, vector_dim)。"""
    from qdrant_client import QdrantClient

    print(f"  连接本地 Qdrant: {local_path}")
    client = QdrantClient(path=local_path)

    collections = [c.name for c in client.get_collections().collections]
    print(f"  本地 collections: {collections}")

    target = source_collection if source_collection in collections else (
        collections[0] if collections else None
    )
    if not target:
        client.close()
        print("ERROR: 本地无可用 collection")
        sys.exit(1)

    info = client.get_collection(target)
    total = info.points_count or 0
    vector_dim = None
    vc = info.config.params.vectors
    if hasattr(vc, "size"):
        vector_dim = vc.size
    elif isinstance(vc, dict):
        first = next(iter(vc.values()), None)
        vector_dim = getattr(first, "size", None) if first else None

    print(f"  源 collection: {target} ({total} points, dim={vector_dim})")
    if total == 0:
        client.close()
        print("ERROR: collection 为空")
        sys.exit(1)

    points = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=target, limit=500, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not results:
            break
        for p in results:
            vec = p.vector if isinstance(p.vector, list) else (
                list(p.vector.values())[0] if isinstance(p.vector, dict) else p.vector
            )
            points.append({"id": p.id, "vector": vec, "payload": p.payload or {}})
        if offset is None:
            break

    client.close()
    return points, vector_dim


def _read_from_chromadb(host, port, source_collection):
    """从 ChromaDB 服务器读取数据，返回 (points_list, vector_dim)。"""
    import chromadb

    print(f"  连接 ChromaDB 服务器: {host}:{port}")
    client = chromadb.HttpClient(host=host, port=port)

    collections = [c.name for c in client.list_collections()]
    print(f"  ChromaDB collections: {collections}")

    target = source_collection if source_collection in collections else (
        collections[0] if collections else None
    )
    if not target:
        print("ERROR: ChromaDB 无可用 collection")
        sys.exit(1)

    col = client.get_collection(target)
    total = col.count()
    print(f"  源 collection: {target} ({total} records)")

    if total == 0:
        print("ERROR: collection 为空")
        sys.exit(1)

    # 分批读取
    points = []
    batch = 500
    offset = 0
    vector_dim = None
    while offset < total:
        chunk = col.get(
            limit=batch, offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        ids = chunk.get("ids", [])
        embeddings = chunk.get("embeddings", [])
        metadatas = chunk.get("metadatas", [])
        documents = chunk.get("documents", [])

        if not ids:
            break

        for i, _id in enumerate(ids):
            vec = embeddings[i] if i < len(embeddings) else None
            if vec and vector_dim is None:
                vector_dim = len(vec)
            payload = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            if i < len(documents) and documents[i]:
                payload["document"] = documents[i]
            # 确保 ID 是有效 UUID（Qdrant 要求 int 或 UUID）
            try:
                uid = uuid.UUID(str(_id))
            except (ValueError, AttributeError):
                uid = uuid.uuid5(uuid.NAMESPACE_DNS, str(_id))
            points.append({"id": str(uid), "vector": vec, "payload": payload})

        offset += len(ids)
        print(f"  读取进度: {offset}/{total}")

    return points, vector_dim


def seed_qdrant_cloud():
    """主入口：从本地数据源读取，上传到 Qdrant Cloud。"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance

    data_source = os.getenv("DATA_SOURCE", "qdrant").lower()
    cloud_url = os.getenv("QDRANT_CLOUD_URL", "")
    cloud_key = os.getenv("QDRANT_CLOUD_KEY", "")
    cloud_collection = os.getenv("QDRANT_COLLECTION", "deep_rag_docs")
    source_collection = os.getenv("SOURCE_COLLECTION", cloud_collection)

    if not cloud_url or not cloud_key:
        print("ERROR: 请设置 QDRANT_CLOUD_URL 和 QDRANT_CLOUD_KEY 环境变量")
        sys.exit(1)

    # Step 1: 从本地读取数据
    print(f"\n[1/4] 从 {data_source} 读取数据...")
    if data_source == "chromadb":
        host = os.getenv("CHROMA_SERVER_HOST", "localhost")
        port = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
        points, vector_dim = _read_from_chromadb(host, port, source_collection)
    else:
        local_path = os.getenv(
            "QDRANT_LOCAL_PATH",
            r"D:\文档\ai提问相关\哲思灵智\qdrant_data",
        )
        points, vector_dim = _read_from_qdrant(local_path, source_collection)

    total = len(points)
    print(f"  读取完成: {total} points, vector_dim={vector_dim}")
    if total == 0:
        print("ERROR: 无数据可上传")
        sys.exit(1)

    # Step 2: 连接 Qdrant Cloud
    print(f"\n[2/4] 连接 Qdrant Cloud: {cloud_url}")
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_key, timeout=60)

    # 创建 collection（如果不存在）
    cloud_collections = [c.name for c in cloud_client.get_collections().collections]
    if cloud_collection not in cloud_collections:
        if vector_dim is None:
            vector_dim = 512  # 默认 bge-small-zh
        print(f"  创建 Cloud collection: {cloud_collection} (dim={vector_dim})")
        cloud_client.create_collection(
            collection_name=cloud_collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print("  Collection 创建成功")
    else:
        print(f"  Collection 已存在: {cloud_collection}")

    # Step 3: 分批上传
    print(f"\n[3/4] 上传数据（{total} points）...")
    batch_size = 200
    uploaded = 0
    for i in range(0, total, batch_size):
        batch = points[i:i + batch_size]
        qdrant_points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in batch if p["vector"] is not None
        ]
        if not qdrant_points:
            continue
        cloud_client.upsert(collection_name=cloud_collection, points=qdrant_points)
        uploaded += len(qdrant_points)
        pct = uploaded * 100 // total
        print(f"  进度: {uploaded}/{total} ({pct}%)")
        time.sleep(0.3)

    # Step 4: 验证
    print(f"\n[4/4] 验证上传结果...")
    cloud_info = cloud_client.get_collection(cloud_collection)
    cloud_count = cloud_info.points_count or 0
    print(f"  Cloud '{cloud_collection}': {cloud_count} points")
    print(f"  上传: {uploaded} | Cloud: {cloud_count}")

    if cloud_count >= uploaded * 0.95:
        print("\n✅ 种子数据上传成功！")
        print(f"   Railway Demo 可使用 collection: {cloud_collection}")
    else:
        print("\n⚠️  上传数量不匹配，请检查 Cloud collection 状态")

    cloud_client.close()


if __name__ == "__main__":
    seed_qdrant_cloud()
