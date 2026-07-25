"""seed_qdrant_cloud.py — 将本地 Qdrant collection 数据上传到 Qdrant Cloud

用途：Railway 在线 Demo 部署前，把本地知识库数据推到 Qdrant Cloud。

使用方式：
    python scripts/seed_qdrant_cloud.py

环境变量：
    QDRANT_LOCAL_PATH  — 本地 Qdrant 数据路径（默认 Windows 路径）
    QDRANT_CLOUD_URL   — Qdrant Cloud URL
    QDRANT_CLOUD_KEY   — Qdrant Cloud API Key
    QDRANT_COLLECTION  — 目标 collection 名称
"""
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def seed_qdrant_cloud():
    """从本地 Qdrant 读取数据，上传到 Qdrant Cloud。"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    local_path = os.getenv(
        "QDRANT_LOCAL_PATH",
        r"D:\文档\ai提问相关\哲思灵智\qdrant_data"
    )
    cloud_url = os.getenv("QDRANT_CLOUD_URL", "")
    cloud_key = os.getenv("QDRANT_CLOUD_KEY", "")
    collection_name = os.getenv("QDRANT_COLLECTION", "proj_psychology")

    if not cloud_url or not cloud_key:
        print("ERROR: 请设置 QDRANT_CLOUD_URL 和 QDRANT_CLOUD_KEY 环境变量")
        sys.exit(1)

    print(f"[1/4] 连接本地 Qdrant: {local_path}")
    local_client = QdrantClient(path=local_path)

    # 列出本地 collections
    collections = local_client.get_collections()
    print(f"  本地 collections: {[c.name for c in collections.collections]}")

    # 查找目标 collection
    target = None
    for c in collections.collections:
        if c.name == collection_name:
            target = c.name
            break

    if not target:
        print(f"  WARNING: 未找到 collection '{collection_name}'，使用第一个可用 collection")
        target = collections.collections[0].name if collections.collections else None

    if not target:
        print("ERROR: 本地无可用 collection")
        sys.exit(1)

    # 获取 collection 信息
    info = local_client.get_collection(target)
    total_points = info.points_count or 0
    vectors_config = info.config.params.vectors
    print(f"  目标 collection: {target} ({total_points} points)")

    if total_points == 0:
        print("ERROR: collection 为空，无数据可上传")
        sys.exit(1)

    print(f"\n[2/4] 连接 Qdrant Cloud: {cloud_url}")
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_key, timeout=60)

    # 在 Cloud 上创建 collection（如果不存在）
    cloud_collections = [c.name for c in cloud_client.get_collections().collections]
    if target not in cloud_collections:
        print(f"  在 Cloud 上创建 collection: {target}")
        cloud_client.create_collection(
            collection_name=target,
            vectors_config=vectors_config,
        )
        print(f"  Collection 创建成功")
    else:
        print(f"  Collection 已存在，将追加数据")

    print(f"\n[3/4] 上传数据（{total_points} points）...")
    batch_size = 500
    uploaded = 0

    # 使用 scroll API 分批读取本地数据
    offset = None
    while True:
        results, offset = local_client.scroll(
            collection_name=target,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not results:
            break

        points = [
            PointStruct(
                id=p.id,
                vector=p.vector if isinstance(p.vector, list) else list(p.vector.values())[0],
                payload=p.payload or {}
            )
            for p in results
        ]

        cloud_client.upsert(
            collection_name=target,
            points=points,
        )

        uploaded += len(points)
        print(f"  进度: {uploaded}/{total_points} ({uploaded * 100 // total_points}%)")

        if offset is None:
            break
        time.sleep(0.2)  # 避免 Cloud 限流

    print(f"\n[4/4] 验证上传结果...")
    cloud_info = cloud_client.get_collection(target)
    cloud_count = cloud_info.points_count or 0
    print(f"  Cloud collection '{target}': {cloud_count} points")
    print(f"  上传: {uploaded} | Cloud: {cloud_count}")

    if cloud_count >= uploaded * 0.95:
        print("\n✅ 种子数据上传成功！")
        print(f"   Railway Demo 可使用 collection: {target}")
    else:
        print("\n⚠️  上传数量不匹配，请检查 Cloud collection 状态")

    # 关闭连接
    local_client.close()
    cloud_client.close()


if __name__ == "__main__":
    seed_qdrant_cloud()
