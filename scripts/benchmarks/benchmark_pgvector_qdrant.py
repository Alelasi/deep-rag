"""
测试 pgvector 和 Qdrant 持久化模式性能
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 测试数据
BATCH_SIZE = 256
NUM_BATCHES = 20
EMBEDDING_DIM = 512

print("=" * 70)
print("🚀 pgvector & Qdrant 持久化性能测试")
print("=" * 70)
print(f"批次大小: {BATCH_SIZE}")
print(f"批次数量: {NUM_BATCHES}")
print(f"总文档数: {BATCH_SIZE * NUM_BATCHES:,}")
print()

# 加载模型
print("📦 加载GPU模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")
print("✅ 模型已加载")
print()

# 生成测试数据
print("📝 生成测试数据...")
test_texts = [f"这是测试文档 {i}，包含一些中文内容用于向量化测试。" * 10 for i in range(BATCH_SIZE * NUM_BATCHES)]
test_ids = [f"doc_{i}" for i in range(len(test_texts))]

# GPU向量化
print("🔥 GPU向量化...")
start = time.time()
embeddings = model.encode(
    test_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True
)
gpu_time = time.time() - start
print(f"✅ 向量化完成: {gpu_time:.2f}秒 ({len(test_texts)/gpu_time:.0f} 文档/秒)")
print()

results = []

# ============================================================
# 测试 1: pgvector
# ============================================================
print("=" * 70)
print("测试 1: pgvector (PostgreSQL)")
print("=" * 70)

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="postgres"
    )
    register_vector(conn)
    cur = conn.cursor()

    # 创建表
    cur.execute("DROP TABLE IF EXISTS benchmark_vectors")
    cur.execute(f"""
        CREATE TABLE benchmark_vectors (
            id TEXT PRIMARY KEY,
            text TEXT,
            embedding vector({EMBEDDING_DIM})
        )
    """)

    start = time.time()

    # 批量插入（使用 executemany）
    data = [(test_ids[i], test_texts[i], embeddings[i].tolist()) for i in range(len(test_texts))]
    cur.executemany(
        "INSERT INTO benchmark_vectors (id, text, embedding) VALUES (%s, %s, %s)",
        data
    )

    conn.commit()

    pgvector_time = time.time() - start
    pgvector_speed = len(test_texts) / pgvector_time

    print(f"✅ 写入完成: {pgvector_time:.2f}秒")
    print(f"📊 速度: {pgvector_speed:.0f} 文档/秒")

    results.append(("pgvector", pgvector_time, pgvector_speed))

    # 清理
    cur.execute("DROP TABLE benchmark_vectors")
    conn.close()

except Exception as e:
    print(f"❌ pgvector 测试失败: {e}")
    results.append(("pgvector", float('inf'), 0))

print()

# ============================================================
# 测试 2: Qdrant (持久化模式)
# ============================================================
print("=" * 70)
print("测试 2: Qdrant (持久化模式)")
print("=" * 70)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(host="localhost", port=6333)

    collection_name = "benchmark"

    # 删除旧集合
    try:
        client.delete_collection(collection_name)
    except:
        pass

    # 创建集合
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )

    start = time.time()

    # 批量写入
    points = []
    for i in range(len(test_texts)):
        points.append(PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                'text': test_texts[i],
            }
        ))

    # 分批上传
    for i in range(NUM_BATCHES):
        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE
        client.upsert(
            collection_name=collection_name,
            points=points[batch_start:batch_end]
        )

    qdrant_time = time.time() - start
    qdrant_speed = len(test_texts) / qdrant_time

    print(f"✅ 写入完成: {qdrant_time:.2f}秒")
    print(f"📊 速度: {qdrant_speed:.0f} 文档/秒")

    results.append(("Qdrant (持久化)", qdrant_time, qdrant_speed))

    # 清理
    client.delete_collection(collection_name)

except Exception as e:
    print(f"❌ Qdrant 测试失败: {e}")
    results.append(("Qdrant (持久化)", float('inf'), 0))

print()

# ============================================================
# 总结
# ============================================================
print("=" * 70)
print("📊 测试总结")
print("=" * 70)

if results:
    results.sort(key=lambda x: x[1])

    print(f"{'数据库':<25} {'写入时间':<12} {'速度 (文档/秒)'}")
    print("-" * 70)

    for name, write_time, speed in results:
        if write_time == float('inf'):
            print(f"{name:<25} {'失败':<12} {'-'}")
        else:
            print(f"{name:<25} {write_time:>6.2f}秒     {speed:>8.0f} 文档/秒")

print("=" * 70)
