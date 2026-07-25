"""
向量数据库写入性能对比测试
测试 ChromaDB、FAISS、LanceDB 的写入速度
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
print("🚀 向量数据库写入性能对比")
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
test_metadata = [{'source': f'test_{i}.txt', 'page': i % 10} for i in range(len(test_texts))]

# GPU向量化
print("🔥 GPU向量化...")
start = time.time()
embeddings = model.encode(
    test_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
gpu_time = time.time() - start
print(f"✅ 向量化完成: {gpu_time:.2f}秒 ({len(test_texts)/gpu_time:.0f} 文档/秒)")
print()

# ============================================================
# 测试 1: ChromaDB
# ============================================================
print("=" * 70)
print("测试 1: ChromaDB")
print("=" * 70)

try:
    import chromadb
    import shutil
    from pathlib import Path

    db_path = "./benchmark_chroma"
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection("benchmark")

    start = time.time()

    # 批量写入
    for i in range(NUM_BATCHES):
        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE

        collection.add(
            documents=test_texts[batch_start:batch_end],
            ids=test_ids[batch_start:batch_end],
            metadatas=test_metadata[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end].tolist()
        )

    chroma_time = time.time() - start
    chroma_speed = len(test_texts) / chroma_time

    print(f"✅ 写入完成: {chroma_time:.2f}秒")
    print(f"📊 速度: {chroma_speed:.0f} 文档/秒")
    print(f"📦 数据库大小: {sum(f.stat().st_size for f in Path(db_path).rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")

    # 清理
    shutil.rmtree(db_path)

except Exception as e:
    print(f"❌ ChromaDB 测试失败: {e}")
    chroma_time = float('inf')
    chroma_speed = 0

print()

# ============================================================
# 测试 2: FAISS
# ============================================================
print("=" * 70)
print("测试 2: FAISS")
print("=" * 70)

try:
    import faiss

    # 创建索引
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    start = time.time()

    # 批量写入
    index.add(embeddings)

    faiss_time = time.time() - start
    faiss_speed = len(test_texts) / faiss_time

    print(f"✅ 写入完成: {faiss_time:.2f}秒")
    print(f"📊 速度: {faiss_speed:.0f} 文档/秒")
    print(f"📦 索引大小: {index.ntotal * EMBEDDING_DIM * 4 / 1024 / 1024:.1f} MB")

except Exception as e:
    print(f"❌ FAISS 测试失败: {e}")
    faiss_time = float('inf')
    faiss_speed = 0

print()

# ============================================================
# 测试 3: LanceDB
# ============================================================
print("=" * 70)
print("测试 3: LanceDB")
print("=" * 70)

try:
    import lancedb
    import shutil
    from pathlib import Path

    db_path = "./benchmark_lance"
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    db = lancedb.connect(db_path)

    # 准备数据
    data = []
    for i in range(len(test_texts)):
        data.append({
            'id': test_ids[i],
            'text': test_texts[i],
            'vector': embeddings[i].tolist(),
            'metadata': str(test_metadata[i])
        })

    start = time.time()

    # 批量写入
    table = db.create_table("benchmark", data=data)

    lance_time = time.time() - start
    lance_speed = len(test_texts) / lance_time

    print(f"✅ 写入完成: {lance_time:.2f}秒")
    print(f"📊 速度: {lance_speed:.0f} 文档/秒")
    print(f"📦 数据库大小: {sum(f.stat().st_size for f in Path(db_path).rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")

    # 清理
    shutil.rmtree(db_path)

except Exception as e:
    print(f"❌ LanceDB 测试失败: {e}")
    lance_time = float('inf')
    lance_speed = 0

print()

# ============================================================
# 测试 4: Qdrant (内存模式)
# ============================================================
print("=" * 70)
print("测试 4: Qdrant (内存模式)")
print("=" * 70)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(":memory:")

    # 创建集合
    client.create_collection(
        collection_name="benchmark",
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
                'metadata': test_metadata[i]
            }
        ))

    # 分批上传
    for i in range(NUM_BATCHES):
        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE
        client.upsert(
            collection_name="benchmark",
            points=points[batch_start:batch_end]
        )

    qdrant_time = time.time() - start
    qdrant_speed = len(test_texts) / qdrant_time

    print(f"✅ 写入完成: {qdrant_time:.2f}秒")
    print(f"📊 速度: {qdrant_speed:.0f} 文档/秒")

except Exception as e:
    print(f"❌ Qdrant 测试失败: {e}")
    qdrant_time = float('inf')
    qdrant_speed = 0

print()

# ============================================================
# 总结
# ============================================================
print("=" * 70)
print("📊 性能对比总结")
print("=" * 70)

results = [
    ("ChromaDB", chroma_time, chroma_speed),
    ("FAISS", faiss_time, faiss_speed),
    ("LanceDB", lance_time, lance_speed),
    ("Qdrant", qdrant_time, qdrant_speed),
]

# 排序
results.sort(key=lambda x: x[1])

print(f"{'数据库':<15} {'写入时间':<12} {'速度 (文档/秒)':<20} {'相对速度'}")
print("-" * 70)

fastest_time = results[0][1]
for name, write_time, speed in results:
    if write_time == float('inf'):
        print(f"{name:<15} {'失败':<12} {'-':<20} {'-'}")
    else:
        speedup = fastest_time / write_time
        print(f"{name:<15} {write_time:>6.2f}秒     {speed:>8.0f} 文档/秒      {speedup:.1f}x")

print("=" * 70)
print(f"🏆 最快: {results[0][0]} ({results[0][2]:.0f} 文档/秒)")
print(f"🐌 最慢: {results[-1][0]} ({results[-1][2]:.0f} 文档/秒)")
print(f"⚡ 加速比: {results[-1][1] / results[0][1]:.1f}x")
print("=" * 70)
