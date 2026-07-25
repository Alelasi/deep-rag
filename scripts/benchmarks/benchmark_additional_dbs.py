"""
补充测试：pgvector、Elasticsearch、Milvus、Weaviate
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
print("🚀 补充向量数据库性能测试")
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

results = []

# ============================================================
# 测试 1: pgvector (需要 PostgreSQL)
# ============================================================
print("=" * 70)
print("测试 1: pgvector")
print("=" * 70)

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector

    # 尝试连接本地 PostgreSQL
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

    # 批量插入
    for i in range(len(test_texts)):
        cur.execute(
            "INSERT INTO benchmark_vectors (id, text, embedding) VALUES (%s, %s, %s)",
            (test_ids[i], test_texts[i], embeddings[i].tolist())
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
    print("💡 提示: 需要安装 PostgreSQL 并启用 pgvector 扩展")
    results.append(("pgvector", float('inf'), 0))

print()

# ============================================================
# 测试 2: Elasticsearch
# ============================================================
print("=" * 70)
print("测试 2: Elasticsearch")
print("=" * 70)

try:
    from elasticsearch import Elasticsearch

    # 尝试连接本地 Elasticsearch
    es = Elasticsearch(["http://localhost:9200"])

    if not es.ping():
        raise Exception("无法连接到 Elasticsearch")

    index_name = "benchmark_vectors"

    # 删除旧索引
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # 创建索引
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
    )

    start = time.time()

    # 批量写入
    for i in range(len(test_texts)):
        es.index(
            index=index_name,
            id=test_ids[i],
            body={
                "text": test_texts[i],
                "embedding": embeddings[i].tolist()
            }
        )

    es.indices.refresh(index=index_name)

    es_time = time.time() - start
    es_speed = len(test_texts) / es_time

    print(f"✅ 写入完成: {es_time:.2f}秒")
    print(f"📊 速度: {es_speed:.0f} 文档/秒")

    results.append(("Elasticsearch", es_time, es_speed))

    # 清理
    es.indices.delete(index=index_name)

except Exception as e:
    print(f"❌ Elasticsearch 测试失败: {e}")
    print("💡 提示: 需要启动 Elasticsearch 服务")
    results.append(("Elasticsearch", float('inf'), 0))

print()

# ============================================================
# 测试 3: Milvus
# ============================================================
print("=" * 70)
print("测试 3: Milvus")
print("=" * 70)

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

    # 连接 Milvus
    connections.connect(host="localhost", port="19530")

    collection_name = "benchmark_vectors"

    # 删除旧集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # 创建集合
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)

    start = time.time()

    # 批量插入
    collection.insert([
        test_ids,
        test_texts,
        embeddings.tolist()
    ])

    collection.flush()

    milvus_time = time.time() - start
    milvus_speed = len(test_texts) / milvus_time

    print(f"✅ 写入完成: {milvus_time:.2f}秒")
    print(f"📊 速度: {milvus_speed:.0f} 文档/秒")

    results.append(("Milvus", milvus_time, milvus_speed))

    # 清理
    utility.drop_collection(collection_name)
    connections.disconnect("default")

except Exception as e:
    print(f"❌ Milvus 测试失败: {e}")
    print("💡 提示: 需要启动 Milvus 服务")
    results.append(("Milvus", float('inf'), 0))

print()

# ============================================================
# 测试 4: Weaviate
# ============================================================
print("=" * 70)
print("测试 4: Weaviate")
print("=" * 70)

try:
    import weaviate

    # 连接 Weaviate
    client = weaviate.Client("http://localhost:8080")

    class_name = "BenchmarkVectors"

    # 删除旧类
    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)

    # 创建类
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
        ]
    }
    client.schema.create_class(class_obj)

    start = time.time()

    # 批量写入
    with client.batch as batch:
        for i in range(len(test_texts)):
            batch.add_data_object(
                data_object={"text": test_texts[i]},
                class_name=class_name,
                uuid=test_ids[i],
                vector=embeddings[i].tolist()
            )

    weaviate_time = time.time() - start
    weaviate_speed = len(test_texts) / weaviate_time

    print(f"✅ 写入完成: {weaviate_time:.2f}秒")
    print(f"📊 速度: {weaviate_speed:.0f} 文档/秒")

    results.append(("Weaviate", weaviate_time, weaviate_speed))

    # 清理
    client.schema.delete_class(class_name)

except Exception as e:
    print(f"❌ Weaviate 测试失败: {e}")
    print("💡 提示: 需要启动 Weaviate 服务")
    results.append(("Weaviate", float('inf'), 0))

print()

# ============================================================
# 总结
# ============================================================
print("=" * 70)
print("📊 补充测试总结")
print("=" * 70)

if results:
    results.sort(key=lambda x: x[1])

    print(f"{'数据库':<20} {'写入时间':<12} {'速度 (文档/秒)'}")
    print("-" * 70)

    for name, write_time, speed in results:
        if write_time == float('inf'):
            print(f"{name:<20} {'失败':<12} {'-'}")
        else:
            print(f"{name:<20} {write_time:>6.2f}秒     {speed:>8.0f} 文档/秒")

print("=" * 70)
