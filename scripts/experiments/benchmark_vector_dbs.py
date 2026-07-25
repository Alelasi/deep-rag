"""向量数据库性能对比测试

对比数据库：
1. FAISS (Facebook AI)
2. ChromaDB (轻量级)
3. LanceDB (列式存储)
4. Qdrant (Rust高性能)
5. pgvector (PostgreSQL扩展)

测试维度：
- 索引速度 (docs/s)
- 内存占用 (MB)
- 检索速度 (ms)
- 压缩比 (原始/压缩)
"""

import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List

# 测试数据准备
def prepare_test_data(limit: int = 1000) -> tuple:
    """准备测试数据（使用随机向量，无需模型）

    Args:
        limit: 文档数量限制

    Returns:
        (texts, vectors)
    """
    print(f"准备测试数据（{limit} 条）...")

    # 生成随机文本
    texts = [f"This is test document number {i} with some random content." for i in range(limit)]

    # 生成随机向量（384维，模拟 all-MiniLM-L6-v2）
    np.random.seed(42)
    vectors = np.random.randn(limit, 384).astype(np.float32)

    print(f"✅ 准备了 {len(texts)} 条文本")
    print(f"✅ 向量形状: {vectors.shape}")

    return texts, vectors


# 1. FAISS 测试
def benchmark_faiss(vectors: np.ndarray, query_vector: np.ndarray) -> Dict:
    """测试 FAISS"""
    import faiss

    print("\n=== FAISS 测试 ===")
    result = {}

    # 索引速度
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    start = time.time()
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    index_time = time.time() - start

    mem_after = process.memory_info().rss / 1024 / 1024

    result["index_time"] = index_time
    result["index_speed"] = len(vectors) / index_time
    result["memory_mb"] = mem_after - mem_before

    # 检索速度
    start = time.time()
    distances, indices = index.search(query_vector.reshape(1, -1).astype(np.float32), k=10)
    search_time = (time.time() - start) * 1000

    result["search_ms"] = search_time

    print(f"索引速度: {result['index_speed']:.0f} docs/s")
    print(f"内存占用: {result['memory_mb']:.1f} MB")
    print(f"检索速度: {result['search_ms']:.2f} ms")

    return result


# 2. ChromaDB 测试
def benchmark_chromadb(texts: List[str], vectors: np.ndarray, query_text: str) -> Dict:
    """测试 ChromaDB"""
    import chromadb
    from chromadb.config import Settings

    print("\n=== ChromaDB 测试 ===")
    result = {}

    # 索引速度
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.create_collection("benchmark")

    start = time.time()
    collection.add(
        ids=[str(i) for i in range(len(texts))],
        documents=texts,
        embeddings=vectors.tolist()
    )
    index_time = time.time() - start

    mem_after = process.memory_info().rss / 1024 / 1024

    result["index_time"] = index_time
    result["index_speed"] = len(texts) / index_time
    result["memory_mb"] = mem_after - mem_before

    # 检索速度
    start = time.time()
    results = collection.query(query_texts=[query_text], n_results=10)
    search_time = (time.time() - start) * 1000

    result["search_ms"] = search_time

    print(f"索引速度: {result['index_speed']:.0f} docs/s")
    print(f"内存占用: {result['memory_mb']:.1f} MB")
    print(f"检索速度: {result['search_ms']:.2f} ms")

    return result


# 3. LanceDB 测试
def benchmark_lancedb(texts: List[str], vectors: np.ndarray, query_vector: np.ndarray) -> Dict:
    """测试 LanceDB"""
    import lancedb
    import pyarrow as pa

    print("\n=== LanceDB 测试 ===")
    result = {}

    # 索引速度
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    db = lancedb.connect("data/benchmark_lancedb")

    # 准备数据
    data = []
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        data.append({"id": i, "text": text, "vector": vec})

    start = time.time()
    table = db.create_table("benchmark", data=data, mode="overwrite")
    index_time = time.time() - start

    mem_after = process.memory_info().rss / 1024 / 1024

    result["index_time"] = index_time
    result["index_speed"] = len(texts) / index_time
    result["memory_mb"] = mem_after - mem_before

    # 检索速度
    start = time.time()
    results = table.search(query_vector).limit(10).to_list()
    search_time = (time.time() - start) * 1000

    result["search_ms"] = search_time

    print(f"索引速度: {result['index_speed']:.0f} docs/s")
    print(f"内存占用: {result['memory_mb']:.1f} MB")
    print(f"检索速度: {result['search_ms']:.2f} ms")

    return result


# 4. Qdrant 测试
def benchmark_qdrant(texts: List[str], vectors: np.ndarray, query_vector: np.ndarray) -> Dict:
    """测试 Qdrant (in-memory模式)"""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        print("❌ Qdrant未安装，跳过测试")
        return {}

    print("\n=== Qdrant 测试 ===")
    result = {}

    # 索引速度
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    # 使用内存模式
    client = QdrantClient(":memory:")
    collection_name = "benchmark"

    # 创建集合
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
    )

    # 批量添加
    start = time.time()
    points = [
        PointStruct(id=i, vector=vec.tolist(), payload={"text": text})
        for i, (text, vec) in enumerate(zip(texts, vectors))
    ]
    client.upsert(collection_name=collection_name, points=points)
    index_time = time.time() - start

    mem_after = process.memory_info().rss / 1024 / 1024

    result["index_time"] = index_time
    result["index_speed"] = len(texts) / index_time
    result["memory_mb"] = mem_after - mem_before

    # 检索速度
    start = time.time()
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=10
    )
    search_time = (time.time() - start) * 1000

    result["search_ms"] = search_time

    print(f"索引速度: {result['index_speed']:.0f} docs/s")
    print(f"内存占用: {result['memory_mb']:.1f} MB")
    print(f"检索速度: {result['search_ms']:.2f} ms")

    return result


# 5. pgvector 测试
def benchmark_pgvector(texts: List[str], vectors: np.ndarray, query_vector: np.ndarray) -> Dict:
    """测试 pgvector (需要PostgreSQL)"""
    try:
        from src.retrieval.pgvector_retriever import PgvectorRetriever, PSYCOPG2_AVAILABLE
        if not PSYCOPG2_AVAILABLE:
            print("❌ psycopg2未安装，跳过测试")
            return {}
    except ImportError:
        print("❌ pgvector模块未找到，跳过测试")
        return {}

    print("\n=== pgvector 测试 ===")
    result = {}

    try:
        # 连接数据库
        retriever = PgvectorRetriever(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password="postgres",
            table_name="benchmark_test"
        )

        # 索引速度
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        # 创建表
        retriever.create_table(embedding_dim=vectors.shape[1])

        # 准备文档
        docs = [
            {"doc_id": str(i), "content": text, "source": "benchmark", "page": i, "metadata": {}}
            for i, text in enumerate(texts)
        ]

        start = time.time()
        retriever.add_documents(docs, vectors.tolist())
        index_time = time.time() - start

        mem_after = process.memory_info().rss / 1024 / 1024

        result["index_time"] = index_time
        result["index_speed"] = len(texts) / index_time
        result["memory_mb"] = mem_after - mem_before

        # 检索速度
        start = time.time()
        search_results = retriever.search(query_vector.tolist(), top_k=10)
        search_time = (time.time() - start) * 1000

        result["search_ms"] = search_time

        print(f"索引速度: {result['index_speed']:.0f} docs/s")
        print(f"内存占用: {result['memory_mb']:.1f} MB")
        print(f"检索速度: {result['search_ms']:.2f} ms")

        # 清理
        retriever.delete_collection()
        retriever.close()

    except Exception as e:
        print(f"❌ pgvector 测试失败: {e}")
        return {}

    return result


# 主测试流程
def main():
    """主测试流程"""
    print("=" * 60)
    print("向量数据库性能对比测试")
    print("=" * 60)

    # 准备数据
    texts, vectors = prepare_test_data(limit=1000)  # 使用1000条数据快速测试
    query_text = texts[0]
    query_vector = vectors[0]

    results = {}

    # 测试 FAISS
    try:
        results["FAISS"] = benchmark_faiss(vectors, query_vector)
    except Exception as e:
        print(f"❌ FAISS 测试失败: {e}")

    # 测试 ChromaDB
    try:
        results["ChromaDB"] = benchmark_chromadb(texts, vectors, query_text)
    except Exception as e:
        print(f"❌ ChromaDB 测试失败: {e}")

    # 测试 LanceDB
    try:
        results["LanceDB"] = benchmark_lancedb(texts, vectors, query_vector)
    except Exception as e:
        print(f"❌ LanceDB 测试失败: {e}")

    # 测试 Qdrant
    try:
        results["Qdrant"] = benchmark_qdrant(texts, vectors, query_vector)
    except Exception as e:
        print(f"❌ Qdrant 测试失败: {e}")

    # 测试 pgvector
    try:
        results["pgvector"] = benchmark_pgvector(texts, vectors, query_vector)
    except Exception as e:
        print(f"❌ pgvector 测试失败: {e}")

    # 输出对比表格
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    print(f"{'数据库':<15} {'索引速度':<20} {'内存占用':<15} {'检索速度':<15}")
    print("-" * 80)

    for db_name, result in results.items():
        if result:  # 跳过失败的测试
            print(f"{db_name:<15} {result['index_speed']:>15.0f} docs/s {result['memory_mb']:>10.1f} MB {result['search_ms']:>12.2f} ms")

    print("=" * 80)

    # 生成排名
    if results:
        print("\n" + "=" * 80)
        print("性能排名")
        print("=" * 80)

        # 索引速度排名
        index_ranking = sorted(
            [(name, r['index_speed']) for name, r in results.items() if r],
            key=lambda x: x[1],
            reverse=True
        )
        print("\n📊 索引速度排名（越高越好）：")
        for i, (name, speed) in enumerate(index_ranking, 1):
            print(f"  {i}. {name}: {speed:.0f} docs/s")

        # 检索速度排名
        search_ranking = sorted(
            [(name, r['search_ms']) for name, r in results.items() if r],
            key=lambda x: x[1]
        )
        print("\n⚡ 检索速度排名（越低越好）：")
        for i, (name, ms) in enumerate(search_ranking, 1):
            print(f"  {i}. {name}: {ms:.2f} ms")

        # 内存占用排名
        memory_ranking = sorted(
            [(name, r['memory_mb']) for name, r in results.items() if r],
            key=lambda x: x[1]
        )
        print("\n💾 内存占用排名（越低越好）：")
        for i, (name, mb) in enumerate(memory_ranking, 1):
            print(f"  {i}. {name}: {mb:.1f} MB")

        print("=" * 80)


if __name__ == "__main__":
    main()
