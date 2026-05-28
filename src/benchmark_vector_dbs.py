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
from sentence_transformers import SentenceTransformer

# 测试数据准备
def prepare_test_data(limit: int = 10000) -> tuple:
    """准备测试数据

    Args:
        limit: 文档数量限制

    Returns:
        (texts, vectors)
    """
    print(f"准备测试数据（限制 {limit} 条）...")

    # 读取工作目录的文档
    texts = []
    root = Path("D:/文档/ai提问相关/工作")

    for file in root.rglob("*.md"):
        if len(texts) >= limit:
            break
        try:
            content = file.read_text(encoding="utf-8")
            # 简单分块
            chunks = [content[i:i+500] for i in range(0, len(content), 500)]
            texts.extend(chunks[:100])  # 每个文件最多100块
        except:
            continue

    texts = texts[:limit]
    print(f"✅ 准备了 {len(texts)} 条文本")

    # 生成向量
    print("生成向量...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    vectors = embedder.encode(texts, convert_to_numpy=True, batch_size=512)
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


# 主测试流程
def main():
    """主测试流程"""
    print("=" * 60)
    print("向量数据库性能对比测试")
    print("=" * 60)

    # 准备数据
    texts, vectors = prepare_test_data(limit=10000)
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

    # 输出对比表格
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(f"{'数据库':<15} {'索引速度':<15} {'内存占用':<15} {'检索速度':<15}")
    print("-" * 60)

    for db_name, result in results.items():
        print(f"{db_name:<15} {result['index_speed']:>10.0f} docs/s {result['memory_mb']:>10.1f} MB {result['search_ms']:>10.2f} ms")

    print("=" * 60)


if __name__ == "__main__":
    main()
