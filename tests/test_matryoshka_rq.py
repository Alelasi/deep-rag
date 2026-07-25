"""测试 Matryoshka RQ 压缩索引

验证：
1. 索引文件大小
2. chunk 数量
3. 查询功能
4. 压缩比
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.matryoshka_rq_rag import MatryoshkaRQRAG

def test_index():
    """测试索引"""
    print("=" * 60)
    print("Matryoshka RQ 索引验证")
    print("=" * 60)

    # 初始化
    rag = MatryoshkaRQRAG(db_path="../data/matryoshka_rq_rag")

    # 检查索引文件
    index_dir = Path("../data/matryoshka_rq_rag")
    if not index_dir.exists():
        print("❌ 索引目录不存在")
        return

    rq_codes_file = index_dir / "rq_codes.npy"
    if not rq_codes_file.exists():
        print("❌ RQ codes 文件不存在")
        return

    file_size_mb = rq_codes_file.stat().st_size / 1024 / 1024
    print(f"✅ RQ codes 文件大小: {file_size_mb:.2f} MB")

    # 检查元数据
    metadata_file = index_dir / "metadata.json"
    if not metadata_file.exists():
        print("❌ 元数据文件不存在")
        return

    import json
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total_chunks = len(metadata.get("chunks", {}))

    print(f"✅ 总 chunk 数: {total_chunks:,}")

    # 计算压缩比
    # 原始向量：total_chunks * 64 * 4 bytes (float32)
    original_size_mb = total_chunks * 64 * 4 / 1024 / 1024
    compression_ratio = original_size_mb / file_size_mb

    print(f"✅ 原始大小（估算）: {original_size_mb:.2f} MB")
    print(f"✅ 压缩比: {compression_ratio:.1f}:1")

    print("\n" + "=" * 60)
    print("测试查询功能")
    print("=" * 60)

    # 测试查询
    queries = [
        "GPU加速向量检索",
        "Agentic RAG",
        "Matryoshka压缩",
        "INTJ的主导功能",
        "Python异步编程"
    ]

    for query in queries:
        print(f"\n🔍 查询: {query}")
        results = rag.search(query, top_k=3)

        if not results:
            print("  ❌ 无结果")
            continue

        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            text = result.get("text", "")[:80]
            file_path = result.get("metadata", {}).get("file_path", "unknown")

            print(f"  {i}. Score: {score:.4f}")
            print(f"     File: {file_path}")
            print(f"     Text: {text}...")

    print("\n" + "=" * 60)
    print("✅ 验证完成")
    print("=" * 60)


if __name__ == "__main__":
    test_index()
