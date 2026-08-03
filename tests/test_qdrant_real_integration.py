"""Qdrant 真实集成测试（in-memory 模式）
验证完整的混合检索流程：
1. 真实 Qdrant 客户端（:memory: 模式，无需 Docker）
2. 中文文本 + jieba 分词
3. Dense + Sparse 向量
4. Query API + RRF 融合
"""
import sys
import pytest
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from src.retrieval.qdrant_retriever import QdrantRetriever
from src.state import Document

# 模块级标记：L2 集成测试
pytestmark = pytest.mark.L2


def test_real_hybrid_search_chinese():
    """真实混合检索：中文 MBTI 文档"""
    print("=== 真实集成测试：Qdrant Hybrid Search (中文) ===\n")

    # 1. 创建 in-memory Qdrant 客户端
    client = QdrantClient(":memory:")
    retriever = QdrantRetriever(client=client, collection_name="mbti_test")

    # 2. 创建集合（dense + sparse）
    retriever.create_collection(embedding_dim=384)
    print("✅ Collection created (dense + sparse vectors)\n")

    # 3. 准备中文测试文档
    docs = [
        Document(
            doc_id="d1",
            content="INTJ的主导功能是Ni（内倾直觉），辅助功能是Te（外倾思考）。他们善于系统性思考和长期规划。",
            source="mbti_theory.md",
            page=1,
            metadata={"type": "INTJ", "category": "psychology"}
        ),
        Document(
            doc_id="d2",
            content="ENFP的主导功能是Ne（外倾直觉），辅助功能是Fi（内倾情感）。他们充满创造力和热情。",
            source="mbti_theory.md",
            page=2,
            metadata={"type": "ENFP", "category": "psychology"}
        ),
        Document(
            doc_id="d3",
            content="ISTJ的主导功能是Si（内倾感觉），辅助功能是Te（外倾思考）。他们注重细节和责任感。",
            source="mbti_theory.md",
            page=3,
            metadata={"type": "ISTJ", "category": "psychology"}
        ),
    ]

    # 4. 生成 embedding（简化版：用随机向量模拟）
    import random
    random.seed(42)
    embeddings = [
        [random.random() for _ in range(384)] for _ in docs
    ]

    # 5. 添加文档
    retriever.add_documents(docs, embeddings)
    print(f"✅ Added {len(docs)} documents\n")

    # 6. 混合检索测试
    query = "INTJ的主导功能是什么"
    query_embedding = [random.random() for _ in range(384)]

    print(f"Query: {query}")
    results = retriever.hybrid_search(query, query_embedding, top_k=2)

    print(f"\n✅ Hybrid Search Results (RRF fusion):")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc['doc_id']}] score={doc['metadata']['score']:.4f}")
        print(f"     {doc['content'][:50]}...")
        print(f"     type={doc['metadata'].get('type', 'N/A')}")

    # 7. 验证结果
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert all("score" in doc["metadata"] for doc in results), "Missing scores"
    print("\n✅ All assertions passed")

    # 8. 清理
    retriever.delete_collection()
    print("✅ Collection deleted\n")


def test_real_metadata_filtering():
    """真实元数据过滤测试"""
    print("=== 真实集成测试：Metadata Filtering ===\n")

    client = QdrantClient(":memory:")
    retriever = QdrantRetriever(client=client, collection_name="filter_test")
    retriever.create_collection(embedding_dim=128)

    # 准备文档（不同类型）
    docs = [
        Document(
            doc_id="d1",
            content="Python 编程语言",
            source="tech.md",
            page=1,
            metadata={"category": "programming", "language": "python"}
        ),
        Document(
            doc_id="d2",
            content="心理学基础理论",
            source="psychology.md",
            page=1,
            metadata={"category": "psychology", "topic": "theory"}
        ),
    ]

    import random
    random.seed(100)
    embeddings = [[random.random() for _ in range(128)] for _ in docs]
    retriever.add_documents(docs, embeddings)

    # 过滤检索：只要 psychology 类别
    query_embedding = [random.random() for _ in range(128)]
    results = retriever.search(
        "心理学",
        query_embedding,
        top_k=5,
        filters={"category": "psychology"}
    )

    print(f"✅ Filtered results (category=psychology):")
    for doc in results:
        print(f"  [{doc['doc_id']}] {doc['content']}")
        assert doc["metadata"]["category"] == "psychology", "Filter failed"

    print("\n✅ Metadata filtering works correctly\n")
    retriever.delete_collection()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Qdrant Real Integration Tests (in-memory mode)")
    print("="*60 + "\n")

    try:
        test_real_hybrid_search_chinese()
        test_real_metadata_filtering()

        print("="*60)
        print("✅ ALL REAL INTEGRATION TESTS PASSED")
        print("="*60)
        sys.exit(0)

    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        traceback.print_exc()
        sys.exit(1)
