"""Qdrant 端到端测试（真实 embedding + 真实 Qdrant in-memory）

覆盖生产流程：
1. sentence-transformers 真实 embedding（惰性导入，避免收集期 pyarrow 崩溃）
2. Qdrant in-memory 真实存储（384 维集合）
3. 向量检索 + payload 映射
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.retrieval.qdrant_retriever import QdrantRetriever

# 5 篇真实文档（content / source / page / metadata）
DOCS = [
    {
        "doc_id": "d1",
        "content": "INTJ的主导功能是Ni（内倾直觉），辅助功能是Te（外倾思考）。他们善于系统性思考和长期规划，被称为'建筑师'人格。",
        "source": "mbti_theory.md", "page": 1,
        "metadata": {"type": "INTJ", "category": "psychology"},
    },
    {
        "doc_id": "d2",
        "content": "ENFP的主导功能是Ne（外倾直觉），辅助功能是Fi（内倾情感）。他们充满创造力和热情，被称为'活动家'人格。",
        "source": "mbti_theory.md", "page": 2,
        "metadata": {"type": "ENFP", "category": "psychology"},
    },
    {
        "doc_id": "d3",
        "content": "ISTJ的主导功能是Si（内倾感觉），辅助功能是Te（外倾思考）。他们注重细节和责任感，被称为'物流师'人格。",
        "source": "mbti_theory.md", "page": 3,
        "metadata": {"type": "ISTJ", "category": "psychology"},
    },
    {
        "doc_id": "d4",
        "content": "ENTP的主导功能是Ne（外倾直觉），辅助功能是Ti（内倾思考）。他们富有想象力和辩证思维，被称为'辩论家'人格。",
        "source": "mbti_theory.md", "page": 4,
        "metadata": {"type": "ENTP", "category": "psychology"},
    },
    {
        "doc_id": "d5",
        "content": "Python是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。",
        "source": "programming.md", "page": 1,
        "metadata": {"category": "programming", "language": "python"},
    },
]


def test_e2e_real_embedding_and_qdrant():
    """端到端测试：真实 embedding + 真实 Qdrant + 向量检索"""
    # 惰性导入：sentence_transformers 导入链含 sklearn→pyarrow，
    # 在 pytest 收集期加载会触发 Windows access violation 崩溃
    from sentence_transformers import SentenceTransformer

    print("=== 端到端测试：真实 embedding + 真实 Qdrant ===\n")

    # 1. 加载真实 embedding 模型（多语言，支持中文）
    print("[1/4] Loading embedding model...")
    t0 = time.time()
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded in {time.time() - t0:.1f}s (dim={embedding_dim})\n")

    # 2. 创建 Qdrant 客户端 + 384 维集合（当前实现默认 768，手动建集合后检索器会复用）
    print("[2/4] Creating Qdrant client (in-memory mode)...")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="mbti_e2e",
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    retriever = QdrantRetriever(client=client, collection_name="mbti_e2e")
    print(f"  Collection ready (dim={embedding_dim})\n")

    # 3. 真实 embedding + 写入
    print("[3/4] Embedding and indexing documents...")
    t0 = time.time()
    contents = [d["content"] for d in DOCS]
    embeddings = model.encode(contents, convert_to_numpy=True, show_progress_bar=False).tolist()
    metadatas = [{"source": d["source"], "page": d["page"], **d["metadata"]} for d in DOCS]
    ids = [d["doc_id"] for d in DOCS]
    retriever.add_documents(contents, embeddings, metadatas, ids)
    print(f"  Indexed {len(DOCS)} docs in {time.time() - t0:.2f}s\n")

    # 4. 检索验证
    print("[4/4] Running search queries...\n")
    test_queries = [
        ("INTJ的主导功能是什么", "d1"),
        ("外倾直觉是什么", "d2"),  # Ne 相关，d2/d4 都可能命中
        ("Python编程语言", "d5"),
    ]
    for query, expected_top in test_queries:
        query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].tolist()
        results = retriever.search(query_emb, top_k=3)
        top = results[0] if results else {}
        print(f"[Query] {query}")
        print(f"  Top result: [{top.get('doc_id', 'N/A')}] (expected: {expected_top})")
        print(f"  Score: {top.get('_score', 0):.4f}")
        print(f"  Content: {top.get('content', '')[:50]}...\n")

        assert len(results) > 0, f"查询无结果: {query}"
        # 允许 d2/d4 互换（Ne 相关），其余严格匹配
        if expected_top == "d2":
            assert top["doc_id"] in ("d2", "d4"), f"Top={top['doc_id']}, expected d2/d4"
        else:
            assert top["doc_id"] == expected_top, f"Top={top['doc_id']}, expected {expected_top}"

    print("=" * 60)
    print("E2E PASSED: Real Embedding + Real Qdrant + Vector Search")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_e2e_real_embedding_and_qdrant()
        sys.exit(0)
    except Exception as e:
        import traceback
        print("\nTEST FAILED:")
        traceback.print_exc()
        sys.exit(1)
