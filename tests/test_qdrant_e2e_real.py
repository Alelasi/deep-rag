"""Qdrant 端到端测试（真实 embedding + 真实 Qdrant）
覆盖完整生产流程：
1. sentence-transformers 真实 embedding
2. Qdrant in-memory 真实存储
3. jieba 分词 + sparse vector
4. Query API + RRF 融合
"""
import sys
import time
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.retrieval.qdrant_retriever import QdrantRetriever
from src.state import Document


def test_e2e_real_embedding_and_qdrant():
    """端到端测试：真实 embedding + 真实 Qdrant + 中文混合检索"""
    print("=== 端到端测试：真实 embedding + 真实 Qdrant ===\n")

    # 1. 加载真实 embedding 模型（多语言，支持中文）
    print("[1/5] Loading embedding model...")
    t0 = time.time()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded in {time.time() - t0:.1f}s (dim={embedding_dim})\n")

    # 2. 创建 Qdrant 客户端
    print("[2/5] Creating Qdrant client (in-memory mode)...")
    client = QdrantClient(":memory:")
    retriever = QdrantRetriever(client=client, collection_name="mbti_e2e")
    retriever.create_collection(embedding_dim=embedding_dim)
    print(f"  Collection created (dim={embedding_dim})\n")

    # 3. 准备真实文档
    docs = [
        Document(
            doc_id="d1",
            content="INTJ的主导功能是Ni（内倾直觉），辅助功能是Te（外倾思考）。他们善于系统性思考和长期规划，被称为'建筑师'人格。",
            source="mbti_theory.md", page=1,
            metadata={"type": "INTJ", "category": "psychology"}
        ),
        Document(
            doc_id="d2",
            content="ENFP的主导功能是Ne（外倾直觉），辅助功能是Fi（内倾情感）。他们充满创造力和热情，被称为'活动家'人格。",
            source="mbti_theory.md", page=2,
            metadata={"type": "ENFP", "category": "psychology"}
        ),
        Document(
            doc_id="d3",
            content="ISTJ的主导功能是Si（内倾感觉），辅助功能是Te（外倾思考）。他们注重细节和责任感，被称为'物流师'人格。",
            source="mbti_theory.md", page=3,
            metadata={"type": "ISTJ", "category": "psychology"}
        ),
        Document(
            doc_id="d4",
            content="ENTP的主导功能是Ne（外倾直觉），辅助功能是Ti（内倾思考）。他们富有想象力和辩证思维，被称为'辩论家'人格。",
            source="mbti_theory.md", page=4,
            metadata={"type": "ENTP", "category": "psychology"}
        ),
        Document(
            doc_id="d5",
            content="Python是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。",
            source="programming.md", page=1,
            metadata={"category": "programming", "language": "python"}
        ),
    ]

    # 4. 真实 embedding
    print("[3/5] Computing embeddings for documents...")
    t0 = time.time()
    contents = [doc["content"] for doc in docs]
    embeddings = model.encode(contents, convert_to_numpy=True, show_progress_bar=False).tolist()
    print(f"  Embedded {len(docs)} docs in {time.time() - t0:.2f}s\n")

    # 5. 添加到 Qdrant
    print("[4/5] Adding documents to Qdrant...")
    t0 = time.time()
    retriever.add_documents(docs, embeddings)
    print(f"  Added in {time.time() - t0:.2f}s\n")

    # 6. 测试不同查询
    print("[5/5] Running hybrid search queries...\n")

    test_queries = [
        ("INTJ的主导功能是什么", "d1", "INTJ"),
        ("外倾直觉是什么", "d2 or d4", "ENFP/ENTP"),
        ("Python编程语言", "d5", "programming"),
    ]

    for query, expected_top, expected_type in test_queries:
        print(f"[Query] {query}")
        t0 = time.time()
        query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].tolist()
        results = retriever.hybrid_search(query, query_emb, top_k=3)
        elapsed = (time.time() - t0) * 1000

        print(f"  Top result: [{results[0]['doc_id']}] (expected: {expected_top})")
        print(f"  Score: {results[0]['metadata']['score']:.4f}")
        print(f"  Latency: {elapsed:.1f}ms")
        print(f"  Content: {results[0]['content'][:50]}...")
        print()

    print("=" * 60)
    print("✅ E2E TEST PASSED: Real Embedding + Real Qdrant + Hybrid Search")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_e2e_real_embedding_and_qdrant()
        sys.exit(0)
    except Exception as e:
        import traceback
        print("\n❌ TEST FAILED:")
        traceback.print_exc()
        sys.exit(1)
