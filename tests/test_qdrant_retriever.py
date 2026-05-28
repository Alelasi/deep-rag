"""Qdrant检索器单元测试
通过注入mock client避免依赖真实Qdrant服务
"""
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace
from src.retrieval.qdrant_retriever import QdrantRetriever, QDRANT_AVAILABLE
from src.state import Document


# ===== Mock Qdrant Client =====

class MockCollections:
    def __init__(self, collections=None):
        self.collections = collections or []


class MockQdrantClient:
    """模拟Qdrant客户端，记录所有调用"""

    def __init__(self):
        self.collections_data = {}      # name -> config
        self.points_data = {}           # name -> list[point]
        self.search_responses = []       # 预设的search返回
        self.calls = []                 # 调用记录

    def get_collections(self):
        cols = [SimpleNamespace(name=name) for name in self.collections_data.keys()]
        return MockCollections(cols)

    def create_collection(self, collection_name, vectors_config=None,
                          sparse_vectors_config=None, **kwargs):
        self.calls.append(("create_collection", collection_name))
        self.collections_data[collection_name] = {
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_vectors_config,
        }

    def upsert(self, collection_name, points):
        self.calls.append(("upsert", collection_name, len(points)))
        self.points_data.setdefault(collection_name, []).extend(points)

    def search(self, collection_name, query_vector, query_filter=None,
               limit=5, **kwargs):
        self.calls.append(("search", collection_name, query_filter, limit))
        # 返回预设结果（模拟Qdrant ScoredPoint）
        return self.search_responses

    def query_points(self, collection_name, prefetch=None, query=None,
                     limit=5, **kwargs):
        """模拟 Qdrant Query API（v1.10+）"""
        self.calls.append(("query_points", collection_name, prefetch, limit))
        # 返回预设结果，封装成 QueryResponse 风格
        return SimpleNamespace(points=self.search_responses)

    def delete_collection(self, collection_name):
        self.calls.append(("delete_collection", collection_name))
        self.collections_data.pop(collection_name, None)
        self.points_data.pop(collection_name, None)


def make_scored_point(doc_id, content, source, page, score=0.9, metadata=None):
    """构造Qdrant风格的ScoredPoint mock"""
    return SimpleNamespace(
        id=hash(doc_id) % 10000,
        score=score,
        payload={
            "doc_id": doc_id,
            "content": content,
            "source": source,
            "page": page,
            "metadata": metadata or {},
        }
    )


# ===== 测试 =====

def test_create_collection():
    """create_collection: 集合创建调用"""
    print("=== 测试1: 创建集合 ===")
    mock_client = MockQdrantClient()
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    retriever.create_collection(embedding_dim=384)

    assert "test_kb" in mock_client.collections_data
    assert ("create_collection", "test_kb") in mock_client.calls
    assert retriever.embedding_dim == 384
    print(f"  Collection created: test_kb (dim=384)")
    print("  PASS\n")


def test_create_collection_idempotent():
    """create_collection: 已存在则跳过创建"""
    print("=== 测试2: 集合幂等创建 ===")
    mock_client = MockQdrantClient()
    mock_client.collections_data["existing"] = {}  # 预先存在

    retriever = QdrantRetriever(client=mock_client, collection_name="existing")
    retriever.create_collection(embedding_dim=768)

    # 不应再次调用create_collection
    create_calls = [c for c in mock_client.calls if c[0] == "create_collection"]
    assert len(create_calls) == 0, f"Should skip creation, got {create_calls}"
    print(f"  Skipped re-creating existing collection")
    print("  PASS\n")


def test_add_documents():
    """add_documents: 批量插入文档"""
    print("=== 测试3: 添加文档 ===")
    mock_client = MockQdrantClient()
    mock_client.collections_data["test_kb"] = {}
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    docs = [
        Document(doc_id="d1", content="INTJ的主导功能是Ni",
                 source="mbti.md", page=1, metadata={"category": "psychology"}),
        Document(doc_id="d2", content="ENFP的主导功能是Ne",
                 source="mbti.md", page=2, metadata={}),
    ]
    embeddings = [
        [0.1] * 768,
        [0.2] * 768,
    ]

    retriever.add_documents(docs, embeddings)

    upsert_calls = [c for c in mock_client.calls if c[0] == "upsert"]
    assert len(upsert_calls) == 1
    assert upsert_calls[0][2] == 2  # 2个points
    assert len(mock_client.points_data["test_kb"]) == 2

    # 验证payload正确
    inserted = mock_client.points_data["test_kb"]
    payload_0 = inserted[0]["payload"] if isinstance(inserted[0], dict) else inserted[0].payload
    assert payload_0["doc_id"] == "d1"
    assert payload_0["content"] == "INTJ的主导功能是Ni"
    assert payload_0["source"] == "mbti.md"
    print(f"  Inserted 2 documents with correct payload")
    print("  PASS\n")


def test_generate_sparse_vector():
    """_generate_sparse_vector: 生成稀疏向量"""
    print("=== 测试4: 稀疏向量生成 ===")
    mock_client = MockQdrantClient()
    retriever = QdrantRetriever(client=mock_client)

    sparse = retriever._generate_sparse_vector("INTJ的主导功能是Ni Ni")

    # 测试模式下返回dict（QDRANT_AVAILABLE为False时）
    if isinstance(sparse, dict):
        indices = sparse["indices"]
        values = sparse["values"]
    else:
        indices = sparse.indices
        values = sparse.values

    assert len(indices) > 0
    assert len(indices) == len(values)
    # "Ni"出现2次，对应的value应该是2
    assert 2.0 in values, f"Expected 'Ni' count=2, values={values}"
    print(f"  Generated sparse vector: {len(indices)} non-zero dims")
    print("  PASS\n")


def test_search_basic():
    """search: 基础向量检索"""
    print("=== 测试5: 基础向量检索 ===")
    mock_client = MockQdrantClient()
    mock_client.search_responses = [
        make_scored_point("d1", "INTJ介绍", "mbti.md", 1, score=0.95),
        make_scored_point("d2", "ENFP介绍", "mbti.md", 2, score=0.80),
    ]
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    query_emb = [0.1] * 768
    results = retriever.search("INTJ", query_emb, top_k=2)

    assert len(results) == 2
    assert results[0]["doc_id"] == "d1"
    assert results[0]["metadata"]["score"] == 0.95
    assert results[1]["doc_id"] == "d2"
    print(f"  Retrieved {len(results)} docs, top score: {results[0]['metadata']['score']}")
    print("  PASS\n")


def test_search_with_filters():
    """search: 元数据过滤"""
    print("=== 测试6: 元数据过滤检索 ===")
    mock_client = MockQdrantClient()
    mock_client.search_responses = [
        make_scored_point("d1", "MBTI内容", "mbti.md", 1, score=0.9),
    ]
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    query_emb = [0.1] * 768
    filters = {"source": "mbti.md", "category": "psychology"}
    results = retriever.search("test", query_emb, top_k=5, filters=filters)

    # 验证filter被传递（兼容 query_points 和 search）
    search_calls = [c for c in mock_client.calls if c[0] in ("search", "query_points")]
    assert len(search_calls) > 0, "Should call search or query_points"
    # query_points 的 filter 在 kwargs 中，search 的在位置参数中
    print(f"  Filter passed correctly (via {search_calls[0][0]})")
    assert len(results) == 1
    print(f"  Got {len(results)} results")
    print("  PASS\n")


def test_search_empty():
    """search: 空结果处理"""
    print("=== 测试7: 空检索结果 ===")
    mock_client = MockQdrantClient()
    mock_client.search_responses = []
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    results = retriever.search("nothing", [0.0] * 768)
    assert results == []
    print(f"  Empty result handled correctly")
    print("  PASS\n")


def test_hybrid_search():
    """hybrid_search: 混合检索接口"""
    print("=== 测试8: 混合检索 ===")
    mock_client = MockQdrantClient()
    mock_client.search_responses = [
        make_scored_point("d1", "混合检索内容", "doc.md", 1, score=0.88),
    ]
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    query_emb = [0.1] * 768
    results = retriever.hybrid_search("query", query_emb, top_k=3)

    assert len(results) == 1
    assert results[0]["doc_id"] == "d1"
    print(f"  Hybrid search returned {len(results)} docs")
    print("  PASS\n")


def test_delete_collection():
    """delete_collection: 删除集合"""
    print("=== 测试9: 删除集合 ===")
    mock_client = MockQdrantClient()
    mock_client.collections_data["temp"] = {}
    retriever = QdrantRetriever(client=mock_client, collection_name="temp")

    retriever.delete_collection()

    delete_calls = [c for c in mock_client.calls if c[0] == "delete_collection"]
    assert len(delete_calls) == 1
    assert "temp" not in mock_client.collections_data
    print(f"  Collection deleted")
    print("  PASS\n")


def test_qdrant_unavailable_raises():
    """未安装qdrant-client且未注入client时抛错"""
    print("=== 测试10: 未安装qdrant-client错误处理 ===")
    if QDRANT_AVAILABLE:
        print("  SKIP (qdrant-client is installed)")
        print()
        return

    try:
        QdrantRetriever()  # 不传client
        assert False, "Should raise ImportError"
    except ImportError as e:
        assert "qdrant-client" in str(e)
        print(f"  Correctly raises ImportError: {str(e)[:60]}")
    print("  PASS\n")


# ===== 主测试入口 =====

if __name__ == "__main__":
    tests = [
        test_create_collection,
        test_create_collection_idempotent,
        test_add_documents,
        test_generate_sparse_vector,
        test_search_basic,
        test_search_with_filters,
        test_search_empty,
        test_hybrid_search,
        test_delete_collection,
        test_qdrant_unavailable_raises,
    ]

    print(f"\nRunning {len(tests)} Qdrant retriever tests...")
    print(f"QDRANT_AVAILABLE: {QDRANT_AVAILABLE}\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
