"""QdrantRetriever 单元测试（对齐当前 API，注入 mock client 避免依赖真实 Qdrant 服务）

当前实现（qdrant_retriever.py）：
- __init__ 自动确保 collection（768 维 cosine）
- add_documents(documents, embeddings, metadatas, ids)
- search(query_embedding, top_k)
- count / is_ready / clear
"""
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace
from src.retrieval.qdrant_retriever import QdrantRetriever, QDRANT_AVAILABLE, VECTOR_SIZE

# 模块级标记：L2 集成测试
pytestmark = pytest.mark.L2


# ===== Mock Qdrant Client =====

class MockCollections:
    def __init__(self, collections=None):
        self.collections = collections or []


class MockQdrantClient:
    """模拟Qdrant客户端，记录所有调用"""

    def __init__(self):
        self.collections_data = {}      # name -> config
        self.points_data = {}           # name -> list[point]
        self.search_responses = []       # 预设的 query_points 返回
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

    def query_points(self, collection_name, prefetch=None, query=None,
                     limit=5, **kwargs):
        """模拟 Qdrant Query API（v1.10+）"""
        self.calls.append(("query_points", collection_name, prefetch, limit))
        return SimpleNamespace(points=self.search_responses)

    def get_collection(self, collection_name):
        self.calls.append(("get_collection", collection_name))
        cfg = self.collections_data.get(collection_name, {})
        return SimpleNamespace(
            points_count=len(self.points_data.get(collection_name, []))
        )

    def delete_collection(self, collection_name):
        self.calls.append(("delete_collection", collection_name))
        self.collections_data.pop(collection_name, None)
        self.points_data.pop(collection_name, None)


def make_scored_point(doc_id, content, source, page, score=0.9):
    """构造 Qdrant 风格 ScoredPoint mock"""
    return SimpleNamespace(
        id=hash(doc_id) % 10000,
        score=score,
        payload={
            "doc_id": doc_id,
            "content": content,
            "source": source,
            "page": page,
        },
    )


# ===== 测试 =====

def test_init_creates_collection():
    """初始化时自动创建 768 维 cosine 集合"""
    mock_client = MockQdrantClient()
    QdrantRetriever(client=mock_client, collection_name="test_kb")

    assert "test_kb" in mock_client.collections_data
    cfg = mock_client.collections_data["test_kb"]
    assert cfg["vectors_config"].size == VECTOR_SIZE
    assert cfg["vectors_config"].distance == "Cosine"
    assert ("create_collection", "test_kb") in mock_client.calls


def test_init_skips_existing_collection():
    """集合已存在时不重复创建"""
    mock_client = MockQdrantClient()
    mock_client.collections_data["existing"] = {}
    QdrantRetriever(client=mock_client, collection_name="existing")

    create_calls = [c for c in mock_client.calls if c[0] == "create_collection"]
    assert create_calls == []


def test_add_documents_payload():
    """add_documents 批量 upsert，payload 字段完整"""
    mock_client = MockQdrantClient()
    mock_client.collections_data["test_kb"] = {}
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    docs = [
        "INTJ的主导功能是Ni",
        "ENFP的主导功能是Ne",
    ]
    embeddings = [[0.1] * 768, [0.2] * 768]
    metadatas = [{"source": "mbti.md", "page": 1, "category": "psychology"}, {"source": "mbti.md", "page": 2}]
    ids = ["d1", "d2"]

    retriever.add_documents(docs, embeddings, metadatas, ids)

    upsert_calls = [c for c in mock_client.calls if c[0] == "upsert"]
    assert len(upsert_calls) == 1
    assert upsert_calls[0][2] == 2
    points = mock_client.points_data["test_kb"]
    assert len(points) == 2

    p0 = points[0]
    payload = p0.payload
    assert payload["doc_id"] == "d1"
    assert payload["content"] == "INTJ的主导功能是Ni"
    assert payload["source"] == "mbti.md"
    assert payload["page"] == 1
    assert payload["category"] == "psychology"


def test_search_mapping():
    """search 返回统一文档字典（含分数）"""
    mock_client = MockQdrantClient()
    mock_client.search_responses = [
        make_scored_point("d1", "INTJ介绍", "mbti.md", 1, score=0.95),
        make_scored_point("d2", "ENFP介绍", "mbti.md", 2, score=0.80),
    ]
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    results = retriever.search([0.1] * 768, top_k=2)

    assert len(results) == 2
    assert results[0]["doc_id"] == "d1"
    assert results[0]["_score"] == 0.95
    assert results[0]["_vector_distance"] == pytest.approx(0.05)
    assert results[0]["metadata"]["content"] == "INTJ介绍"
    assert ("query_points", "test_kb") in [c[:2] for c in mock_client.calls]


def test_search_empty():
    """search 空结果返回空列表"""
    mock_client = MockQdrantClient()
    mock_client.search_responses = []
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    assert retriever.search([0.0] * 768) == []


def test_count_and_is_ready():
    """count / is_ready 反映集合内点数"""
    mock_client = MockQdrantClient()
    mock_client.collections_data["test_kb"] = {}
    mock_client.points_data["test_kb"] = [SimpleNamespace()] * 3
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    assert retriever.count() == 3
    assert retriever.is_ready() is True


def test_clear_recreates_collection():
    """clear 删除并重建空集合"""
    mock_client = MockQdrantClient()
    mock_client.collections_data["test_kb"] = {}
    mock_client.points_data["test_kb"] = [SimpleNamespace()] * 2
    retriever = QdrantRetriever(client=mock_client, collection_name="test_kb")

    retriever.clear()

    assert ("delete_collection", "test_kb") in mock_client.calls
    assert "test_kb" in mock_client.collections_data
    assert retriever.count() == 0


def test_qdrant_available_flag():
    """QDRANT_AVAILABLE 为布尔值，且当前环境已安装 qdrant-client"""
    assert isinstance(QDRANT_AVAILABLE, bool)
    assert QDRANT_AVAILABLE is True
