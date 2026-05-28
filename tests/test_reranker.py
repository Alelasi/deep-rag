"""Reranker模块单元测试
覆盖CrossEncoderReranker（用mock model）、KeywordReranker、两阶段检索
"""
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    KeywordReranker,
    get_reranker,
    two_stage_retrieve,
    CROSS_ENCODER_AVAILABLE,
)
from src.state import Document


# ===== Mock组件 =====

class MockCrossEncoder:
    """模拟CrossEncoder模型"""
    def __init__(self, mock_scores=None):
        self.mock_scores = mock_scores or []
        self.last_pairs = None

    def predict(self, pairs):
        self.last_pairs = pairs
        if self.mock_scores:
            return self.mock_scores[:len(pairs)]
        # 默认返回每对的简单分数（基于doc长度）
        return [float(len(p[1])) for p in pairs]


class MockRetriever:
    """模拟第一阶段检索器"""
    def __init__(self, results):
        self.results = results
        self.last_top_k = None

    def retrieve(self, query, top_k=10):
        self.last_top_k = top_k
        return self.results[:top_k]


def make_doc(doc_id, content, source="t.md", page=1, metadata=None):
    return Document(doc_id=doc_id, content=content, source=source,
                    page=page, metadata=metadata or {})


# ===== 测试：KeywordReranker =====

def test_keyword_reranker_basic():
    """KeywordReranker: 基础重排序"""
    print("=== 测试1: KeywordReranker基础重排 ===")
    docs = [
        make_doc("d1", "ENFP是外向直觉情感型人格"),
        make_doc("d2", "INTJ的主导功能是内向直觉Ni"),
        make_doc("d3", "MBTI是一种人格分类工具"),
    ]
    reranker = KeywordReranker()
    results = reranker.rerank("INTJ的主导功能", docs, top_k=2)

    assert len(results) == 2
    # d2包含"INTJ"和"主导"和"功能"，应该排第一
    assert results[0]["doc_id"] == "d2", f"Expected d2 first, got {results[0]['doc_id']}"
    assert "rerank_score" in results[0]["metadata"]
    print(f"  Top-1: {results[0]['doc_id']} (score={results[0]['metadata']['rerank_score']:.2f})")
    print(f"  Top-2: {results[1]['doc_id']} (score={results[1]['metadata']['rerank_score']:.2f})")
    print("  PASS\n")


def test_keyword_reranker_empty():
    """KeywordReranker: 空输入处理"""
    print("=== 测试2: KeywordReranker空输入 ===")
    reranker = KeywordReranker()
    results = reranker.rerank("query", [], top_k=5)
    assert results == []
    print(f"  Empty input handled correctly")
    print("  PASS\n")


def test_keyword_reranker_top_k_limit():
    """KeywordReranker: top_k限制生效"""
    print("=== 测试3: KeywordReranker top_k限制 ===")
    docs = [make_doc(f"d{i}", f"内容{i} 关键词") for i in range(10)]
    reranker = KeywordReranker()
    results = reranker.rerank("关键词", docs, top_k=3)
    assert len(results) == 3
    print(f"  Limited to top_k=3 from 10 candidates")
    print("  PASS\n")


# ===== 测试：CrossEncoderReranker =====

def test_cross_encoder_with_mock():
    """CrossEncoderReranker: 使用mock model"""
    print("=== 测试4: CrossEncoderReranker mock测试 ===")
    docs = [
        make_doc("d1", "短文本"),
        make_doc("d2", "这是一个比较长的文本内容"),
        make_doc("d3", "中等长度"),
    ]
    # 预设分数：d2最高，d1次之，d3最低
    mock_model = MockCrossEncoder(mock_scores=[0.5, 0.95, 0.3])
    reranker = CrossEncoderReranker(model_name="mock", model=mock_model)

    results = reranker.rerank("query", docs, top_k=3)
    assert len(results) == 3
    assert results[0]["doc_id"] == "d2", f"Expected d2 first, got {results[0]['doc_id']}"
    assert results[1]["doc_id"] == "d1"
    assert results[2]["doc_id"] == "d3"
    # 验证score写入metadata
    assert results[0]["metadata"]["rerank_score"] == 0.95
    print(f"  Reranked by mock scores: d2(0.95) > d1(0.5) > d3(0.3)")
    print("  PASS\n")


def test_cross_encoder_pairs_format():
    """CrossEncoderReranker: 正确构造(query, doc)对"""
    print("=== 测试5: CrossEncoder pairs格式 ===")
    docs = [
        make_doc("d1", "content A"),
        make_doc("d2", "content B"),
    ]
    mock_model = MockCrossEncoder(mock_scores=[0.5, 0.5])
    reranker = CrossEncoderReranker(model_name="mock", model=mock_model)

    reranker.rerank("test query", docs, top_k=2)

    pairs = mock_model.last_pairs
    assert len(pairs) == 2
    assert pairs[0] == ("test query", "content A")
    assert pairs[1] == ("test query", "content B")
    print(f"  Pairs constructed correctly: ('test query', doc.content)")
    print("  PASS\n")


def test_cross_encoder_unavailable_raises():
    """CrossEncoderReranker: 未安装且未注入model时抛错"""
    print("=== 测试6: CrossEncoder依赖缺失处理 ===")
    if CROSS_ENCODER_AVAILABLE:
        print("  SKIP (sentence-transformers is installed)")
        print()
        return

    try:
        CrossEncoderReranker(model_name="any")
        assert False, "Should raise ImportError"
    except ImportError as e:
        assert "sentence-transformers" in str(e)
    print(f"  Correctly raises ImportError when not installed")
    print("  PASS\n")


# ===== 测试：get_reranker工厂 =====

def test_get_reranker_default():
    """get_reranker: 默认返回KeywordReranker"""
    print("=== 测试7: get_reranker默认 ===")
    reranker = get_reranker(use_cross_encoder=False)
    assert isinstance(reranker, KeywordReranker)
    print(f"  Default returns KeywordReranker")
    print("  PASS\n")


def test_get_reranker_fallback():
    """get_reranker: CrossEncoder不可用时fallback到Keyword"""
    print("=== 测试8: get_reranker fallback ===")
    reranker = get_reranker(use_cross_encoder=True)
    if CROSS_ENCODER_AVAILABLE:
        assert isinstance(reranker, CrossEncoderReranker)
        print(f"  CrossEncoder available, returns CrossEncoderReranker")
    else:
        assert isinstance(reranker, KeywordReranker)
        print(f"  CrossEncoder unavailable, fallback to KeywordReranker")
    print("  PASS\n")


# ===== 测试：two_stage_retrieve =====

def test_two_stage_retrieve():
    """two_stage_retrieve: 两阶段检索完整流程"""
    print("=== 测试9: 两阶段检索流程 ===")
    # 第一阶段召回20个候选
    all_docs = [make_doc(f"d{i}", f"内容{i} 关键词频次{i % 3}") for i in range(20)]
    retriever = MockRetriever(all_docs)

    # 用KeywordReranker精排
    results = two_stage_retrieve(retriever, "关键词", recall_k=20, rerank_k=5)

    assert retriever.last_top_k == 20, "Should recall 20 candidates"
    assert len(results) == 5, "Should rerank to 5"
    # 所有结果应有rerank_score
    for r in results:
        assert "rerank_score" in r["metadata"]
    print(f"  Recalled 20 -> Reranked to {len(results)}")
    print(f"  Top-1 rerank_score: {results[0]['metadata']['rerank_score']:.2f}")
    print("  PASS\n")


def test_two_stage_retrieve_empty():
    """two_stage_retrieve: 召回阶段为空"""
    print("=== 测试10: 两阶段检索空召回 ===")
    retriever = MockRetriever([])
    results = two_stage_retrieve(retriever, "query", recall_k=20, rerank_k=5)
    assert results == []
    print(f"  Empty recall handled correctly")
    print("  PASS\n")


def test_rerank_preserves_doc_fields():
    """精排不破坏原Document的核心字段"""
    print("=== 测试11: 精排保留原字段 ===")
    docs = [make_doc("d1", "content", source="src.md", page=42,
                     metadata={"original": "value"})]
    reranker = KeywordReranker()
    results = reranker.rerank("content", docs, top_k=1)

    assert results[0]["doc_id"] == "d1"
    assert results[0]["source"] == "src.md"
    assert results[0]["page"] == 42
    assert results[0]["metadata"]["original"] == "value"  # 原metadata保留
    assert "rerank_score" in results[0]["metadata"]       # 新增rerank_score
    print(f"  Original fields preserved, rerank_score added")
    print("  PASS\n")


# ===== 主测试入口 =====

if __name__ == "__main__":
    tests = [
        test_keyword_reranker_basic,
        test_keyword_reranker_empty,
        test_keyword_reranker_top_k_limit,
        test_cross_encoder_with_mock,
        test_cross_encoder_pairs_format,
        test_cross_encoder_unavailable_raises,
        test_get_reranker_default,
        test_get_reranker_fallback,
        test_two_stage_retrieve,
        test_two_stage_retrieve_empty,
        test_rerank_preserves_doc_fields,
    ]

    print(f"\nRunning {len(tests)} reranker tests...")
    print(f"CROSS_ENCODER_AVAILABLE: {CROSS_ENCODER_AVAILABLE}\n")
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
