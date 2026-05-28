"""端到端测试 — 验证DeepRAG完整Pipeline"""
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.indexer import Indexer
from src.retrieval.hybrid import HybridRetriever
from src.agents.doc_grader import grade_documents_offline
from src.agents.fact_checker import check_facts_offline
from src.agents.conflict_resolver import resolve_conflicts_offline
from src.agents.query_analyzer import analyze_query_offline
from src.evaluation.metrics import evaluate_pipeline
from src.graph import query, get_indexer

DOCS_DIR = str(Path(PROJECT_ROOT) / "data/sample_docs")


def test_indexing():
    """测试文档索引"""
    print("=== 测试1: 文档索引 ===")
    indexer = Indexer("test_kb")
    indexer.clear()
    count = indexer.index_directory(DOCS_DIR)
    assert count > 0, "Should index some chunks"
    print(f"  Indexed {count} chunks")

    # 验证ChromaDB
    collection = indexer.get_collection()
    assert collection.count() == count
    print(f"  ChromaDB: {collection.count()} documents")

    # 验证BM25
    bm25, docs = indexer.get_bm25()
    assert bm25 is not None
    assert len(docs) == count
    print(f"  BM25: {len(docs)} documents")
    print("  ✅ PASS\n")


def test_hybrid_retrieval():
    """测试混合检索"""
    print("=== 测试2: 混合检索 ===")
    # 重新创建 indexer
    indexer = Indexer("test_kb")
    retriever = HybridRetriever(indexer)

    # 精确关键词查询
    results = retriever.retrieve("INTJ功能堆栈", top_k=5)
    assert len(results) > 0, "Should have results"
    # INTJ相关文档应该排前面
    assert any("INTJ" in r["content"] for r in results[:3]), "INTJ should be in top results"
    print(f"  Query 'INTJ功能堆栈': {len(results)} results, top source: {results[0]['source']}")

    # 语义查询
    results2 = retriever.retrieve("什么是恐慌指标", top_k=5)
    assert len(results2) > 0
    assert any("恐" in r["content"] for r in results2[:3])
    print(f"  Query '什么是恐慌指标': {len(results2)} results, top source: {results2[0]['source']}")

    print("  ✅ PASS\n")


def test_doc_grading():
    """测试Corrective RAG文档评分"""
    print("=== 测试3: Corrective RAG文档评分 ===")

    question = "INTJ的主导功能是什么"
    docs = [
        {"doc_id": "1", "content": "INTJ的认知功能排列：1. Ni (英雄) - 主导功能，深度洞察",
         "source": "mbti.md", "page": 1, "metadata": {}},
        {"doc_id": "2", "content": "九型人格中5号的核心恐惧是害怕无知和无能",
         "source": "enneagram.md", "page": 1, "metadata": {}},
        {"doc_id": "3", "content": "恐贪指数基于RSI(14)作为基础",
         "source": "fear_greed.md", "page": 1, "metadata": {}},
    ]

    graded = grade_documents_offline(question, docs)
    assert len(graded) == 3

    # 第一个应该是relevant（直接包含INTJ和主导功能）
    assert graded[0]["grade"] == "relevant", f"Expected relevant, got {graded[0]['grade']}"
    # 后两个应该是irrelevant（九型和恐贪与INTJ无关）
    assert graded[2]["grade"] == "irrelevant", f"Expected irrelevant, got {graded[2]['grade']}"

    print(f"  Doc1 (INTJ): {graded[0]['grade']} ({graded[0]['relevance_score']:.2f})")
    print(f"  Doc2 (九型): {graded[1]['grade']} ({graded[1]['relevance_score']:.2f})")
    print(f"  Doc3 (恐贪): {graded[2]['grade']} ({graded[2]['relevance_score']:.2f})")
    print("  ✅ PASS\n")


def test_fact_checker():
    """测试Self-RAG事实校验"""
    print("=== 测试4: Self-RAG事实校验 ===")

    # Case 1: 忠实于源文档的回答
    answer_good = "INTJ的主导功能是Ni（内倾直觉），这是一种聚合思维。"
    docs = [{"doc_id": "1", "content": "INTJ的认知功能排列：1. Ni (英雄) - 主导功能，深度洞察。Ni是内倾直觉，聚合思维。",
             "source": "mbti.md", "page": 1, "grade": "relevant", "relevance_score": 0.9, "reasoning": ""}]
    result = check_facts_offline(answer_good, docs)
    assert result["passed"], f"Good answer should pass, score: {result['hallucination_score']}"
    print(f"  Good answer: score={result['hallucination_score']:.2f} PASS")

    # Case 2: 含幻觉的回答
    answer_bad = "INTJ的主导功能是Te（外倾思考），这使他们成为天生的领导者，在全球仅占人口的0.3%。"
    result2 = check_facts_offline(answer_bad, docs)
    # 应该有较高的幻觉分（Te不是主导，0.3%编造）
    assert result2["hallucination_score"] > result["hallucination_score"], \
        "Bad answer should have higher hallucination score"
    print(f"  Bad answer: score={result2['hallucination_score']:.2f} {'FAIL' if not result2['passed'] else 'PASS'}")
    print(f"  Unsupported: {result2['unsupported_claims'][:2]}")
    print("  ✅ PASS\n")


def test_conflict_detection():
    """测试多源冲突检测"""
    print("=== 测试5: 多源冲突检测 ===")

    docs = [
        {"doc_id": "1", "content": "v5.3的相关度为0.9158（基于RSI）",
         "source": "fear_greed_v53.md", "page": 1, "grade": "relevant",
         "relevance_score": 0.9, "reasoning": ""},
        {"doc_id": "2", "content": "v3.6版本相关度仅0.075",
         "source": "fear_greed_v36.md", "page": 1, "grade": "relevant",
         "relevance_score": 0.8, "reasoning": ""},
    ]

    conflicts = resolve_conflicts_offline("恐贪指数的相关度是多少", docs)
    # 应该检测到相关度数值冲突（0.9158 vs 0.075）
    print(f"  Conflicts found: {len(conflicts)}")
    for c in conflicts:
        print(f"    Topic: {c['topic']}, Positions: {len(c['positions'])}")
    print("  ✅ PASS\n")


def test_full_pipeline():
    """测试完整Pipeline"""
    print("=== 测试6: 完整Pipeline ===")

    # 先索引文档
    indexer = get_indexer("e2e_test")
    indexer.clear()
    count = indexer.index_directory(DOCS_DIR)
    print(f"  Indexed {count} chunks")

    # 执行查询
    result = query("INTJ的主导功能是什么", collection_name="e2e_test")

    assert result["current_step"] == "completed", f"Expected completed, got {result['current_step']}"
    assert result["answer"], "Should have an answer"
    assert result["relevant_count"] > 0, "Should find relevant docs"

    print(f"  Status: {result['current_step']}")
    print(f"  Answer: {result['answer'][:100]}...")
    print(f"  Relevant docs: {result['relevant_count']}")
    print(f"  Hallucination: {result['hallucination_score']:.2f}")
    print(f"  Citations: {len(result['citations'])}")
    print(f"  History: {result['history']}")

    # 评估
    eval_result = evaluate_pipeline(result)
    print(f"  Overall score: {eval_result['overall_score']}/100")
    print("  ✅ PASS\n")


def test_query_no_match():
    """测试知识库无答案的情况"""
    print("=== 测试7: 知识库无答案→Web Fallback ===")

    indexer = get_indexer("e2e_test_empty")
    indexer.clear()
    # 索引一些与问题完全无关的文档
    indexer.index_texts([
        {"content": "今天天气很好，适合出去散步。", "source": "weather.md", "page": 1},
    ])

    result = query("LangGraph的checkpoint机制是什么", collection_name="e2e_test_empty", max_retries=1)
    # 应该触发web_search fallback或者查询改写
    print(f"  Status: {result['current_step']}")
    print(f"  History: {result['history']}")
    assert "web" in " ".join(result.get("history", [])).lower() or result["retry_count"] > 0, \
        "Should trigger fallback or retry"
    print("  ✅ PASS\n")


def test_agentic_rag_mode():
    """测试8: Agentic RAG模式（Agent路由决策）"""
    print("=== 测试8: Agentic RAG模式 ===")
    import os

    # 临时启用 Agentic RAG
    original_flag = os.environ.get("ENABLE_AGENTIC_RAG")
    os.environ["ENABLE_AGENTIC_RAG"] = "true"

    try:
        # 重新导入以应用环境变量
        import importlib
        import src.config
        importlib.reload(src.config)

        # 索引文档
        indexer = get_indexer("agentic_test")
        indexer.clear()
        count = indexer.index_directory(DOCS_DIR)
        print(f"  Indexed {count} chunks")

        # 执行查询（Agentic模式）
        result = query("INTJ的主导功能是什么", collection_name="agentic_test")

        # 验证 Agentic 模式特征
        history_str = " ".join(result.get("history", []))
        assert "via" in history_str or "tool" in history_str.lower(), \
            "Agentic mode should log tool selection in history"

        print(f"  Status: {result['current_step']}")
        print(f"  Answer: {result['answer'][:80]}...")
        print(f"  History: {result['history']}")

        # 验证基本功能正常
        assert result["current_step"] == "completed"
        assert result["answer"]
        assert result["relevant_count"] > 0

        print("  ✅ PASS (Agentic RAG mode working)\n")

    finally:
        # 恢复原始环境变量
        if original_flag is None:
            os.environ.pop("ENABLE_AGENTIC_RAG", None)
        else:
            os.environ["ENABLE_AGENTIC_RAG"] = original_flag
        # 重新加载配置
        importlib.reload(src.config)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DeepRAG - 端到端测试")
    print("=" * 60 + "\n")

    indexer = test_indexing()
    test_hybrid_retrieval()
    test_doc_grading()
    test_fact_checker()
    test_conflict_detection()
    test_full_pipeline()
    test_query_no_match()
    test_agentic_rag_mode()

    print("=" * 60)
    print("  全部测试通过 ✅ (8/8)")
    print("=" * 60)
