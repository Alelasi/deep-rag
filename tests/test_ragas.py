"""RAGAS评测测试 — 验证评测系统"""
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.ragas_evaluator import RAGASEvaluator, evaluate_from_state
from src.graph import query, get_indexer
from src.retrieval.indexer import Indexer

DOCS_DIR = str(Path(PROJECT_ROOT) / "data/sample_docs")


def test_ragas_single_query():
    """测试单个查询的RAGAS评估"""
    print("=== 测试1: 单个查询RAGAS评估 ===")

    evaluator = RAGASEvaluator()

    # 模拟一个查询结果
    question = "INTJ的主导功能是什么？"
    answer = "INTJ的主导功能是Ni（内倾直觉），这是一种聚合思维，负责深度洞察和模式识别。"
    contexts = [
        {
            "doc_id": "1",
            "content": "INTJ的认知功能排列：1. Ni (英雄) - 主导功能，深度洞察。Ni是内倾直觉，聚合思维。",
            "source": "mbti.md",
            "page": 1,
            "grade": "relevant",
            "relevance_score": 0.95
        },
        {
            "doc_id": "2",
            "content": "INTJ属于NT理性者气质，擅长战略思考和系统分析。",
            "source": "mbti.md",
            "page": 2,
            "grade": "relevant",
            "relevance_score": 0.75
        },
        {
            "doc_id": "3",
            "content": "九型人格中5号的核心恐惧是害怕无知和无能。",
            "source": "enneagram.md",
            "page": 1,
            "grade": "irrelevant",
            "relevance_score": 0.1
        }
    ]

    result = evaluator.evaluate_single_query(
        question=question,
        answer=answer,
        contexts=contexts,
        hallucination_score=0.05
    )

    evaluator.print_report(result)

    # 验证指标范围
    assert 0 <= result["answer_relevancy"] <= 1
    assert 0 <= result["context_precision"] <= 1
    assert 0 <= result["context_recall"] <= 1
    assert 0 <= result["faithfulness"] <= 1
    assert 0 <= result["ragas_score"] <= 1

    # 验证合理性
    assert result["context_precision"] > 0.5, "应该有2/3的文档相关"
    assert result["faithfulness"] > 0.9, "幻觉评分0.05，忠实度应该>0.9"

    print("  ✅ PASS\n")
    return result


def test_ragas_batch_evaluation():
    """测试批量评估"""
    print("=== 测试2: 批量RAGAS评估 ===")

    evaluator = RAGASEvaluator()

    # 准备测试用例
    test_cases = [
        {
            "question": "INTJ的主导功能是什么？",
            "answer": "INTJ的主导功能是Ni（内倾直觉）。",
            "contexts": [
                {
                    "content": "INTJ的认知功能排列：1. Ni (英雄) - 主导功能",
                    "grade": "relevant"
                }
            ],
            "hallucination_score": 0.0
        },
        {
            "question": "什么是恐贪指数？",
            "answer": "恐贪指数是基于RSI(14)的市场情绪指标，用于判断市场的恐慌和贪婪程度。",
            "contexts": [
                {
                    "content": "恐贪指数基于RSI(14)作为基础，通过技术指标量化市场情绪。",
                    "grade": "relevant"
                },
                {
                    "content": "RSI低于30表示恐慌，高于70表示贪婪。",
                    "grade": "relevant"
                }
            ],
            "hallucination_score": 0.1
        },
        {
            "question": "九型5号的核心恐惧是什么？",
            "answer": "九型5号的核心恐惧是害怕无知和无能，担心自己没有足够的知识和能力。",
            "contexts": [
                {
                    "content": "九型人格中5号的核心恐惧是害怕无知和无能。",
                    "grade": "relevant"
                }
            ],
            "hallucination_score": 0.05
        }
    ]

    result = evaluator.evaluate_batch(test_cases)
    evaluator.print_report(result)

    # 验证批量结果
    assert result["total_cases"] == 3
    assert "average_metrics" in result
    assert "individual_results" in result

    avg = result["average_metrics"]
    assert 0 <= avg["ragas_score"] <= 1

    print(f"  平均RAGAS得分: {avg['ragas_score']:.3f}")
    print("  ✅ PASS\n")

    return result


def test_ragas_with_real_pipeline():
    """测试与真实Pipeline集成"""
    print("=== 测试3: 与真实Pipeline集成 ===")

    # 索引文档
    indexer = Indexer("test_kb")
    indexer.clear()
    count = indexer.index_directory(DOCS_DIR)
    print(f"  已索引 {count} 个文档块")

    # 运行查询
    question = "INTJ的主导功能是什么"
    state = query(question, "test_kb")

    # 使用RAGAS评估
    result = evaluate_from_state(state)

    print(f"\n  问题: {question}")
    print(f"  答案: {state['answer'][:100]}...")
    print(f"\n  RAGAS指标:")
    print(f"    • Answer Relevancy:   {result['answer_relevancy']:.3f}")
    print(f"    • Context Precision:  {result['context_precision']:.3f}")
    print(f"    • Context Recall:     {result['context_recall']:.3f}")
    print(f"    • Faithfulness:       {result['faithfulness']:.3f}")
    print(f"    • RAGAS Score:        {result['ragas_score']:.3f}")

    # 验证结果合理性
    assert result["ragas_score"] > 0.5, "真实查询的RAGAS得分应该>0.5"

    print("\n  ✅ PASS\n")
    return result


def test_ragas_edge_cases():
    """测试边界情况"""
    print("=== 测试4: 边界情况 ===")

    evaluator = RAGASEvaluator()

    # Case 1: 空答案
    result1 = evaluator.evaluate_single_query(
        question="测试问题",
        answer="",
        contexts=[{"content": "测试内容", "grade": "relevant"}]
    )
    assert result1["answer_relevancy"] == 0.0
    print("  Case 1 (空答案): ✅")

    # Case 2: 无上下文
    result2 = evaluator.evaluate_single_query(
        question="测试问题",
        answer="测试答案",
        contexts=[]
    )
    assert result2["context_precision"] == 0.0
    assert result2["faithfulness"] == 0.0
    print("  Case 2 (无上下文): ✅")

    # Case 3: 拒答
    result3 = evaluator.evaluate_single_query(
        question="测试问题",
        answer="抱歉，我不知道这个问题的答案。",
        contexts=[{"content": "测试内容", "grade": "relevant"}]
    )
    assert result3["answer_relevancy"] < 0.3
    print("  Case 3 (拒答): ✅")

    # Case 4: 高幻觉
    result4 = evaluator.evaluate_single_query(
        question="测试问题",
        answer="这是一个完全编造的答案。",
        contexts=[{"content": "完全不相关的内容", "grade": "irrelevant"}],
        hallucination_score=0.9
    )
    assert result4["faithfulness"] < 0.2
    print("  Case 4 (高幻觉): ✅")

    print("\n  ✅ PASS\n")


def run_all_tests():
    """运行所有RAGAS评测测试"""
    print("\n" + "="*60)
    print("🧪 RAGAS评测系统测试")
    print("="*60 + "\n")

    try:
        test_ragas_single_query()
        test_ragas_batch_evaluation()
        test_ragas_with_real_pipeline()
        test_ragas_edge_cases()

        print("="*60)
        print("✅ 所有RAGAS评测测试通过！")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
