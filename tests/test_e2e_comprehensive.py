"""L3 端到端测试 — 通过 pytest 运行综合测评

此文件是 scripts/comprehensive_test.py 的 pytest 包装器，
使 run_pyramid_tests.py 能通过 -m L3 标记过滤运行。

需要外部服务：Qdrant (6333) + Ollama (11434)
"""
import socket
import pytest
import sys
from pathlib import Path

# 添加 scripts 目录到 path
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))


def _check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """检查端口是否可连接"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# 外部服务检查 fixture
@pytest.fixture(scope="module")
def external_services():
    """检查外部服务是否可用"""
    qdrant_ok = _check_port("localhost", 6333)
    ollama_ok = _check_port("localhost", 11434)
    if not (qdrant_ok and ollama_ok):
        pytest.skip("需要 Qdrant (6333) 和 Ollama (11434) 服务运行中")
    return True


@pytest.mark.L3
class TestComprehensiveEvaluation:
    """综合测评 — L3 端到端测试"""

    def test_import_comprehensive(self, external_services):
        """能正确导入 comprehensive_test 模块"""
        import comprehensive_test
        assert hasattr(comprehensive_test, "TEST_QUESTIONS")
        assert len(comprehensive_test.TEST_QUESTIONS) == 100

    def test_eval_score_calculation(self, external_services):
        """EvalScore 综合得分计算正确"""
        from comprehensive_test import EvalScore
        score = EvalScore(
            question_id=1,
            category="RAG",
            difficulty="easy",
            question="测试问题",
            accuracy=8.0,
            completeness=7.0,
            relevance=9.0,
            citation_quality=6.0,
            response_time=3.0,
            hallucination=2.0,
            format_score=8.0,
            fluency=7.0,
        )
        total = score.calc_total()
        assert 0 < total <= 10
        assert isinstance(total, float)

    def test_evaluate_answer_function(self, external_services):
        """evaluate_answer 评分函数正常工作"""
        from comprehensive_test import evaluate_answer
        question_data = {
            "id": 1,
            "category": "RAG",
            "difficulty": "easy",
            "question": "什么是RAG？",
            "expected_keywords": ["检索增强生成", "Retrieval"],
        }
        answer = "RAG是检索增强生成（Retrieval Augmented Generation）技术，通过检索知识库文档来增强LLM生成质量。[来源1]"
        score = evaluate_answer(question_data, answer, 2.5)
        assert score.accuracy > 0
        assert score.completeness > 0
        assert score.total_score > 0

    def test_real_rag_single_query(self, external_services):
        """真实 RAG 管道单题测试"""
        from comprehensive_test import real_test, TEST_QUESTIONS
        # 选取一道简单题
        question = next(q for q in TEST_QUESTIONS if q["id"] == 21)  # "什么是RAG？"
        score = real_test(question)
        assert score is not None
        assert score.question_id == 21
        # 真实调用可能有错误，但不能是 None
        assert hasattr(score, "total_score")

    def test_real_rag_multiple_categories(self, external_services):
        """真实 RAG 管道多类别测试（每类别1题）"""
        from comprehensive_test import real_test, TEST_QUESTIONS
        # 每类别选1题
        tested_categories = set()
        for q in TEST_QUESTIONS:
            if q["category"] not in tested_categories:
                tested_categories.add(q["category"])
                score = real_test(q)
                assert score is not None
                assert hasattr(score, "total_score")
            if len(tested_categories) >= 3:  # 测试3个类别即可
                break

    def test_generate_report(self, external_services):
        """报告生成函数正常工作"""
        from comprehensive_test import EvalScore, generate_report
        scores = [
            EvalScore(
                question_id=1, category="RAG", difficulty="easy",
                question="测试1", accuracy=8.0, completeness=7.0,
                relevance=9.0, citation_quality=6.0, response_time=3.0,
                hallucination=2.0, format_score=8.0, fluency=7.0,
            ),
            EvalScore(
                question_id=2, category="MBTI", difficulty="medium",
                question="测试2", accuracy=6.0, completeness=5.0,
                relevance=7.0, citation_quality=4.0, response_time=5.0,
                hallucination=3.0, format_score=6.0, fluency=5.0,
            ),
        ]
        for s in scores:
            s.calc_total()
        report = generate_report(scores, 10.0)
        assert isinstance(report, str)
        assert "综合测评报告" in report
        assert "总体统计" in report
