"""L1 单元测试 — 评估指标与文本处理（无外部依赖）

测试 src/evaluation/metrics.py 的纯逻辑函数：
- 检索质量评估（precision, recall）
- 生成质量评估（faithfulness, citation_density, completeness）
- Pipeline 综合评估
"""
import pytest


@pytest.mark.L1
class TestEvaluateRetrieval:
    """检索质量评估测试"""

    def test_empty_docs_returns_zeros(self):
        """空文档列表返回零值"""
        from src.evaluation.metrics import evaluate_retrieval
        result = evaluate_retrieval([])
        assert result["precision"] == 0
        assert result["total"] == 0
        assert result["recall"] is None

    def test_all_relevant_docs(self):
        """全部相关文档 precision=1.0"""
        from src.evaluation.metrics import evaluate_retrieval
        docs = [
            {"grade": "relevant"},
            {"grade": "relevant"},
            {"grade": "relevant"},
        ]
        result = evaluate_retrieval(docs)
        assert result["precision"] == 1.0
        assert result["relevant_count"] == 3
        assert result["total_retrieved"] == 3

    def test_mixed_relevance(self):
        """混合相关度文档"""
        from src.evaluation.metrics import evaluate_retrieval
        docs = [
            {"grade": "relevant"},
            {"grade": "irrelevant"},
            {"grade": "relevant"},
            {"grade": "ambiguous"},
        ]
        result = evaluate_retrieval(docs)
        assert result["precision"] == 0.5
        assert result["relevant_count"] == 2
        assert result["total_retrieved"] == 4

    def test_recall_calculation(self):
        """召回率计算"""
        from src.evaluation.metrics import evaluate_retrieval
        docs = [
            {"grade": "relevant"},
            {"grade": "relevant"},
            {"grade": "irrelevant"},
        ]
        result = evaluate_retrieval(docs, total_relevant_in_kb=10)
        assert result["recall"] == 0.2  # 2/10

    def test_recall_none_when_no_total(self):
        """未提供总数时 recall 为 None"""
        from src.evaluation.metrics import evaluate_retrieval
        docs = [{"grade": "relevant"}]
        result = evaluate_retrieval(docs)
        assert result["recall"] is None

    def test_precision_rounding(self):
        """precision 保留3位小数"""
        from src.evaluation.metrics import evaluate_retrieval
        docs = [
            {"grade": "relevant"},
            {"grade": "irrelevant"},
            {"grade": "irrelevant"},
        ]
        result = evaluate_retrieval(docs)
        assert result["precision"] == round(1/3, 3)


@pytest.mark.L1
class TestEvaluateGeneration:
    """生成质量评估测试"""

    def test_faithfulness_calculation(self):
        """忠实度 = 1 - 幻觉评分"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.3,
            citations=[{"text": "test", "source": "doc", "page": 1}],
            answer="这是一个测试答案" * 50,
            conflicts=[],
        )
        assert result["faithfulness"] == 0.7

    def test_zero_hallucination_max_faithfulness(self):
        """零幻觉时忠实度为1.0"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.0,
            citations=[],
            answer="答案",
            conflicts=[],
        )
        assert result["faithfulness"] == 1.0

    def test_completeness_long_answer(self):
        """长答案完成度为1.0"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.0,
            citations=[],
            answer="x" * 600,
            conflicts=[],
        )
        assert result["completeness"] == 1.0

    def test_completeness_short_answer(self):
        """短答案完成度低"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.0,
            citations=[],
            answer="短",
            conflicts=[],
        )
        assert result["completeness"] == 0.1

    def test_completeness_medium_answer(self):
        """中等长度答案完成度0.4"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.0,
            citations=[],
            answer="x" * 100,
            conflicts=[],
        )
        assert result["completeness"] == 0.4

    def test_has_conflicts_true(self):
        """有冲突时 has_conflicts=True"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.0,
            citations=[],
            answer="答案",
            conflicts=[{"topic": "test"}],
        )
        assert result["has_conflicts"] is True

    def test_has_conflicts_false(self):
        """无冲突时 has_conflicts=False"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.0,
            citations=[],
            answer="答案",
            conflicts=[],
        )
        assert result["has_conflicts"] is False

    def test_empty_answer(self):
        """空答案不报错"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.5,
            citations=[],
            answer="",
            conflicts=[],
        )
        assert result["answer_length"] == 0
        assert result["faithfulness"] == 0.5

    def test_citation_density_capped_at_1(self):
        """引用密度上限为1.0"""
        from src.evaluation.metrics import evaluate_generation
        result = evaluate_generation(
            hallucination_score=0.0,
            citations=[{"text": "t"} for _ in range(20)],
            answer="短答案",
            conflicts=[],
        )
        assert result["citation_density"] <= 1.0


@pytest.mark.L1
class TestEvaluatePipeline:
    """Pipeline 综合评估测试"""

    def test_pipeline_with_valid_state(self):
        """有效状态的 Pipeline 评估"""
        from src.evaluation.metrics import evaluate_pipeline
        state = {
            "graded_docs": [
                {"grade": "relevant"},
                {"grade": "irrelevant"},
            ],
            "hallucination_score": 0.1,
            "citations": [{"text": "cite1"}],
            "answer": "这是一个参考答案" * 30,
            "conflicts": [],
        }
        result = evaluate_pipeline(state)
        assert "retrieval" in result
        assert "generation" in result

    def test_pipeline_empty_state(self):
        """空状态的 Pipeline 评估不报错"""
        from src.evaluation.metrics import evaluate_pipeline
        state = {
            "graded_docs": [],
            "hallucination_score": 0.0,
            "citations": [],
            "answer": "",
            "conflicts": [],
        }
        result = evaluate_pipeline(state)
        assert result["retrieval"]["total"] == 0
