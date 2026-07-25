"""Self-RAG / Corrective 路由与评测 smoke（不依赖向量库）"""
import importlib


def test_route_after_fact_check_disabled_always_conflicts(monkeypatch):
    import src.config as config
    import src.pipeline_routing as pr

    monkeypatch.setattr(config, "ENABLE_SELF_RAG_LOOP", False)
    monkeypatch.setattr(pr, "ENABLE_SELF_RAG_LOOP", False)
    state = {"fact_check_passed": False, "regenerate_count": 0}
    assert pr.route_after_fact_check(state) == "check_conflicts"


def test_route_after_fact_check_enabled_regenerate(monkeypatch):
    import src.config as config
    import src.pipeline_routing as pr

    monkeypatch.setattr(config, "ENABLE_SELF_RAG_LOOP", True)
    monkeypatch.setattr(pr, "ENABLE_SELF_RAG_LOOP", True)
    monkeypatch.setattr(pr, "SELF_RAG_MAX_REGENERATE", 1)

    assert pr.route_after_fact_check(
        {"fact_check_passed": False, "regenerate_count": 0}
    ) == "regenerate"
    assert pr.route_after_fact_check(
        {"fact_check_passed": False, "regenerate_count": 1}
    ) == "check_conflicts"
    assert pr.route_after_fact_check(
        {"fact_check_passed": True, "regenerate_count": 0}
    ) == "check_conflicts"


def test_route_after_grading_basic():
    from src.pipeline_routing import route_after_grading

    assert route_after_grading({"relevant_count": 2, "retry_count": 0}) == "generate"
    assert route_after_grading({"relevant_count": 0, "retry_count": 0}) == "rewrite_query"
    assert route_after_grading({"relevant_count": 0, "retry_count": 1}) == "web_search"


def test_filter_real_web():
    from src.pipeline_routing import filter_real_web_results

    docs = [
        {"source": "https://a.com", "metadata": {"engine": "ddg"}},
        {"source": "mock://x", "metadata": {"is_mock": True}},
    ]
    real = filter_real_web_results(docs)
    assert len(real) == 1
    assert real[0]["source"].startswith("https")


def test_answer_relevancy_chinese_not_zero():
    from src.evaluation.ragas_evaluator import RAGASEvaluator

    e = RAGASEvaluator()
    score = e.evaluate_answer_relevancy(
        "什么是MBTI",
        "MBTI是一种人格类型理论，包含十六种类型。",
        [],
    )
    assert score > 0.0
