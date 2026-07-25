"""22指标综合评估引擎 — 整合所有评估模块，加权评分"""
import time
from typing import Dict, List, Optional


# 22 指标权重定义（百分比，总计100%）
METRIC_WEIGHTS = {
    # 生成质量 (42%)
    "faithfulness": 17.0,
    "answer_relevancy": 13.0,
    "completeness": 7.0,
    "citation_density": 5.0,
    # 检索质量 (34%)
    "precision": 8.0,
    "recall": 8.0,
    "context_precision": 8.0,
    "context_relevancy": 7.0,
    "context_recall": 3.0,
    # 效率 (5%)
    "response_time": 5.0,
    # 可靠性 (7%)
    "conflict_resolution": 5.0,
    "fact_check_passed": 2.0,
    # Multi-Agent (1.5%)
    "collaboration_efficiency": 0.5,
    "communication_cost": 0.5,
    "decision_accuracy": 0.5,
    # 意图 (0.5%)
    "intent_accuracy": 0.5,
    # 详细评分 (10%)
    "accuracy_score": 2.0,
    "relevance_score": 2.0,
    "conciseness_score": 2.0,
    "faithfulness_score": 2.0,
    "completeness_score": 2.0,
}

# 分组定义（用于雷达图）
METRIC_GROUPS = {
    "生成质量": ["faithfulness", "answer_relevancy", "completeness", "citation_density"],
    "检索质量": ["precision", "recall", "context_precision", "context_relevancy", "context_recall"],
    "效率": ["response_time"],
    "可靠性": ["conflict_resolution", "fact_check_passed"],
    "详细评分": ["accuracy_score", "relevance_score", "conciseness_score", "faithfulness_score", "completeness_score"],
}


class EvaluationEngine:
    """综合评估引擎"""

    def evaluate(self, state: dict, response_time: float = 0.0,
                 question: str = "", answer: str = "",
                 contexts: List[str] = None,
                 agent_trace: List = None) -> dict:
        """运行所有评估指标

        Args:
            state: RAG 状态字典
            response_time: 响应时间（秒）
            question: 用户问题
            answer: 生成的回答
            contexts: 检索到的上下文列表
            agent_trace: Agent 决策轨迹

        Returns:
            {metrics: {name: score}, overall_score: float, group_scores: {group: score}}
        """
        metrics = {}
        contexts = contexts or state.get("contexts", [])
        answer = answer or state.get("answer", "")
        question = question or state.get("question", "")

        # === 1. 基础指标（来自 metrics.py）===
        try:
            from src.evaluation.metrics import evaluate_retrieval, evaluate_generation
            retrieval = evaluate_retrieval(
                state.get("graded_docs", []),
                state.get("total_relevant_in_kb"),
            )
            generation = evaluate_generation(
                state.get("hallucination_score", 0),
                state.get("citations", []),
                answer,
                state.get("conflicts", []),
            )
            metrics["precision"] = retrieval.get("precision", 0)
            metrics["recall"] = retrieval.get("recall") or 0
            metrics["faithfulness"] = generation.get("faithfulness", 0)
            metrics["citation_density"] = generation.get("citation_density", 0)
            metrics["completeness"] = generation.get("completeness", 0)
            metrics["conflict_resolution"] = 0.0 if generation.get("has_conflicts") else 1.0
        except Exception:
            pass

        # === 2. RAGAS 指标（来自 ragas_evaluator.py）===
        try:
            from src.evaluation.ragas_evaluator import RAGASEvaluator
            evaluator = RAGASEvaluator()
            ragas_result = evaluator.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
            )
            metrics["answer_relevancy"] = ragas_result.get("answer_relevancy", 0)
            metrics["context_precision"] = ragas_result.get("context_precision", 0)
            metrics["context_relevancy"] = ragas_result.get("context_relevancy", 0)
            metrics["context_recall"] = ragas_result.get("context_recall", 0)
        except Exception:
            pass

        # === 3. 响应时间评分（< 2s = 1.0, > 10s = 0.0）===
        if response_time > 0:
            if response_time <= 2:
                metrics["response_time"] = 1.0
            elif response_time >= 10:
                metrics["response_time"] = 0.0
            else:
                metrics["response_time"] = round(1.0 - (response_time - 2) / 8, 3)

        # === 4. 事实检查 ===
        fact_checker_result = state.get("fact_check_result", {})
        metrics["fact_check_passed"] = 1.0 if fact_checker_result.get("passed", True) else 0.0

        # === 5. Multi-Agent 指标 ===
        try:
            from src.evaluation.multi_agent_metrics import MultiAgentMetrics
            ma = MultiAgentMetrics()
            ma_result = ma.evaluate(state.get("history", []))
            metrics["collaboration_efficiency"] = ma_result.get("collaboration_efficiency", 0)
            metrics["communication_cost"] = ma_result.get("communication_cost", 0)
            metrics["decision_accuracy"] = ma_result.get("decision_accuracy", 0)
        except Exception:
            pass

        # === 6. 意图识别 ===
        try:
            from src.evaluation.intent_evaluator import IntentEvaluator
            ie = IntentEvaluator()
            intent_result = ie.evaluate(question, state.get("intent", ""))
            metrics["intent_accuracy"] = intent_result.get("accuracy", 0)
        except Exception:
            pass

        # === 7. 详细评分（来自 rag_evaluation_v3.py）===
        try:
            import sys
            from pathlib import Path
            eval_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "evaluation" / "rag_evaluation_v3.py"
            if eval_script.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("rag_evaluation_v3", eval_script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                detail = mod.evaluate_detailed(question, answer, contexts)
                metrics["accuracy_score"] = detail.get("accuracy", 0) / 100
                metrics["relevance_score"] = detail.get("relevance", 0) / 100
                metrics["conciseness_score"] = detail.get("conciseness", 0) / 100
                metrics["faithfulness_score"] = detail.get("faithfulness", 0) / 100
                metrics["completeness_score"] = detail.get("completeness", 0) / 100
        except Exception:
            pass

        # === 计算加权综合评分 ===
        overall = self._weighted_score(metrics)
        group_scores = self._group_scores(metrics)

        return {
            "metrics": metrics,
            "overall_score": overall,
            "group_scores": group_scores,
            "metric_count": len(metrics),
            "total_metrics": len(METRIC_WEIGHTS),
        }

    def _weighted_score(self, metrics: dict) -> float:
        """按权重计算综合得分 (0-100)"""
        total_weight = 0
        weighted_sum = 0
        for name, score in metrics.items():
            weight = METRIC_WEIGHTS.get(name, 0)
            if weight > 0:
                weighted_sum += score * weight
                total_weight += weight
        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight * 100, 1)

    def _group_scores(self, metrics: dict) -> dict:
        """计算分组得分"""
        result = {}
        for group, metric_names in METRIC_GROUPS.items():
            scores = [metrics.get(name, 0) for name in metric_names if name in metrics]
            if scores:
                result[group] = round(sum(scores) / len(scores) * 100, 1)
            else:
                result[group] = 0.0
        return result

    def get_metric_info(self) -> dict:
        """获取指标权重和分组信息（供前端展示）"""
        return {
            "weights": METRIC_WEIGHTS,
            "groups": METRIC_GROUPS,
        }
