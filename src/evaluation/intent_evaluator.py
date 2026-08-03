"""
意图识别评测模块
参考马丁第193-202行：意图识别与路由决策评测

评测指标：
1. Top-1 准确率（一级意图）
2. Top-1 准确率（二级意图）
3. 路由决策准确率
4. 混淆矩阵
"""

import json
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.intent.intent_classifier_v2 import IntentClassifier, IntentL1, IntentL2


try:
    from src.logging_config import get_logger
except Exception:
    import logging

    def get_logger(n):  # type: ignore
        return logging.getLogger(n)

logger = get_logger(__name__)

class IntentEvaluator:
    """意图识别评测器"""

    def __init__(self, classifier: IntentClassifier):
        self.classifier = classifier

    def evaluate(self, eval_dataset: List[Dict]) -> Dict:
        """
        评测意图识别准确率

        参数：
            eval_dataset: 评估集，每条数据包含：
                - query: 用户问题
                - intent_l1: 标注的一级意图
                - intent_l2: 标注的二级意图
                - need_tool_call: 是否需要工具调用
                - should_refuse: 是否应该拒答

        返回：
            评测结果字典
        """
        results = {
            "total": len(eval_dataset),
            "l1_correct": 0,
            "l2_correct": 0,
            "route_correct": 0,
            "confusion_matrix_l1": defaultdict(lambda: defaultdict(int)),
            "confusion_matrix_l2": defaultdict(lambda: defaultdict(int)),
            "route_confusion": defaultdict(lambda: defaultdict(int)),
            "low_confidence_cases": [],
            "error_cases": []
        }

        for item in eval_dataset:
            query = item["query"]
            true_l1 = item["intent_l1"]
            true_l2 = item["intent_l2"]
            need_tool = item.get("need_tool_call", False)
            should_refuse = item.get("should_refuse", False)

            # 预测
            pred = self.classifier.classify(query)

            # 一级意图准确率
            pred_l1 = pred.intent_l1.value
            if pred_l1 == true_l1:
                results["l1_correct"] += 1
            results["confusion_matrix_l1"][true_l1][pred_l1] += 1

            # 二级意图准确率
            pred_l2 = pred.intent_l2.value
            if pred_l2 == true_l2:
                results["l2_correct"] += 1
            results["confusion_matrix_l2"][true_l2][pred_l2] += 1

            # 路由决策准确率
            expected_route = self._get_expected_route(need_tool, should_refuse)
            if pred.route_decision == expected_route:
                results["route_correct"] += 1
            results["route_confusion"][expected_route][pred.route_decision] += 1

            # 记录低置信度案例
            if pred.confidence < 0.7:
                results["low_confidence_cases"].append({
                    "query": query,
                    "confidence": pred.confidence,
                    "pred_l1": pred_l1,
                    "true_l1": true_l1
                })

            # 记录错误案例
            if pred_l1 != true_l1 or pred_l2 != true_l2:
                results["error_cases"].append({
                    "query": query,
                    "true_l1": true_l1,
                    "pred_l1": pred_l1,
                    "true_l2": true_l2,
                    "pred_l2": pred_l2,
                    "confidence": pred.confidence,
                    "reason": pred.reason
                })

        # 计算准确率
        results["l1_accuracy"] = results["l1_correct"] / results["total"]
        results["l2_accuracy"] = results["l2_correct"] / results["total"]
        results["route_accuracy"] = results["route_correct"] / results["total"]

        return results

    def _get_expected_route(self, need_tool: bool, should_refuse: bool) -> str:
        """根据标注获取期望的路由决策"""
        if should_refuse:
            return "refuse"
        if need_tool:
            return "kb_and_tool"  # 简化：需要工具就是混合
        return "kb_only"

    def print_report(self, results: Dict):
        """打印评测报告"""
        logger.info("=" * 80)
        logger.info("意图识别评测报告")
        logger.info("=" * 80)

        logger.info(f"\n📊 总体指标")
        logger.info(f"  - 总数: {results['total']}")
        logger.info(f"  - 一级意图准确率: {results['l1_accuracy']:.2%} ({results['l1_correct']}/{results['total']})")
        logger.info(f"  - 二级意图准确率: {results['l2_accuracy']:.2%} ({results['l2_correct']}/{results['total']})")
        logger.info(f"  - 路由决策准确率: {results['route_accuracy']:.2%} ({results['route_correct']}/{results['total']})")

        # 参考目标（马丁第197行）
        target = 0.90
        logger.info(f"\n🎯 参考目标: ≥{target:.0%}")
        if results['l1_accuracy'] >= target:
            logger.info(f"  ✅ 一级意图准确率达标")
        else:
            logger.error(f"  ❌ 一级意图准确率未达标（差距: {target - results['l1_accuracy']:.2%}）")

        # 低置信度案例
        logger.warning(f"\n⚠️  低置信度案例（<0.7）: {len(results['low_confidence_cases'])} 个")
        if results['low_confidence_cases']:
            logger.info("  前5个案例:")
            for i, case in enumerate(results['low_confidence_cases'][:5]):
                logger.info(f"    {i+1}. {case['query'][:50]}... (置信度: {case['confidence']:.2f})")

        # 错误案例
        logger.error(f"\n❌ 错误案例: {len(results['error_cases'])} 个")
        if results['error_cases']:
            logger.info("  前5个案例:")
            for i, case in enumerate(results['error_cases'][:5]):
                logger.info(f"    {i+1}. {case['query'][:50]}...")
                logger.info(f"       预测: {case['pred_l1']} / {case['pred_l2']}")
                logger.info(f"       真值: {case['true_l1']} / {case['true_l2']}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 加载评估集
    eval_file = "data/evaluation/evaluation_dataset_150_v3_real.json"
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_dataset = json.load(f)

    # 创建分类器和评测器
    classifier = IntentClassifier()
    evaluator = IntentEvaluator(classifier)

    # 评测
    logger.info("开始评测意图识别模块...")
    results = evaluator.evaluate(eval_dataset)

    # 打印报告
    evaluator.print_report(results)

    # 保存结果
    output_file = "data/evaluation/intent_eval_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换 defaultdict 为普通 dict
        results_serializable = {
            "total": results["total"],
            "l1_correct": results["l1_correct"],
            "l2_correct": results["l2_correct"],
            "route_correct": results["route_correct"],
            "l1_accuracy": results["l1_accuracy"],
            "l2_accuracy": results["l2_accuracy"],
            "route_accuracy": results["route_accuracy"],
            "confusion_matrix_l1": dict(results["confusion_matrix_l1"]),
            "confusion_matrix_l2": dict(results["confusion_matrix_l2"]),
            "route_confusion": dict(results["route_confusion"]),
            "low_confidence_cases": results["low_confidence_cases"][:10],
            "error_cases": results["error_cases"][:20]
        }
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ 评测结果已保存到: {output_file}")
