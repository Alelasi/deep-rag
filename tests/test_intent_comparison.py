"""意图识别准确率对比测试

对比三种模式：
1. 纯规则（baseline）
2. 规则 + LLM兜底（当前系统）
3. 纯LLM（理想上限）

目标：验证LLM是否真正提升准确率，从77% → 85%+
"""

import json
import sys
import os
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.evaluation.intent_evaluator import IntentEvaluator
from src.intent.intent_classifier_v2 import IntentClassifier


def load_dataset():
    """加载评估数据集"""
    dataset_path = "data/evaluation/evaluation_dataset_150_v5_corrected.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def test_rule_only():
    """测试纯规则模式"""
    print("\n" + "="*80)
    print("【模式1】纯规则（baseline，不用LLM）")
    print("="*80)

    classifier = IntentClassifier(use_llm=False)  # 关闭LLM
    evaluator = IntentEvaluator(classifier)

    dataset = load_dataset()

    start = time()
    result = evaluator.evaluate(dataset)
    elapsed = time() - start

    l1_acc = result["l1_correct"] / result["total"]
    l2_acc = result["l2_correct"] / result["total"]
    route_acc = result["route_correct"] / result["total"]

    print(f"\n总样本数: {result['total']}")
    print(f"L1准确率: {result['l1_correct']}/{result['total']} = {l1_acc:.2%}")
    print(f"L2准确率: {result['l2_correct']}/{result['total']} = {l2_acc:.2%}")
    print(f"路由准确率: {result['route_correct']}/{result['total']} = {route_acc:.2%}")
    print(f"耗时: {elapsed:.2f}s")
    print(f"平均延迟: {elapsed/result['total']*1000:.1f}ms/query")

    # 错误案例分析（前10个）
    print(f"\n错误案例（前10个）：")
    for i, case in enumerate(result["error_cases"][:10], 1):
        print(f"{i}. \"{case['query']}\"")
        print(f"   真实: {case['true_l1']} → {case['true_l2']}")
        print(f"   预测: {case['pred_l1']} → {case['pred_l2']}")

    return result


def test_rule_plus_llm():
    """测试规则+LLM混合模式"""
    print("\n" + "="*80)
    print("【模式2】规则 + LLM兜底（当前系统，use_llm=True）")
    print("="*80)

    classifier = IntentClassifier(use_llm=True, llm_threshold=0.7)
    evaluator = IntentEvaluator(classifier)

    dataset = load_dataset()

    start = time()
    result = evaluator.evaluate(dataset)
    elapsed = time() - start

    l1_acc = result["l1_correct"] / result["total"]
    l2_acc = result["l2_correct"] / result["total"]
    route_acc = result["route_correct"] / result["total"]

    print(f"\n总样本数: {result['total']}")
    print(f"L1准确率: {result['l1_correct']}/{result['total']} = {l1_acc:.2%}")
    print(f"L2准确率: {result['l2_correct']}/{result['total']} = {l2_acc:.2%}")
    print(f"路由准确率: {result['route_correct']}/{result['total']} = {route_acc:.2%}")
    print(f"耗时: {elapsed:.2f}s")
    print(f"平均延迟: {elapsed/result['total']*1000:.1f}ms/query")

    # LLM调用统计
    low_conf_count = len(result["low_confidence_cases"])
    print(f"\nLLM调用次数: {low_conf_count}/{result['total']} ({low_conf_count/result['total']:.1%})")

    # 错误案例分析（前10个）
    print(f"\n错误案例（前10个）：")
    for i, case in enumerate(result["error_cases"][:10], 1):
        print(f"{i}. \"{case['query']}\"")
        print(f"   真实: {case['true_l1']} → {case['true_l2']}")
        print(f"   预测: {case['pred_l1']} → {case['pred_l2']}")

    return result


def compare_results(rule_result, hybrid_result):
    """对比两种模式的结果"""
    print("\n" + "="*80)
    print("【对比分析】纯规则 vs 规则+LLM")
    print("="*80)

    rule_l1 = rule_result["l1_correct"] / rule_result["total"]
    hybrid_l1 = hybrid_result["l1_correct"] / hybrid_result["total"]

    rule_l2 = rule_result["l2_correct"] / rule_result["total"]
    hybrid_l2 = hybrid_result["l2_correct"] / hybrid_result["total"]

    print(f"\n{'指标':<20} {'纯规则':<15} {'规则+LLM':<15} {'提升':<10}")
    print("-" * 60)
    print(f"{'L1准确率':<20} {rule_l1:<15.2%} {hybrid_l1:<15.2%} {(hybrid_l1-rule_l1):>9.1%}")
    print(f"{'L2准确率':<20} {rule_l2:<15.2%} {hybrid_l2:<15.2%} {(hybrid_l2-rule_l2):>9.1%}")

    # 找出LLM修正的案例
    rule_errors = {(e["query"], e["pred_l1"]) for e in rule_result["error_cases"]}
    hybrid_errors = {(e["query"], e["pred_l1"]) for e in hybrid_result["error_cases"]}

    fixed_by_llm = rule_errors - hybrid_errors  # 规则错误，但LLM修正了
    introduced_by_llm = hybrid_errors - rule_errors  # 规则正确，但LLM搞错了

    print(f"\nLLM修正的案例数: {len(fixed_by_llm)}")
    print(f"LLM引入的错误: {len(introduced_by_llm)}")

    if fixed_by_llm:
        print(f"\nLLM修正的案例（前5个）：")
        for i, (query, wrong_pred) in enumerate(list(fixed_by_llm)[:5], 1):
            print(f"{i}. \"{query}\" (规则错判为: {wrong_pred})")

    if introduced_by_llm:
        print(f"\nLLM引入的错误（前5个）：")
        for i, (query, wrong_pred) in enumerate(list(introduced_by_llm)[:5], 1):
            print(f"{i}. \"{query}\" (LLM错判为: {wrong_pred})")

    # 判断是否达到目标
    print(f"\n{'='*80}")
    if hybrid_l1 >= 0.85:
        print(f"✅ 达到目标！L1准确率 {hybrid_l1:.2%} ≥ 85%")
    else:
        gap = 0.85 - hybrid_l1
        print(f"⚠️  未达目标。L1准确率 {hybrid_l1:.2%}，距离85%还差 {gap:.1%}")
        print(f"   需要修正约 {int(gap * rule_result['total'])} 个错误案例")


def analyze_error_patterns(result):
    """分析错误模式"""
    print("\n" + "="*80)
    print("【错误模式分析】")
    print("="*80)

    # 统计混淆矩阵
    confusion = result["confusion_matrix_l1"]

    print("\n混淆矩阵（L1）：")
    print(f"{'真实\\预测':<20} {'知识查询':<15} {'实时查询':<15} {'混合查询':<15} {'拒答':<10}")
    print("-" * 75)

    for true_label in ["知识查询", "实时查询", "知识查询+实时查询", "引导/拒答"]:
        row = f"{true_label:<20}"
        for pred_label in ["知识查询", "实时查询", "知识查询+实时查询", "引导/拒答"]:
            count = confusion.get(true_label, {}).get(pred_label, 0)
            row += f" {count:<14}"
        print(row)

    # 找出最常混淆的类别
    max_confusion = 0
    max_pair = None
    for true_label, preds in confusion.items():
        for pred_label, count in preds.items():
            if true_label != pred_label and count > max_confusion:
                max_confusion = count
                max_pair = (true_label, pred_label)

    if max_pair:
        print(f"\n最常混淆: {max_pair[0]} → {max_pair[1]} ({max_confusion}次)")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("意图识别准确率对比测试")
    print("目标：从77% → 85%+")
    print("="*80)

    # 检查Ollama服务
    print("\n检查Ollama服务...")
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            print("✅ Ollama服务正常")
        else:
            print("⚠️  Ollama服务异常")
    except Exception as e:
        print(f"❌ Ollama服务未启动: {e}")
        print("   将只测试纯规则模式")

    try:
        # 测试1：纯规则
        rule_result = test_rule_only()

        # 测试2：规则+LLM
        try:
            hybrid_result = test_rule_plus_llm()

            # 对比分析
            compare_results(rule_result, hybrid_result)

            # 错误模式分析
            analyze_error_patterns(hybrid_result)

        except Exception as e:
            print(f"\n❌ LLM模式测试失败: {e}")
            print("   可能原因：Ollama服务未启动或模型未加载")
            import traceback
            traceback.print_exc()

        print("\n" + "="*80)
        print("✅ 测试完成")
        print("="*80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
