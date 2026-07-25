#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 评测系统 v3.0 - 企业级详细评分版
结合心理人际项目评分方法 + RAGAS 2026 标准

核心改进：
1. 多维度评分（准确性/相关性/简洁性/忠实性/完整性）
2. 每个维度详细分级（0-100 分，含加分点/扣分点/遗漏点）
3. 章节独立评分（参考心理人际项目）
4. RAGAS 标准指标（Faithfulness/Context Precision/Answer Relevance）
5. 自动生成优化建议

作者：基于心理人际项目 + RAGAS 2026
日期：2026-06-05
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

try:
    import httpx
except ImportError:
    print("请安装 httpx: pip install httpx")
    sys.exit(1)


# ════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════

LLM_CONFIG = {
    "base_url": "http://localhost:11434/v1",
    "model": "gemma3-vl:4b",
    "temperature": 0.1,
    "max_tokens": 300,  # 增加到 300，支持详细评分
}

# 评分标准（参考心理人际项目）
FAIL_THRESHOLD = 40      # <40分：不合格
PASS_THRESHOLD = 60      # 40-60分：勉强合格
GOOD_THRESHOLD = 80      # 60-80分：良好
EXCELLENT_THRESHOLD = 90 # ≥90分：优秀


# ════════════════════════════════════════════════
# 测试集定义
# ════════════════════════════════════════════════

def load_test_cases() -> List[Dict]:
    """加载测试用例"""
    from rag_auto_eval_v2 import TEST_CASES

    all_cases = []
    for category, questions in TEST_CASES.items():
        for item in questions:
            all_cases.append({
                "category": category,
                "question": item["q"],
                "answer": item["a"],
            })

    return all_cases


# ════════════════════════════════════════════════
# LLM 调用
# ════════════════════════════════════════════════

def call_llm(prompt: str, max_tokens: int = None) -> str:
    """调用 LLM"""
    try:
        client = httpx.Client(timeout=30.0)
        response = client.post(
            f"{LLM_CONFIG['base_url']}/chat/completions",
            json={
                "model": LLM_CONFIG["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": LLM_CONFIG["temperature"],
                "max_tokens": max_tokens or LLM_CONFIG["max_tokens"],
            }
        )
        result = response.json()["choices"][0]["message"]["content"].strip()
        return result
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return "错误"


# ════════════════════════════════════════════════
# 多维度详细评分（参考心理人际项目 + RAGAS 标准）
# ════════════════════════════════════════════════

def evaluate_answer_comprehensive(
    question: str,
    ground_truth: str,
    rag_response: str,
    retrieval_docs: str = ""
) -> Dict:
    """
    综合评分（5 个维度）

    参考：
    - 心理人际项目：章节评分方法（100 分制 + 分级判定）
    - RAGAS 2026：Faithfulness/Context Precision/Answer Relevance

    Returns:
        {
            "total_score": 85,  # 总分（0-100）
            "grade": "良好 ✅",  # 等级
            "dimensions": {
                "accuracy": {...},      # 准确性（30分）
                "relevance": {...},     # 相关性（20分）
                "completeness": {...},  # 完整性（20分）
                "faithfulness": {...},  # 忠实性（15分）
                "conciseness": {...}    # 简洁性（15分）
            }
        }
    """
    prompt = f"""你是 RAG 评测专家。请对以下 RAG 回答进行全面评分。

用户问题：{question}
标准答案：{ground_truth}
RAG 回答：{rag_response}
检索文档：{retrieval_docs if retrieval_docs else "（未提供）"}

请按以下 5 个维度评分，每个维度 0-100 分：

【1. 准确性 Accuracy（30分权重）】
- 评分：[0-100]
- 理由：[核心判断依据]
- ✅ 加分点：[列出做得好的地方，用分号分隔]
- ❌ 扣分点：[列出不准确的地方，用分号分隔；如果没有则写"无"]
- ⚠️ 遗漏点：[标准答案有但RAG没有的关键信息，用分号分隔；如果没有则写"无"]

【2. 相关性 Relevance（20分权重）】
- 评分：[0-100]
- 理由：[回答是否切题]
- ✅ 加分点：[相关性强的地方]
- ❌ 扣分点：[偏题或无关的地方；如果没有则写"无"]

【3. 完整性 Completeness（20分权重）】
- 评分：[0-100]
- 理由：[回答是否完整覆盖问题]
- ✅ 加分点：[覆盖全面的地方]
- ❌ 缺失项：[应该有但没有的内容；如果没有则写"无"]

【4. 忠实性 Faithfulness（15分权重）】
- 评分：[0-100]
- 理由：[回答是否基于检索文档，有无幻觉]
- ✅ 加分点：[有事实依据的地方]
- ❌ 幻觉点：[没有依据的陈述；如果没有则写"无"]

【5. 简洁性 Conciseness（15分权重）】
- 评分：[0-100]
- 理由：[是否简洁无冗余]
- ✅ 加分点：[精炼得当的地方]
- ❌ 扣分点：[啰嗦冗余的地方；如果没有则写"无"]

请严格按照上述格式输出，每个维度必须包含所有字段。"""

    result = call_llm(prompt, max_tokens=500)

    # 解析结果
    dimensions = _parse_evaluation_result(result)

    # 计算总分（加权平均）
    weights = {
        "accuracy": 0.30,
        "relevance": 0.20,
        "completeness": 0.20,
        "faithfulness": 0.15,
        "conciseness": 0.15
    }

    total_score = sum(
        dimensions[dim]["score"] * weight
        for dim, weight in weights.items()
    )

    # 判定等级（参考心理人际项目）
    if total_score >= EXCELLENT_THRESHOLD:
        grade = "优秀 ⭐⭐⭐⭐⭐"
    elif total_score >= GOOD_THRESHOLD:
        grade = "良好 ✅"
    elif total_score >= PASS_THRESHOLD:
        grade = "勉强合格 ⚠️"
    elif total_score >= FAIL_THRESHOLD:
        grade = "不合格 ❌"
    else:
        grade = "严重不合格 ❌❌"

    return {
        "total_score": round(total_score, 1),
        "grade": grade,
        "dimensions": dimensions,
        "weights": weights
    }


def _parse_evaluation_result(result: str) -> Dict:
    """解析 LLM 评测结果"""
    dimensions = {
        "accuracy": {"score": 0, "reason": "", "good": [], "bad": [], "missing": []},
        "relevance": {"score": 0, "reason": "", "good": [], "bad": []},
        "completeness": {"score": 0, "reason": "", "good": [], "missing": []},
        "faithfulness": {"score": 0, "reason": "", "good": [], "hallucination": []},
        "conciseness": {"score": 0, "reason": "", "good": [], "bad": []}
    }

    current_dim = None
    lines = result.strip().split('\n')

    for line in lines:
        line = line.strip()

        # 识别维度
        if "准确性" in line or "Accuracy" in line:
            current_dim = "accuracy"
        elif "相关性" in line or "Relevance" in line:
            current_dim = "relevance"
        elif "完整性" in line or "Completeness" in line:
            current_dim = "completeness"
        elif "忠实性" in line or "Faithfulness" in line:
            current_dim = "faithfulness"
        elif "简洁性" in line or "Conciseness" in line:
            current_dim = "conciseness"

        if not current_dim:
            continue

        # 解析字段
        if line.startswith("评分") or line.startswith("- 评分"):
            score_text = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            try:
                dimensions[current_dim]["score"] = int(''.join(filter(str.isdigit, score_text)) or "0")
            except:
                pass
        elif line.startswith("理由") or line.startswith("- 理由"):
            dimensions[current_dim]["reason"] = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
        elif "加分点" in line:
            points = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            if points and points != "无":
                dimensions[current_dim]["good"] = [p.strip() for p in points.split('；') if p.strip()]
        elif "扣分点" in line:
            points = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            if points and points != "无":
                dimensions[current_dim]["bad"] = [p.strip() for p in points.split('；') if p.strip()]
        elif "遗漏点" in line or "缺失项" in line:
            points = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            if points and points != "无":
                dimensions[current_dim]["missing"] = [p.strip() for p in points.split('；') if p.strip()]
        elif "幻觉点" in line:
            points = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            if points and points != "无":
                dimensions[current_dim]["hallucination"] = [p.strip() for p in points.split('；') if p.strip()]

    return dimensions


# ════════════════════════════════════════════════
# 简化版 RAG 调用
# ════════════════════════════════════════════════

def call_rag_system(question: str) -> Tuple[str, str]:
    """
    调用 RAG 系统

    Returns:
        (rag_response, retrieval_docs)
    """
    # 这里模拟 RAG 调用
    prompt = f"请简洁回答：{question}"
    rag_response = call_llm(prompt, max_tokens=200)
    retrieval_docs = "（模拟检索文档）"

    return rag_response, retrieval_docs
# ════════════════════════════════════════════════
# 批量评测
# ════════════════════════════════════════════════

def batch_evaluate(test_cases: List[Dict], limit: int = None) -> List[Dict]:
    """批量评测（v3.0 企业级版本）"""
    if limit:
        test_cases = test_cases[:limit]

    results = []
    total = len(test_cases)

    print(f"\n🚀 开始评测 {total} 个问题（v3.0 企业级版本）")
    print("="*80)

    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{total}] {case['category']}")
        print(f"问题: {case['question']}")

        # 1. 调用 RAG 系统
        start_time = time.time()
        rag_response, retrieval_docs = call_rag_system(case['question'])
        response_time = time.time() - start_time

        print(f"回答: {rag_response[:80]}...")
        print(f"耗时: {response_time:.2f}秒")

        # 2. 综合评分（5 个维度）
        eval_result = evaluate_answer_comprehensive(
            question=case['question'],
            ground_truth=case['answer'],
            rag_response=rag_response,
            retrieval_docs=retrieval_docs
        )

        print(f"\n📊 综合评分: {eval_result['total_score']:.1f}/100 - {eval_result['grade']}")

        # 打印各维度得分
        for dim_name, dim_data in eval_result['dimensions'].items():
            weight = eval_result['weights'][dim_name]
            print(f"\n  【{dim_name.upper()}】{dim_data['score']}/100 (权重{weight*100:.0f}%)")
            print(f"    理由: {dim_data['reason']}")
            if dim_data.get('good'):
                print(f"    ✅ 加分: {'; '.join(dim_data['good'][:2])}")  # 只显示前2个
            if dim_data.get('bad'):
                print(f"    ❌ 扣分: {'; '.join(dim_data['bad'][:2])}")
            if dim_data.get('missing'):
                print(f"    ⚠️  遗漏: {'; '.join(dim_data['missing'][:2])}")

        # 3. 保存结果
        result_data = {
            "category": case['category'],
            "question": case['question'],
            "ground_truth": case['answer'],
            "rag_response": rag_response,
            "retrieval_docs": retrieval_docs,
            "total_score": eval_result['total_score'],
            "grade": eval_result['grade'],
            "response_time": response_time,
        }

        # 展开各维度数据
        for dim_name, dim_data in eval_result['dimensions'].items():
            result_data[f"{dim_name}_score"] = dim_data['score']
            result_data[f"{dim_name}_reason"] = dim_data['reason']
            result_data[f"{dim_name}_good"] = "; ".join(dim_data.get('good', []))
            result_data[f"{dim_name}_bad"] = "; ".join(dim_data.get('bad', []))
            if 'missing' in dim_data:
                result_data[f"{dim_name}_missing"] = "; ".join(dim_data.get('missing', []))
            if 'hallucination' in dim_data:
                result_data[f"{dim_name}_hallucination"] = "; ".join(dim_data.get('hallucination', []))

        results.append(result_data)

        # 保存检查点
        if (i + 1) % 5 == 0:
            save_checkpoint(results)

    return results


def save_checkpoint(results: List[Dict]):
    """保存检查点"""
    checkpoint_file = "rag_evaluation_v3_checkpoint.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存检查点: {checkpoint_file}")


def save_results(results: List[Dict], output_file: str):
    """保存最终结果"""
    # JSON
    json_file = output_file.replace(".xlsx", ".json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

    print(f"\n✅ 结果已保存:")
    print(f"  - JSON: {json_file}")
    print(f"  - Excel: {output_file}")

# ════════════════════════════════════════════════
# 统计分析（参考心理人际项目 + RAGAS 标准）
# ════════════════════════════════════════════════

def analyze_results(results: List[Dict]) -> Dict:
    """统计分析（企业级）"""
    total = len(results)

    # 总体指标
    avg_score = sum(r['total_score'] for r in results) / total
    avg_time = sum(r['response_time'] for r in results) / total

    # 等级分布（参考心理人际项目）
    grade_distribution = {
        "优秀 ⭐⭐⭐⭐⭐": sum(1 for r in results if r['total_score'] >= EXCELLENT_THRESHOLD),
        "良好 ✅": sum(1 for r in results if GOOD_THRESHOLD <= r['total_score'] < EXCELLENT_THRESHOLD),
        "勉强合格 ⚠️": sum(1 for r in results if PASS_THRESHOLD <= r['total_score'] < GOOD_THRESHOLD),
        "不合格 ❌": sum(1 for r in results if r['total_score'] < PASS_THRESHOLD),
    }

    # 各维度平均分
    dimensions_avg = {}
    for dim in ['accuracy', 'relevance', 'completeness', 'faithfulness', 'conciseness']:
        scores = [r[f'{dim}_score'] for r in results if f'{dim}_score' in r]
        if scores:
            dimensions_avg[dim] = sum(scores) / len(scores)

    # 按类型统计
    by_category = {}
    for r in results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    category_stats = {}
    for cat, items in by_category.items():
        cat_total = len(items)
        cat_avg_score = sum(r['total_score'] for r in items) / cat_total
        cat_pass_rate = sum(1 for r in items if r['total_score'] >= PASS_THRESHOLD) / cat_total * 100
        category_stats[cat] = {
            "total": cat_total,
            "avg_score": cat_avg_score,
            "pass_rate": cat_pass_rate
        }

    # 统计常见问题（Top 10）
    common_issues = {}
    for r in results:
        # 收集所有扣分点和遗漏点
        for dim in ['accuracy', 'relevance', 'completeness', 'faithfulness', 'conciseness']:
            for issue_type in ['bad', 'missing', 'hallucination']:
                key = f'{dim}_{issue_type}'
                if key in r and r[key]:
                    for issue in r[key].split('; '):
                        if issue and issue != "无":
                            common_issues[issue] = common_issues.get(issue, 0) + 1

    # 排序 Top 10
    top_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total": total,
        "avg_score": avg_score,
        "avg_time": avg_time,
        "grade_distribution": grade_distribution,
        "dimensions_avg": dimensions_avg,
        "category_stats": category_stats,
        "top_issues": top_issues
    }


def print_analysis(stats: Dict):
    """打印分析结果（企业级报告）"""
    print("\n" + "="*80)
    print("📊 RAG 评测结果分析报告（v3.0 企业级）")
    print("="*80)

    # 1. 总体指标
    print(f"\n【总体指标】")
    print(f"  测试用例数: {stats['total']}")
    print(f"  平均得分: {stats['avg_score']:.1f}/100")
    print(f"  平均响应时间: {stats['avg_time']:.2f}秒")

    # 2. 等级分布
    print(f"\n【等级分布】")
    for grade, count in stats['grade_distribution'].items():
        percentage = count / stats['total'] * 100
        print(f"  {grade}: {count} 个 ({percentage:.1f}%)")

    # 3. 各维度平均分
    print(f"\n【各维度平均分】")
    dim_names = {
        'accuracy': '准确性',
        'relevance': '相关性',
        'completeness': '完整性',
        'faithfulness': '忠实性',
        'conciseness': '简洁性'
    }
    for dim, score in stats['dimensions_avg'].items():
        print(f"  {dim_names[dim]}: {score:.1f}/100")

    # 4. 按类型统计
    print(f"\n【按类型统计】")
    for cat, cat_stats in stats['category_stats'].items():
        print(f"  {cat}: 平均 {cat_stats['avg_score']:.1f}分, "
              f"及格率 {cat_stats['pass_rate']:.1f}% ({cat_stats['total']} 个)")

    # 5. Top 10 常见问题
    if stats['top_issues']:
        print(f"\n【🔴 Top 10 常见问题】")
        for i, (issue, count) in enumerate(stats['top_issues'], 1):
            print(f"  {i}. {issue} ({count} 次)")

    # 6. 优化建议
    print(f"\n【💡 优化建议】")
    if stats['top_issues']:
        top3 = stats['top_issues'][:3]
        print(f"  优先解决以下问题：")
        for issue, count in top3:
            print(f"    • {issue}")

    # 7. RAGAS 对标
    print(f"\n【📈 RAGAS 标准对标】")
    print(f"  Faithfulness (忠实性): {stats['dimensions_avg'].get('faithfulness', 0):.1f}/100")
    print(f"  Context Precision (相关性): {stats['dimensions_avg'].get('relevance', 0):.1f}/100")
    print(f"  Answer Relevance (准确性): {stats['dimensions_avg'].get('accuracy', 0):.1f}/100")

    print("\n" + "="*80)


# ════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n使用方法:")
        print("  python rag_evaluation_v3.py test    # 测试 5 个问题")
        print("  python rag_evaluation_v3.py run     # 完整评测 50 个")
        print("  python rag_evaluation_v3.py analyze # 分析已有结果")
        sys.exit(0)

    command = sys.argv[1]

    if command == "test":
        print("🧪 测试模式（5 个问题）")
        test_cases = load_test_cases()
        results = batch_evaluate(test_cases, limit=5)
        save_results(results, "rag_evaluation_v3_test.xlsx")
        stats = analyze_results(results)
        print_analysis(stats)

    elif command == "run":
        print("🚀 完整评测（50 个问题）")
        test_cases = load_test_cases()
        results = batch_evaluate(test_cases)
        save_results(results, "rag_evaluation_v3_full.xlsx")
        stats = analyze_results(results)
        print_analysis(stats)

    elif command == "analyze":
        json_file = "rag_evaluation_v3_full.json"
        if not Path(json_file).exists():
            print(f"❌ 找不到结果文件: {json_file}")
            sys.exit(1)

        with open(json_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        stats = analyze_results(results)
        print_analysis(stats)

    else:
        print(f"未知命令: {command}")


if __name__ == "__main__":
    main()
