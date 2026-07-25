"""
评估脚本 - deep-rag系统质量评估

使用方法：
    python evaluation/run_evaluation.py              # 使用基础测试集
    python evaluation/run_evaluation.py --golden      # 使用Golden Test Set（60条标准化用例）
    python evaluation/run_evaluation.py --golden --judge llm  # Golden + LLM-as-Judge

输出：
    - evaluation/results.json：详细结果
    - evaluation/report.md：评估报告
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_dataset() -> List[Dict]:
    """加载基础测试数据集"""
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_golden_dataset() -> List[Dict]:
    """加载Golden Test Set（v2.9.1新增）

    60条标准化测试用例，5类别×3难度，包含标准答案和关键词。
    """
    dataset_path = Path(__file__).parent / "golden_test_set.json"
    if not dataset_path.exists():
        print(f"❌ Golden Test Set 不存在: {dataset_path}")
        print("请先运行 P3-1 创建 golden_test_set.json")
        sys.exit(1)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_query(query: str) -> tuple[str, float]:
    """
    运行单个查询

    返回：(答案, 响应时间)
    """
    # 导入Agent
    from src.tools.agent_executor_v2 import AgentExecutorV2

    # 创建Agent实例（只创建一次，避免重复初始化）
    if not hasattr(run_query, 'agent'):
        run_query.agent = AgentExecutorV2(
            llm_base_url="http://localhost:11434/v1/chat/completions",
            model="gemma-3-4b-it",  # 根据你的实际模型修改
            max_iterations=5,
            enable_planning=False,  # 评估时关闭规划，提高速度
            enable_memory=False,    # 评估时关闭记忆，保证独立性
            enable_logging=False    # 评估时关闭日志，提高速度
        )

    # 运行查询
    start = time.time()
    result = run_query.agent.run(query, verbose=False, use_memory=False)
    elapsed = time.time() - start

    # 提取答案
    if result["success"]:
        answer = result.get("answer", "无答案")
    else:
        answer = f"[错误] {result.get('error', '未知错误')}"

    return answer, elapsed

def evaluate_answer(query_data: Dict, answer: str) -> Dict:
    """
    评估答案质量

    评分标准：
    5分：完全正确，信息完整
    4分：大部分正确，有小瑕疵
    3分：部分正确，信息不完整
    2分：有少量正确信息，但主要错误
    1分：几乎完全错误
    0分：完全错误或无法回答
    """
    print(f"\n{'='*60}")
    print(f"问题 #{query_data['id']}: {query_data['query']}")
    print(f"类别: {query_data['category']} | 难度: {query_data['difficulty']}")
    print(f"数据源: {query_data['source_type']}")
    print(f"\n答案:\n{answer}")
    print(f"\n期望包含的关键词: {', '.join(query_data['expected_contains'])}")
    print(f"{'='*60}")

    # 检查关键词命中
    keywords_found = []
    for keyword in query_data['expected_contains']:
        if keyword.lower() in answer.lower():
            keywords_found.append(keyword)

    keyword_coverage = len(keywords_found) / len(query_data['expected_contains']) * 100
    print(f"关键词覆盖率: {keyword_coverage:.1f}% ({len(keywords_found)}/{len(query_data['expected_contains'])})")

    # 人工打分
    while True:
        try:
            score_input = input("\n请打分 (0-5，或按Enter跳过手动打分): ").strip()
            if score_input == "":
                # 基于关键词覆盖率自动打分
                if keyword_coverage >= 80:
                    score = 5
                elif keyword_coverage >= 60:
                    score = 4
                elif keyword_coverage >= 40:
                    score = 3
                elif keyword_coverage >= 20:
                    score = 2
                elif keyword_coverage > 0:
                    score = 1
                else:
                    score = 0
                print(f"自动打分: {score}分 (基于关键词覆盖率)")
                break
            score = int(score_input)
            if 0 <= score <= 5:
                break
            print("请输入0-5之间的整数")
        except ValueError:
            print("请输入有效的数字")

    # 记录错误类型
    error_type = None
    if score <= 2:
        print("\n错误类型:")
        print("1. 幻觉（编造信息）")
        print("2. 信息遗漏")
        print("3. 理解错误")
        print("4. 完全无法回答")
        print("5. 其他")
        error_input = input("选择错误类型 (1-5，或按Enter跳过): ").strip()
        error_types = {
            "1": "幻觉",
            "2": "信息遗漏",
            "3": "理解错误",
            "4": "无法回答",
            "5": "其他"
        }
        error_type = error_types.get(error_input)

    return {
        "score": score,
        "keyword_coverage": keyword_coverage,
        "keywords_found": keywords_found,
        "error_type": error_type
    }

def generate_report(dataset: List[Dict], results: List[Dict]) -> str:
    """生成评估报告"""

    # 统计数据
    total_queries = len(results)
    avg_score = sum(r['score'] for r in results) / total_queries
    avg_time = sum(r['time'] for r in results) / total_queries

    # 按难度统计
    difficulty_stats = {}
    for difficulty in ['easy', 'medium', 'hard']:
        filtered = [r for r in results if r['difficulty'] == difficulty]
        if filtered:
            difficulty_stats[difficulty] = {
                'count': len(filtered),
                'avg_score': sum(r['score'] for r in filtered) / len(filtered),
                'avg_time': sum(r['time'] for r in filtered) / len(filtered)
            }

    # 按类别统计
    category_stats = {}
    for result in results:
        category = result['category']
        if category not in category_stats:
            category_stats[category] = []
        category_stats[category].append(result['score'])

    for category in category_stats:
        scores = category_stats[category]
        category_stats[category] = {
            'count': len(scores),
            'avg_score': sum(scores) / len(scores)
        }

    # 统计错误类型
    error_counts = {}
    for result in results:
        if result.get('error_type'):
            error_type = result['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

    # 生成报告
    report = f"""# Deep-RAG 系统评估报告

**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**数据集规模**: {total_queries}个真实查询
**评估维度**: 准确性、响应时间、完整性

---

## 📊 核心指标

| 指标 | 数值 |
|------|------|
| **平均准确率** | {avg_score/5*100:.1f}% ({avg_score:.2f}/5) |
| **平均响应时间** | {avg_time:.2f}秒 |
| **完全正确率** | {len([r for r in results if r['score'] == 5])/total_queries*100:.1f}% ({len([r for r in results if r['score'] == 5])}/{total_queries}) |
| **部分正确率** | {len([r for r in results if 3 <= r['score'] < 5])/total_queries*100:.1f}% ({len([r for r in results if 3 <= r['score'] < 5])}/{total_queries}) |
| **错误率** | {len([r for r in results if r['score'] < 3])/total_queries*100:.1f}% ({len([r for r in results if r['score'] < 3])}/{total_queries}) |

---

## 📈 按难度分析

| 难度 | 查询数 | 平均准确率 | 平均响应时间 |
|------|--------|-----------|-------------|
"""

    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in difficulty_stats:
            stats = difficulty_stats[difficulty]
            report += f"| {difficulty} | {stats['count']} | {stats['avg_score']/5*100:.1f}% | {stats['avg_time']:.2f}秒 |\n"

    report += f"""
---

## 📋 按类别分析

| 类别 | 查询数 | 平均准确率 |
|------|--------|-----------|
"""

    for category, stats in sorted(category_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True):
        report += f"| {category} | {stats['count']} | {stats['avg_score']/5*100:.1f}% |\n"

    if error_counts:
        report += f"""
---

## ⚠️ 错误分析

| 错误类型 | 数量 | 占比 |
|---------|------|------|
"""
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"| {error_type} | {count} | {count/total_queries*100:.1f}% |\n"

    # 典型错误案例
    error_cases = [r for r in results if r['score'] <= 2]
    if error_cases:
        report += f"""
---

## 🔍 典型错误案例（得分≤2）

"""
        for i, case in enumerate(error_cases[:5], 1):  # 只展示前5个
            report += f"""
### 案例{i}: {case.get('error_type', '未分类错误')}

- **问题**: {case['query']}
- **类别**: {case['category']} | **难度**: {case['difficulty']}
- **得分**: {case['score']}/5
- **答案**: {case['answer'][:200]}{'...' if len(case['answer']) > 200 else ''}
- **期望关键词**: {', '.join(case['expected_contains'])}
- **关键词覆盖**: {case['keyword_coverage']:.1f}%
"""

    report += f"""
---

## 💡 改进建议

基于评估结果，建议优先改进以下方面：

"""

    # 生成改进建议
    if avg_score < 4:
        report += "1. **提升整体准确率**: 当前平均准确率{:.1f}%，距离优秀水平（80%+）还有差距\n".format(avg_score/5*100)

    if 'hard' in difficulty_stats and difficulty_stats['hard']['avg_score'] < 3.5:
        report += "2. **加强复杂查询处理**: 复杂查询准确率较低，需优化多步推理能力\n"

    if error_counts.get('幻觉', 0) > 0:
        report += "3. **减少幻觉问题**: 发现{}个幻觉案例，需加强Self-RAG校验\n".format(error_counts['幻觉'])

    if error_counts.get('信息遗漏', 0) > 0:
        report += "4. **提升信息完整性**: 发现{}个信息遗漏案例，需增加检索Top-K或优化精排\n".format(error_counts['信息遗漏'])

    if avg_time > 3:
        report += "5. **优化响应时间**: 当前平均响应时间{:.2f}秒，可考虑缓存和并行优化\n".format(avg_time)

    return report

def evaluate_golden(query_data: Dict, answer: str) -> Dict:
    """评估Golden Test Set答案（v2.9.1新增）

    评估维度：
    - keyword_coverage: 期望关键词命中率
    - answer_relevancy: 答案与期望答案的相似度（简单关键词重叠）
    - citation_check: 引用来源匹配
    """
    expected_keywords = query_data.get("expected_keywords", [])
    expected_answer = query_data.get("expected_answer", "")
    expected_citations = query_data.get("expected_citations", [])

    # 关键词覆盖率
    keywords_found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
    keyword_coverage = len(keywords_found) / max(len(expected_keywords), 1) * 100

    # 答案相似度（简单词重叠率）
    answer_words = set(answer.lower().split())
    expected_words = set(expected_answer.lower().split())
    overlap = len(answer_words & expected_words) / max(len(expected_words), 1) * 100

    # 引用来源匹配
    citations_found = [c for c in expected_citations if c.lower() in answer.lower()]
    citation_match = len(citations_found) / max(len(expected_citations), 1) * 100

    # 综合评分（0-10）
    score = min(10, (keyword_coverage * 0.4 + overlap * 0.4 + citation_match * 0.2) / 10)

    return {
        "score": round(score, 1),
        "keyword_coverage": round(keyword_coverage, 1),
        "answer_similarity": round(overlap, 1),
        "citation_match": round(citation_match, 1),
        "keywords_found": keywords_found,
        "citations_found": citations_found,
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepRAG 系统评估")
    parser.add_argument("--golden", action="store_true",
                        help="使用Golden Test Set（60条标准化用例）")
    parser.add_argument("--judge", choices=["keyword", "llm"], default="keyword",
                        help="评估方式：keyword（关键词）或 llm（LLM-as-Judge）")
    args = parser.parse_args()

    print("="*60)
    print("Deep-RAG 系统评估")
    if args.golden:
        print("📊 模式: Golden Test Set（60条标准化用例）")
    else:
        print("📊 模式: 基础测试集")
    if args.judge == "llm":
        print("⚖️ 评估方式: LLM-as-Judge")
    print("="*60)

    # 加载数据集
    print("\n加载测试数据集...")
    if args.golden:
        dataset = load_golden_dataset()
    else:
        dataset = load_dataset()
    print(f"✓ 加载了 {len(dataset)} 个测试查询")

    # 运行评估
    print("\n开始评估（可随时按Ctrl+C中断）...\n")
    results = []
    use_golden = args.golden

    try:
        for i, query_data in enumerate(dataset, 1):
            print(f"\n[{i}/{len(dataset)}] 正在处理...")

            # 运行查询
            answer, elapsed = run_query(query_data['query'])

            # 评估答案
            if use_golden:
                evaluation = evaluate_golden(query_data, answer)
            else:
                evaluation = evaluate_answer(query_data, answer)

            # LLM-as-Judge评估（如果指定）
            if args.judge == "llm" and use_golden:
                try:
                    from src.evaluation.llm_judge import LLMJudge
                    judge = LLMJudge()
                    llm_result = judge.evaluate(
                        query_data['query'], answer,
                        query_data.get('expected_answer', ''),
                        query_data.get('context', '')
                    )
                    evaluation["llm_judge"] = llm_result
                    evaluation["score"] = llm_result.get("overall", evaluation["score"])
                except Exception as e:
                    print(f"  ⚠️ LLM-as-Judge 失败: {e}")

            # 保存结果
            result = {
                'id': query_data['id'],
                'query': query_data['query'],
                'category': query_data.get('category', ''),
                'difficulty': query_data.get('difficulty', ''),
                'answer': answer,
                'time': elapsed,
                **evaluation,
            }
            results.append(result)

            # 实时保存（防止中断丢失数据）
            output_file = "golden_results.json" if use_golden else "results.json"
            with open(Path(__file__).parent / output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("\n\n用户中断评估")

    # 生成报告
    print("\n\n生成评估报告...")
    report = generate_report(dataset, results)

    # 保存报告
    report_file = "golden_report.md" if use_golden else "report.md"
    report_path = Path(__file__).parent / report_file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ 评估完成！")
    output_file = "golden_results.json" if use_golden else "results.json"
    print(f"✓ 详细结果: evaluation/{output_file}")
    print(f"✓ 评估报告: evaluation/{report_file}")
    print(f"\n已完成 {len(results)}/{len(dataset)} 个查询的评估")

if __name__ == "__main__":
    main()
