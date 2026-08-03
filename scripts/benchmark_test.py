#!/usr/bin/env python3
"""DeepRAG 性能基准测试 — L4 层级

测量指标：
1. 响应时间：P50/P90/P99
2. 召回率：Hit@K（前K个结果中包含正确答案的比例）
3. Token 使用量：输入/输出 token 数

用法：
    python scripts/benchmark_test.py
    python scripts/benchmark_test.py --rounds 5  # 每题运行5轮
"""
import argparse
import json
import time
import sys
import statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# ANSI 颜色码
# ============================================================
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{Color.RESET}"


# ============================================================
# 选取测试题（每类别4题，共20题）
# ============================================================

BENCHMARK_QUESTIONS = [
    # MBTI
    {"id": 1, "category": "MBTI", "question": "INTJ的主导功能是什么？", "expected_keywords": ["Ni", "内向直觉"]},
    {"id": 3, "category": "MBTI", "question": "INTJ和INFJ的核心区别是什么？", "expected_keywords": ["Te", "Fe"]},
    {"id": 5, "category": "MBTI", "question": "INTJ在压力下会表现出哪些特征？", "expected_keywords": ["Se", "劣势功能"]},
    {"id": 9, "category": "MBTI", "question": "ISTJ的性格特点是什么？", "expected_keywords": ["Si", "责任感"]},
    # RAG
    {"id": 21, "category": "RAG", "question": "什么是RAG？", "expected_keywords": ["检索增强生成"]},
    {"id": 23, "category": "RAG", "question": "什么是混合检索？", "expected_keywords": ["BM25", "向量"]},
    {"id": 27, "category": "RAG", "question": "什么是重排序（Reranking）？", "expected_keywords": ["精排", "Cross-Encoder"]},
    {"id": 33, "category": "RAG", "question": "什么是Embedding模型？", "expected_keywords": ["向量化", "bge"]},
    # LLM
    {"id": 41, "category": "LLM", "question": "什么是Transformer？", "expected_keywords": ["注意力机制"]},
    {"id": 43, "category": "LLM", "question": "什么是KV Cache？", "expected_keywords": ["Key", "Value", "缓存"]},
    {"id": 46, "category": "LLM", "question": "什么是Temperature参数？", "expected_keywords": ["随机性", "采样"]},
    {"id": 50, "category": "LLM", "question": "什么是RLHF？", "expected_keywords": ["人类反馈", "强化学习"]},
    # Agent
    {"id": 61, "category": "Agent", "question": "什么是Function Calling？", "expected_keywords": ["工具调用"]},
    {"id": 62, "category": "Agent", "question": "什么是MCP协议？", "expected_keywords": ["Model Context Protocol"]},
    {"id": 66, "category": "Agent", "question": "什么是ReAct模式？", "expected_keywords": ["推理", "行动"]},
    {"id": 76, "category": "Agent", "question": "什么是SSE？", "expected_keywords": ["Server-Sent Events"]},
    # Engineering
    {"id": 81, "category": "Engineering", "question": "如何优化LLM推理速度？", "expected_keywords": ["量化", "KV Cache"]},
    {"id": 83, "category": "Engineering", "question": "什么是模型路由？", "expected_keywords": ["任务类型", "选择模型"]},
    {"id": 89, "category": "Engineering", "question": "什么是降级策略？", "expected_keywords": ["备用方案", "兜底"]},
    {"id": 93, "category": "Engineering", "question": "什么是缓存？", "expected_keywords": ["存储", "复用"]},
]


# ============================================================
# 统计工具
# ============================================================

def calculate_percentiles(latencies: list) -> dict:
    """计算 P50/P90/P99 百分位

    Args:
        latencies: 延迟列表（秒）

    Returns:
        {"p50": float, "p90": float, "p99": float, "mean": float, "min": float, "max": float}
    """
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(p: float) -> float:
        """计算百分位（线性插值法）"""
        if n == 1:
            return round(sorted_lat[0], 3)
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= n:
            return round(sorted_lat[-1], 3)
        return round(sorted_lat[f] + (k - f) * (sorted_lat[c] - sorted_lat[f]), 3)

    return {
        "p50": percentile(50),
        "p90": percentile(90),
        "p99": percentile(99),
        "mean": round(statistics.mean(latencies), 3),
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
    }


def calculate_hit_at_k(answer: str, expected_keywords: list, k: int = 5) -> dict:
    """计算召回率 Hit@K

    判断预期关键词在答案中的命中情况。
    Hit@K 在此实现为：预期关键词命中率（前K个关键词中命中的比例）。

    Args:
        answer: 系统回答
        expected_keywords: 预期关键词列表
        k: 取前K个关键词计算

    Returns:
        {"hit_at_k": float, "hit_count": int, "total": int, "hits": list}
    """
    if not expected_keywords:
        return {"hit_at_k": 0.0, "hit_count": 0, "total": 0, "hits": []}

    top_k = expected_keywords[:k]
    answer_lower = answer.lower()

    hits = []
    for kw in top_k:
        if kw.lower() in answer_lower:
            hits.append(kw)

    hit_count = len(hits)
    hit_rate = round(hit_count / len(top_k), 4)

    return {
        "hit_at_k": hit_rate,
        "hit_count": hit_count,
        "total": len(top_k),
        "hits": hits,
    }


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约1.5字/token，英文约4字符/token）"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4)


# ============================================================
# RAG 调用
# ============================================================

def _try_import_rag():
    """尝试导入 RAG query 函数，不可用则返回 None"""
    try:
        from src.graph import query as rag_query
        return rag_query
    except Exception:
        return None


def call_rag(question: str) -> dict:
    """调用 RAG 管道

    Returns:
        {"answer": str, "sources": list, "error": str, "tokens_in": int, "tokens_out": int}
    """
    rag_query = _try_import_rag()
    if rag_query is None:
        raise ImportError("src.graph.query 不可用")

    result = rag_query(question)

    if isinstance(result, dict):
        answer = result.get("answer", str(result))
        sources = result.get("citations", result.get("sources", []))
    else:
        answer = str(result)
        sources = []

    tokens_in = estimate_tokens(question)
    tokens_out = estimate_tokens(answer)

    return {
        "answer": answer,
        "sources": sources if isinstance(sources, list) else [],
        "error": None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def simulate_rag(question: str) -> dict:
    """模拟 RAG 调用（当真实管道不可用时使用）

    Returns:
        {"answer": str, "sources": list, "error": str, "tokens_in": int, "tokens_out": int}
    """
    import random

    # 基于问题生成模拟回答
    simulated_answers = {
        "MBTI": "根据认知功能理论，该问题涉及MBTI类型的核心特征分析。MBTI基于荣格的认知功能理论，将人的性格分为16种类型。",
        "RAG": "RAG（检索增强生成）是一种结合检索和生成的技术方案，通过从知识库中检索相关文档来增强LLM的生成质量。",
        "LLM": "大语言模型的核心机制基于Transformer架构，通过注意力机制实现序列建模和文本生成。",
        "Agent": "Agent系统通过工具调用和推理循环实现自主决策，核心在于Reasoning和Action的交替执行。",
        "Engineering": "工程实践中的优化策略需要综合考虑性能、成本和可用性，通过缓存、路由和降级等手段提升系统质量。",
    }

    # 从问题中提取类别线索
    category = "Engineering"
    for cat in simulated_answers:
        if cat in question or any(kw in question for kw in simulated_answers[cat][:5]):
            category = cat
            break

    answer = simulated_answers.get(category, simulated_answers["Engineering"])
    # 添加一些随机性
    answer += f" 这是对「{question[:20]}...」的模拟回答。"

    # 模拟延迟
    time.sleep(random.uniform(0.1, 0.3))

    tokens_in = estimate_tokens(question)
    tokens_out = estimate_tokens(answer)

    return {
        "answer": answer,
        "sources": [],
        "error": None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


# ============================================================
# 单题基准测试
# ============================================================

def run_single_benchmark(question_data: dict, rounds: int = 3) -> dict:
    """运行单题多次基准测试

    Args:
        question_data: 题目数据
        rounds: 运行轮数

    Returns:
        {"id": int, "category": str, "question": str,
         "latencies": list, "percentiles": dict,
         "hit_at_k": dict, "tokens_in": int, "tokens_out": int,
         "errors": list, "answers": list, "simulation": bool}
    """
    question = question_data["question"]
    expected_keywords = question_data.get("expected_keywords", [])

    latencies = []
    answers = []
    errors = []
    tokens_in_total = 0
    tokens_out_total = 0
    is_simulation = False

    # 判断是否使用模拟模式
    rag_available = _try_import_rag() is not None
    if not rag_available:
        is_simulation = True
        print(colorize("    [模拟模式] src.graph.query 不可用，使用模拟数据", Color.YELLOW))

    for i in range(rounds):
        start = time.time()

        try:
            if rag_available:
                result = call_rag(question)
            else:
                result = simulate_rag(question)

            elapsed = round(time.time() - start, 3)
            latencies.append(elapsed)
            answers.append(result["answer"])
            tokens_in_total += result["tokens_in"]
            tokens_out_total += result["tokens_out"]
            errors.append(None)

            status = colorize("OK", Color.GREEN)
        except Exception as e:
            elapsed = round(time.time() - start, 3)
            latencies.append(elapsed)
            answers.append(f"[ERROR] {e}")
            errors.append(str(e))

            # 模拟模式下不应该出错，出错说明真实调用失败
            if not is_simulation:
                is_simulation = True
                print(colorize(f"    [降级] RAG 调用失败: {e}，切换到模拟模式", Color.YELLOW))
                rag_available = False

            status = colorize("ERR", Color.RED)

        print(f"    Round {i+1}/{rounds}: {elapsed:.3f}s  {status}")

    # 计算统计
    percentiles = calculate_percentiles(latencies)

    # Hit@K 取最后一轮有效答案
    valid_answers = [a for a, e in zip(answers, errors) if e is None]
    if valid_answers:
        hit_result = calculate_hit_at_k(valid_answers[-1], expected_keywords, k=5)
    else:
        hit_result = {"hit_at_k": 0.0, "hit_count": 0, "total": len(expected_keywords[:5]), "hits": []}

    return {
        "id": question_data["id"],
        "category": question_data["category"],
        "question": question,
        "latencies": latencies,
        "percentiles": percentiles,
        "hit_at_k": hit_result,
        "tokens_in_avg": round(tokens_in_total / max(rounds, 1)),
        "tokens_out_avg": round(tokens_out_total / max(rounds, 1)),
        "errors": [e for e in errors if e is not None],
        "answers": answers,
        "simulation": is_simulation,
    }


# ============================================================
# 全量基准测试
# ============================================================

def run_benchmark(rounds: int = 3) -> dict:
    """运行全部基准测试

    Args:
        rounds: 每题运行轮数

    Returns:
        完整基准测试报告 dict
    """
    print(colorize("\n" + "=" * 70, Color.BOLD + Color.MAGENTA))
    print(colorize("  DeepRAG 性能基准测试 (L4)", Color.BOLD + Color.MAGENTA))
    print(colorize(f"  题目数: {len(BENCHMARK_QUESTIONS)}  |  每题轮数: {rounds}", Color.GRAY))
    print(colorize(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Color.GRAY))
    print(colorize("=" * 70, Color.BOLD + Color.MAGENTA))

    # 检查 RAG 可用性
    rag_available = _try_import_rag() is not None
    if rag_available:
        print(colorize("  RAG 管道: 可用 (真实模式)", Color.GREEN))
    else:
        print(colorize("  RAG 管道: 不可用 (模拟模式)", Color.YELLOW))

    results = []
    start_time = time.time()

    for i, q in enumerate(BENCHMARK_QUESTIONS):
        print()
        print(colorize(f"  [{i+1}/{len(BENCHMARK_QUESTIONS)}] ID={q['id']} [{q['category']}] {q['question'][:40]}",
                       Color.CYAN))

        result = run_single_benchmark(q, rounds=rounds)
        results.append(result)

        # 打印单题结果
        p = result["percentiles"]
        h = result["hit_at_k"]
        sim_tag = colorize(" [模拟]", Color.YELLOW) if result["simulation"] else ""
        print(f"    -> P50={p['p50']:.3f}s  P90={p['p90']:.3f}s  Hit@K={h['hit_at_k']:.2%}  "
              f"Tokens={result['tokens_out_avg']}{sim_tag}")

    total_time = time.time() - start_time

    # 汇总统计
    all_latencies = []
    all_hit_rates = []
    total_errors = 0
    all_simulation = True

    for r in results:
        all_latencies.extend(r["latencies"])
        all_hit_rates.append(r["hit_at_k"]["hit_at_k"])
        total_errors += len(r["errors"])
        if not r["simulation"]:
            all_simulation = False

    overall_percentiles = calculate_percentiles(all_latencies)
    avg_hit_rate = round(statistics.mean(all_hit_rates), 4) if all_hit_rates else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "questions": len(BENCHMARK_QUESTIONS),
            "rounds_per_question": rounds,
            "total_runs": len(BENCHMARK_QUESTIONS) * rounds,
            "simulation_mode": all_simulation,
        },
        "summary": {
            "total_time": round(total_time, 2),
            "latency": overall_percentiles,
            "avg_hit_at_k": avg_hit_rate,
            "total_errors": total_errors,
            "error_rate": round(total_errors / (len(BENCHMARK_QUESTIONS) * rounds), 4),
        },
        "per_question": results,
    }

    return report


# ============================================================
# 报告输出
# ============================================================

def save_report(report: dict):
    """保存 JSON 报告到 tests/reports/"""
    reports_dir = Path(__file__).parent.parent / "tests" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"benchmark_{ts}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(colorize(f"\n  JSON 报告已保存: {report_path}", Color.CYAN))
    return report_path


def print_console_summary(report: dict):
    """控制台打印汇总表"""
    summary = report["summary"]
    latency = summary["latency"]

    print()
    print(colorize("=" * 70, Color.BOLD + Color.BLUE))
    print(colorize("  基准测试汇总", Color.BOLD + Color.BLUE))
    print(colorize("=" * 70, Color.BOLD + Color.BLUE))
    print()

    # 延迟统计
    print(colorize("  响应时间统计", Color.BOLD + Color.CYAN))
    print(colorize("  " + "-" * 50, Color.GRAY))
    p50_str = colorize("{:.3f}s".format(latency["p50"]), Color.GREEN)
    p90_str = colorize("{:.3f}s".format(latency["p90"]), Color.YELLOW)
    p99_str = colorize("{:.3f}s".format(latency["p99"]), Color.RED)
    print("    P50  (中位数):  {}".format(p50_str))
    print("    P90  (90线):    {}".format(p90_str))
    print("    P99  (99线):    {}".format(p99_str))
    print("    Mean (平均):    {:.3f}s".format(latency["mean"]))
    print("    Min  (最小):    {:.3f}s".format(latency["min"]))
    print("    Max  (最大):    {:.3f}s".format(latency["max"]))
    print()

    # 召回率
    print(colorize("  召回率统计", Color.BOLD + Color.CYAN))
    print(colorize("  " + "-" * 50, Color.GRAY))
    hit_rate = summary["avg_hit_at_k"]
    hit_color = Color.GREEN if hit_rate >= 0.7 else Color.YELLOW if hit_rate >= 0.4 else Color.RED
    hit_str = colorize("{:.2%}".format(hit_rate), hit_color)
    print("    平均 Hit@K:     {}".format(hit_str))
    print()

    # 错误率
    print(colorize("  错误统计", Color.BOLD + Color.CYAN))
    print(colorize("  " + "-" * 50, Color.GRAY))
    err_color = Color.GREEN if summary["total_errors"] == 0 else Color.RED
    print(f"    总错误数:       {colorize(str(summary['total_errors']), err_color)}")
    print(f"    错误率:         {summary['error_rate']:.2%}")
    print()

    # 模拟模式提示
    if report["config"]["simulation_mode"]:
        print(colorize("  [注意] 本次测试在模拟模式下运行，数据不代表真实性能", Color.YELLOW))
        print()

    # 按类别统计
    print(colorize("  按类别统计", Color.BOLD + Color.CYAN))
    print(colorize("  " + "-" * 50, Color.GRAY))
    print(f"    {'类别':<14} {'P50':<10} {'P90':<10} {'Hit@K':<10} {'错误'}")
    print(colorize("    " + "-" * 46, Color.GRAY))

    categories = {}
    for r in report["per_question"]:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"latencies": [], "hits": [], "errors": 0}
        categories[cat]["latencies"].extend(r["latencies"])
        categories[cat]["hits"].append(r["hit_at_k"]["hit_at_k"])
        categories[cat]["errors"] += len(r["errors"])

    for cat in sorted(categories.keys()):
        data = categories[cat]
        cat_pct = calculate_percentiles(data["latencies"])
        cat_hit = statistics.mean(data["hits"]) if data["hits"] else 0
        cat_err = data["errors"]
        err_str = colorize(str(cat_err), Color.RED) if cat_err > 0 else colorize("0", Color.GREEN)
        print(f"    {cat:<14} {cat_pct['p50']:<10.3f} {cat_pct['p90']:<10.3f} {cat_hit:<10.2%} {err_str}")

    print()
    print(colorize(f"  总耗时: {summary['total_time']:.1f}s", Color.GRAY))
    print(colorize(f"  总运行次数: {report['config']['total_runs']}", Color.GRAY))
    print(colorize("=" * 70, Color.GRAY))


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepRAG 性能基准测试 — L4 层级",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/benchmark_test.py              # 默认每题3轮
  python scripts/benchmark_test.py --rounds 5   # 每题运行5轮
  python scripts/benchmark_test.py --rounds 1   # 快速模式
        """,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="每题运行轮数 (默认: 3)",
    )
    args = parser.parse_args()

    if args.rounds < 1:
        print(colorize("错误: --rounds 必须 >= 1", Color.RED))
        sys.exit(1)

    report = run_benchmark(rounds=args.rounds)
    print_console_summary(report)
    save_report(report)

    # 返回码：有错误则返回1
    has_errors = report["summary"]["total_errors"] > 0
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
