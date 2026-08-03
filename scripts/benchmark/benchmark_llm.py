# 本地LLM基准测试脚本
# 创建时间：2026-06-08
# 用途：对比Claude vs Ollama vs 规则模式的真实性能
"""
本地LLM基准测试

使用方法：
python scripts/benchmark/benchmark_llm.py --backends claude,ollama,rule --queries 50
"""
import time
import os
from typing import List, Dict
import json


def test_llm_backend(backend: str, queries: List[str]) -> Dict:
    """测试单个LLM后端"""
    results = {
        'backend': backend,
        'latencies': [],
        'success_count': 0,
        'error_count': 0,
        'total_cost': 0.0
    }

    # 配置LLM
    os.environ['LLM_BACKEND'] = backend
    from src.config import get_llm
    llm = get_llm()

    for query in queries:
        try:
            start = time.time()
            response = llm.invoke(query)
            elapsed = time.time() - start

            results['latencies'].append(elapsed)
            results['success_count'] += 1

        except Exception as e:
            results['error_count'] += 1
            print(f"Error: {e}")

    # 计算统计指标
    if results['latencies']:
        results['latency_p50'] = sorted(results['latencies'])[len(results['latencies']) // 2]
        results['latency_p90'] = sorted(results['latencies'])[int(len(results['latencies']) * 0.9)]
        results['latency_mean'] = sum(results['latencies']) / len(results['latencies'])

    return results


def main():
    # 测试查询（50条）
    test_queries = [
        "如何配置LangChain的API Key？",
        "什么是RAG？",
        "向量数据库有哪些？",
        # ... 更多查询
    ]

    backends = ['claude', 'ollama', 'rule']

    results = {}
    for backend in backends:
        print(f"\n测试 {backend} 后端...")
        results[backend] = test_llm_backend(backend, test_queries)

    # 打印对比报告
    print("\n=== 性能对比报告 ===")
    for backend, result in results.items():
        print(f"\n{backend}:")
        print(f"  成功率: {result['success_count']}/{len(test_queries)}")
        print(f"  延迟P50: {result.get('latency_p50', 0):.2f}s")
        print(f"  延迟P90: {result.get('latency_p90', 0):.2f}s")
        print(f"  平均延迟: {result.get('latency_mean', 0):.2f}s")


if __name__ == "__main__":
    main()

# ===== 使用说明（文档） =====
# 1. 安装Ollama模型: ollama pull qwen2.5:7b
# 2. 运行测试: python scripts/benchmark/benchmark_llm.py
# 3. 获得真实性能数据后更新文档中的性能对比表
# 示例输出:
#   claude: 成功率 50/50, 延迟P50 1.23s, 延迟P90 2.15s, 平均延迟 1.45s
#   ollama: 成功率 48/50, 延迟P50 0.68s, 延迟P90 1.12s, 平均延迟 0.82s
#   rule:   成功率 45/50, 延迟P50 0.001s, 延迟P90 0.002s, 平均延迟 0.001s
