"""
RAG检索性能基准测试
测试延迟、吞吐量、准确率
"""
import sys
sys.path.insert(0, '.')

import time
from src.retrieval.unified_retriever import UnifiedRetriever


def benchmark_latency(retriever, queries, top_k=5):
    """测试延迟"""
    print("🔍 延迟测试")
    print("-" * 50)

    latencies = []

    for query in queries:
        start = time.time()
        result = retriever.search(query, top_k=top_k, mode="simple")
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"平均延迟: {avg_latency:.1f} ms")
    print(f"最小延迟: {min_latency:.1f} ms")
    print(f"最大延迟: {max_latency:.1f} ms")
    print()

    return avg_latency


def benchmark_throughput(retriever, queries, duration=10):
    """测试吞吐量"""
    print("⚡ 吞吐量测试")
    print("-" * 50)

    count = 0
    start = time.time()

    while time.time() - start < duration:
        for query in queries:
            retriever.search(query, top_k=3, mode="simple")
            count += 1

            if time.time() - start >= duration:
                break

    elapsed = time.time() - start
    throughput = count / elapsed

    print(f"总查询数: {count}")
    print(f"总耗时: {elapsed:.1f} s")
    print(f"吞吐量: {throughput:.1f} queries/s")
    print()

    return throughput


def benchmark_modes(retriever, queries):
    """测试不同模式"""
    print("🎯 模式对比测试")
    print("-" * 50)

    modes = ['simple', 'smart', 'expanded']

    for mode in modes:
        latencies = []

        for query in queries:
            start = time.time()
            retriever.search(query, top_k=3, mode=mode)
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)

        avg = sum(latencies) / len(latencies)
        print(f"{mode:10s}: {avg:6.1f} ms")

    print()


def main():
    print("=" * 70)
    print("🚀 RAG 检索性能基准测试")
    print("=" * 70)
    print()

    # 初始化
    print("📦 初始化检索器...")
    retriever = UnifiedRetriever(
        collection_name='full_docs',
        device='cuda'
    )

    stats = retriever.get_stats()
    print(f"✅ 知识库: {stats['total_docs']} 个文档")
    print()

    # 测试查询
    test_queries = [
        "GPU 加速",
        "内存优化",
        "向量检索",
        "RAG 系统",
        "性能提升",
    ]

    print("=" * 70)
    print("📊 基准测试")
    print("=" * 70)
    print()

    # 延迟测试
    avg_latency = benchmark_latency(retriever, test_queries)

    # 吞吐量测试
    throughput = benchmark_throughput(retriever, test_queries, duration=5)

    # 模式对比
    benchmark_modes(retriever, test_queries)

    # 总结
    print("=" * 70)
    print("📈 性能总结")
    print("=" * 70)
    print(f"知识库大小: {stats['total_docs']} 个文档")
    print(f"平均延迟: {avg_latency:.1f} ms")
    print(f"吞吐量: {throughput:.1f} queries/s")
    print()

    # 性能评级
    if avg_latency < 100:
        print("✅ 延迟: 优秀 (<100ms)")
    elif avg_latency < 200:
        print("✅ 延迟: 良好 (100-200ms)")
    else:
        print("⚠️ 延迟: 需优化 (>200ms)")

    if throughput > 10:
        print("✅ 吞吐量: 优秀 (>10 q/s)")
    elif throughput > 5:
        print("✅ 吞吐量: 良好 (5-10 q/s)")
    else:
        print("⚠️ 吞吐量: 需优化 (<5 q/s)")


if __name__ == "__main__":
    main()
