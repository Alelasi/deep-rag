"""
自适应检索器测试脚本
演示四大方案的切换和使用
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.indexer import Indexer
from src.retrieval.adaptive_retriever import AdaptiveRetriever
import time


def demo_strategy_switching():
    """演示策略切换"""
    print("=" * 80)
    print("🚀 自适应检索器 - 四大方案切换演示")
    print("=" * 80)
    print()

    # 1. 初始化
    print("📦 初始化索引器...")
    indexer = Indexer("demo")

    # 索引一些示例文档
    sample_docs = [
        {"content": "深度学习是机器学习的一个分支，使用多层神经网络进行学习。", "source": "demo", "page": 1},
        {"content": "RAG（检索增强生成）结合了检索和生成两种技术。", "source": "demo", "page": 2},
        {"content": "HNSW 是一种高效的近似最近邻搜索算法。", "source": "demo", "page": 3},
        {"content": "向量数据库用于存储和检索高维向量。", "source": "demo", "page": 4},
        {"content": "Agent 是能够自主决策和执行任务的智能体。", "source": "demo", "page": 5},
    ]

    indexer.index_texts(sample_docs)

    print(f"✅ 已索引 {len(sample_docs)} 个文档")
    print()

    # 2. 创建自适应检索器
    print("🔧 创建自适应检索器...")
    retriever = AdaptiveRetriever(indexer, collection_name="demo")
    print("✅ 检索器已就绪")
    print()

    # 测试查询
    query = "什么是深度学习？"
    print(f"🔍 查询：{query}")
    print()

    # 3. 测试四大方案
    strategies = [
        ("accuracy", "方案 A：极致准确率（98%+）"),
        ("speed", "方案 B：极致速度（<5ms）"),
        ("balanced", "方案 C：平衡型（推荐）⭐"),
        ("ultimate", "方案 D：终极方案（2026最新）🆕"),
    ]

    results_summary = []

    for mode, description in strategies:
        print("-" * 80)
        print(f"📊 {description}")
        print("-" * 80)

        # 计时
        start_time = time.time()

        try:
            results = retriever.retrieve(query, mode=mode, top_k=3)
            elapsed = (time.time() - start_time) * 1000  # 转换为毫秒

            print(f"⏱️  延迟：{elapsed:.2f}ms")
            print(f"📄 返回文档数：{len(results)}")

            if results:
                print(f"🏆 Top-1 文档：{results[0].get('content', '')[:60]}...")

            results_summary.append({
                "mode": mode,
                "description": description,
                "latency_ms": elapsed,
                "count": len(results)
            })

        except Exception as e:
            print(f"❌ 错误：{e}")
            results_summary.append({
                "mode": mode,
                "description": description,
                "latency_ms": -1,
                "count": 0
            })

        print()

    # 4. 测试自动选择策略
    print("-" * 80)
    print("🤖 自动选择策略测试")
    print("-" * 80)

    test_contexts = [
        {"domain": "finance", "qps": 1000},
        {"domain": "general", "qps": 15000},
        {"domain": "general", "qps": 5000, "cache_hit_rate": 0.6},
        {"domain": "general", "qps": 5000},
    ]

    for ctx in test_contexts:
        start_time = time.time()
        results = retriever.retrieve(query, mode="auto", top_k=3, context=ctx)
        elapsed = (time.time() - start_time) * 1000

        print(f"📋 上下文：{ctx}")
        print(f"⏱️  延迟：{elapsed:.2f}ms")
        print(f"📄 返回文档数：{len(results)}")
        print()

    # 5. 性能对比总结
    print("=" * 80)
    print("📊 性能对比总结")
    print("=" * 80)
    print()
    print(f"{'方案':<40} {'延迟(ms)':<15} {'文档数':<10}")
    print("-" * 80)

    for r in results_summary:
        if r["latency_ms"] >= 0:
            print(f"{r['description']:<40} {r['latency_ms']:<15.2f} {r['count']:<10}")
        else:
            print(f"{r['description']:<40} {'ERROR':<15} {r['count']:<10}")

    print()

    # 6. 使用建议
    print("=" * 80)
    print("💡 使用建议")
    print("=" * 80)
    print()
    print("1️⃣  金融/医疗/法律 → mode='accuracy'（准确率优先）")
    print("2️⃣  搜索引擎/推荐系统 → mode='speed'（速度优先）")
    print("3️⃣  大多数场景 → mode='balanced'（平衡型，推荐）⭐")
    print("4️⃣  追求极致 → mode='ultimate'（2026最新技术）🆕")
    print("5️⃣  不确定 → mode='auto'（自动选择）🤖")
    print()

    # 7. 代码示例
    print("=" * 80)
    print("💻 代码示例")
    print("=" * 80)
    print()
    print("```python")
    print("from src.retrieval.indexer import Indexer")
    print("from src.retrieval.adaptive_retriever import AdaptiveRetriever")
    print()
    print("# 1. 创建检索器")
    print("indexer = Indexer('my_collection')")
    print("retriever = AdaptiveRetriever(indexer)")
    print()
    print("# 2. 使用不同策略")
    print("query = '什么是深度学习？'")
    print()
    print("# 方案 A：极致准确率")
    print("results = retriever.retrieve(query, mode='accuracy', top_k=10)")
    print()
    print("# 方案 B：极致速度")
    print("results = retriever.retrieve(query, mode='speed', top_k=10)")
    print()
    print("# 方案 C：平衡型（推荐）")
    print("results = retriever.retrieve(query, mode='balanced', top_k=10)")
    print()
    print("# 方案 D：终极方案")
    print("results = retriever.retrieve(query, mode='ultimate', top_k=10)")
    print()
    print("# 自动选择")
    print("context = {'domain': 'finance', 'qps': 5000}")
    print("results = retriever.retrieve(query, mode='auto', context=context)")
    print("```")
    print()

    print("=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    demo_strategy_switching()
