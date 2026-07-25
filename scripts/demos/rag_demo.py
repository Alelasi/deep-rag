"""
RAG知识库应用示例
展示完整的检索功能
"""
import sys
sys.path.insert(0, '.')

from src.retrieval.unified_retriever import UnifiedRetriever


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 RAG 知识库应用")
    print("=" * 70)
    print()

    # 初始化检索器
    print("📦 初始化检索器...")
    retriever = UnifiedRetriever(
        collection_name='full_docs',
        device='cuda',
        enable_query_optimization=True,
        enable_hallucination_detection=True,
        similarity_threshold=0.5
    )

    stats = retriever.get_stats()
    print(f"✅ 知识库已加载: {stats['total_docs']} 个文档")
    print()

    # 示例查询
    demo_queries = [
        "GPU 加速的性能提升是多少",
        "内存占用 5GB 的原因",
        "增量更新如何实现",
        "2026-05-27 完成了什么工作",
        "如何训练 GPT-5 模型",  # 应该被拒绝
    ]

    print("=" * 70)
    print("📚 示例查询")
    print("=" * 70)
    print()

    for i, query in enumerate(demo_queries, 1):
        print(f"查询 {i}: {query}")
        print("-" * 70)

        # 智能检索
        result = retriever.search(query, top_k=3, mode="smart")

        # 显示结果
        print(f"置信度: {result['confidence']}")

        if result.get('explanation'):
            print(f"说明: {result['explanation']}")

        if result['results']:
            print(f"\n找到 {len(result['results'])} 个相关文档:\n")

            for j, doc in enumerate(result['results'], 1):
                print(f"{j}. 相似度: {doc['similarity']:.2%}")
                print(f"   来源: {doc['source']}")
                print(f"   内容: {doc['content'][:150]}...")
                print()
        else:
            print("\n未找到相关文档")

        print()

    # 交互模式
    print("=" * 70)
    print("💬 交互模式（输入 'quit' 退出）")
    print("=" * 70)
    print()

    while True:
        try:
            query = input("请输入查询: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            if not query:
                continue

            # 检索
            result = retriever.search(query, top_k=3, mode="smart")

            # 显示结果
            print(f"\n置信度: {result['confidence']}")

            if result.get('explanation'):
                print(f"说明: {result['explanation']}")

            if result['results']:
                print(f"\n找到 {len(result['results'])} 个相关文档:\n")

                for j, doc in enumerate(result['results'], 1):
                    print(f"{j}. 相似度: {doc['similarity']:.2%}")
                    print(f"   来源: {doc['source']}")
                    print(f"   内容: {doc['content'][:100]}...")
                    print()
            else:
                print("\n未找到相关文档")

            print()

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
