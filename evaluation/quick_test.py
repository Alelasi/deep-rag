"""
快速测试评估系统 - 只运行前5个查询

用于验证评估系统是否正常工作
"""

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.agent_executor_v2 import AgentExecutorV2

def test_evaluation_system():
    """测试评估系统"""
    print("="*60)
    print("评估系统快速测试（前5个查询）")
    print("="*60)

    # 加载数据集
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 只测试前5个
    test_queries = dataset[:5]

    # 创建Agent
    print("\n初始化Agent...")
    agent = AgentExecutorV2(
        llm_base_url="http://localhost:11434/v1/chat/completions",
        model="gemma-3-4b-it",
        max_iterations=5,
        enable_planning=False,
        enable_memory=False,
        enable_logging=False
    )
    print("✓ Agent初始化成功")

    # 运行测试
    print(f"\n开始测试 {len(test_queries)} 个查询...\n")
    results = []

    for i, query_data in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] 查询: {query_data['query']}")

        # 运行查询
        start = time.time()
        result = agent.run(query_data['query'], verbose=False, use_memory=False)
        elapsed = time.time() - start

        # 提取答案
        if result["success"]:
            answer = result.get("answer", "无答案")
            status = "✓"
        else:
            answer = f"[错误] {result.get('error', '未知错误')}"
            status = "✗"

        print(f"  {status} 耗时: {elapsed:.2f}秒")
        print(f"  答案: {answer[:100]}{'...' if len(answer) > 100 else ''}\n")

        results.append({
            'id': query_data['id'],
            'query': query_data['query'],
            'answer': answer,
            'time': elapsed,
            'success': result["success"]
        })

    # 统计
    success_count = sum(1 for r in results if r['success'])
    avg_time = sum(r['time'] for r in results) / len(results)

    print("="*60)
    print("测试完成")
    print("="*60)
    print(f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"平均响应时间: {avg_time:.2f}秒")
    print("\n✓ 评估系统工作正常！")
    print("\n下一步:")
    print("  python evaluation/run_evaluation.py  # 运行完整评估（50个查询）")

if __name__ == "__main__":
    test_evaluation_system()
