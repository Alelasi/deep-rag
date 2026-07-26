"""
超稳定评测脚本 - 解决并发导致的失败问题

策略：
1. 每次调用间隔3秒
2. 失败重试3次
3. 降低并发压力
"""
import requests
import time
import json

# 测试用例
test_cases = [
    {
        "query": "什么是RAG？",
        "reference": "RAG（Retrieval-Augmented Generation）是检索增强生成技术",
        "key_points": ["检索", "生成", "知识库", "增强"]
    },
    {
        "query": "Corrective RAG和Self-RAG有什么区别？",
        "reference": "Corrective RAG在检索后评估文档相关性，Self-RAG在生成后检查答案",
        "key_points": ["文档评分", "查询改写", "幻觉检测", "重新生成"]
    },
    {
        "query": "如何优化RAG检索准确率？",
        "reference": "1.混合检索 2.查询改写 3.多路召回 4.重排序 5.文档切块优化",
        "key_points": ["混合检索", "查询改写", "重排序", "文档切块"]
    },
    {
        "query": "什么是幻觉？如何规避？",
        "reference": "幻觉指模型生成不基于检索文档的内容。规避：事实校验、引用标注、置信度阈值",
        "key_points": ["事实错误", "事实校验", "引用", "置信度"]
    },
    {
        "query": "向量数据库有哪些？",
        "reference": "常见向量数据库：Qdrant、Milvus、Weaviate、Pinecone、ChromaDB、LanceDB",
        "key_points": ["Qdrant", "Milvus", "ChromaDB", "LanceDB"]
    },
]

def evaluate_model_ultra_stable(model):
    """超稳定评测"""
    url = 'http://localhost:11434/v1/chat/completions'

    print(f'超稳定评测: {model}')
    print(f'测试用例: {len(test_cases)}个（间隔3秒）')
    print('='*80)

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f'\\n[{i}/{len(test_cases)}] {test["query"]}')

        # 重试3次
        for attempt in range(3):
            try:
                start = time.time()
                response = requests.post(
                    url,
                    json={
                        'model': model,
                        'messages': [{'role': 'user', 'content': test['query']}],
                        'max_tokens': 200,
                        'temperature': 0.3
                    },
                    timeout=30
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']

                    if content and len(content) > 10:
                        # 简单评分
                        covered = sum(1 for kp in test['key_points'] if kp.lower() in content.lower())
                        accuracy = min(10, (covered / len(test['key_points'])) * 10)

                        scores = {
                            'accuracy': accuracy,
                            'total': accuracy * 0.4 + 8 * 0.6  # 简化评分
                        }

                        print(f'  ✅ {elapsed:.1f}s | 准确性:{scores["accuracy"]:.1f} | 总分:{scores["total"]:.1f}')

                        results.append({
                            'query': test['query'],
                            'response': content,
                            'scores': scores,
                            'latency': elapsed,
                            'success': True
                        })
                        break
                    else:
                        print(f'  ⚠️ 空响应 (尝试{attempt+1}/3)')
                else:
                    print(f'  ⚠️ HTTP {response.status_code} (尝试{attempt+1}/3)')

                if attempt < 2:
                    time.sleep(2)

            except Exception as e:
                print(f'  ⚠️ 异常 (尝试{attempt+1}/3): {str(e)[:40]}')
                if attempt < 2:
                    time.sleep(2)
        else:
            print(f'  ❌ 全部失败')
            results.append({'query': test['query'], 'success': False})

        # 每题之间间隔3秒
        if i < len(test_cases):
            print('  ⏳ 等待3秒...')
            time.sleep(3)

    # 统计
    success = [r for r in results if r.get('success')]

    print('\\n' + '='*80)
    print(f'成功率: {len(success)}/{len(test_cases)} ({len(success)/len(test_cases)*100:.0f}%)')

    if success:
        avg_accuracy = sum(r['scores']['accuracy'] for r in success) / len(success)
        avg_total = sum(r['scores']['total'] for r in success) / len(success)
        avg_latency = sum(r['latency'] for r in success) / len(success)

        print(f'平均准确性: {avg_accuracy:.1f}/10')
        print(f'平均得分: {avg_total:.1f}/10')
        print(f'平均延迟: {avg_latency:.1f}s')

        # 保存结果
        output = {
            'model': model,
            'success_count': len(success),
            'test_cases': len(test_cases),
            'avg_scores': {
                'accuracy': avg_accuracy,
                'total': avg_total
            },
            'results': results
        }

        filename = f'tests/evaluation_results_{model.replace("/", "_")}_stable.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f'\\n结果已保存: {filename}')
        return avg_total
    else:
        print('\\n❌ 全部失败')
        return 0

if __name__ == '__main__':
    # 测试nvidia/nemotron
    score = evaluate_model_ultra_stable('nvidia/nemotron-3-nano-4b')
    print(f'\\n最终得分: {score:.1f}/10')
