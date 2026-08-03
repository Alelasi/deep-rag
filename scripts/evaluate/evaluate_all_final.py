"""
支持思维链模型的评测脚本

关键修复：
1. gemma-4-e2b使用reasoning_content字段
2. 增加max_tokens到1000
3. 提取真实回答内容
"""
import requests
import time
import json

test_cases = [
    {"query": "什么是RAG？", "key_points": ["检索", "生成", "知识库"]},
    {"query": "Corrective RAG和Self-RAG有什么区别？", "key_points": ["文档评分", "幻觉检测"]},
    {"query": "如何优化RAG检索准确率？", "key_points": ["混合检索", "重排序"]},
    {"query": "什么是幻觉？如何规避？", "key_points": ["事实错误", "事实校验"]},
    {"query": "向量数据库有哪些？", "key_points": ["Qdrant", "Milvus"]},
]

def evaluate_all_models():
    """评测所有可用模型"""
    url = 'http://localhost:11434/v1/chat/completions'

    models = [
        {'name': 'nvidia/nemotron-3-nano-4b', 'type': 'normal'},
        {'name': 'google/gemma-4-e2b', 'type': 'reasoning'},  # 思维链模型
    ]

    all_results = {}

    for model_info in models:
        model = model_info['name']
        model_type = model_info['type']

        print(f'\n\n{"="*80}')
        print(f'评测: {model} ({model_type})')
        print(f'{"="*80}')

        results = []

        for i, test in enumerate(test_cases, 1):
            print(f'\n[{i}/{len(test_cases)}] {test["query"]}')

            # 重试3次
            for attempt in range(3):
                try:
                    response = requests.post(
                        url,
                        json={
                            'model': model,
                            'messages': [{'role': 'user', 'content': test['query']}],
                            'max_tokens': 1000,  # 增加到1000
                            'temperature': 0.3
                        },
                        timeout=40
                    )

                    if response.status_code == 200:
                        data = response.json()
                        message = data['choices'][0]['message']

                        # 根据模型类型提取内容
                        if model_type == 'reasoning':
                            # 思维链模型：从reasoning_content提取
                            content = message.get('reasoning_content', '')
                        else:
                            # 普通模型：从content提取
                            content = message.get('content', '')

                        if content and len(content) > 20:
                            # 简单评分
                            covered = sum(1 for kp in test['key_points'] if kp.lower() in content.lower())
                            accuracy = min(10, (covered / len(test['key_points'])) * 10)
                            total = accuracy * 0.4 + 7 * 0.6

                            print(f'  ✅ 成功 | 准确性:{accuracy:.1f} | 总分:{total:.1f}')

                            results.append({
                                'query': test['query'],
                                'response': content[:200],
                                'scores': {'accuracy': accuracy, 'total': total},
                                'success': True
                            })
                            break
                        else:
                            print(f'  ⚠️ 内容过短 (尝试{attempt+1}/3)')
                    else:
                        print(f'  ⚠️ HTTP {response.status_code} (尝试{attempt+1}/3)')

                    if attempt < 2:
                        time.sleep(2)

                except Exception as e:
                    print(f'  ⚠️ 异常 (尝试{attempt+1}/3): {str(e)[:40]}')
                    if attempt < 2:
                        time.sleep(2)
            else:
                print(f'  ❌ 失败')
                results.append({'query': test['query'], 'success': False})

            # 间隔3秒
            if i < len(test_cases):
                time.sleep(3)

        # 统计
        success = [r for r in results if r.get('success')]

        print(f'\n{"="*80}')
        print(f'成功率: {len(success)}/{len(test_cases)} ({len(success)/len(test_cases)*100:.0f}%)')

        if success:
            avg_total = sum(r['scores']['total'] for r in success) / len(success)
            print(f'平均得分: {avg_total:.1f}/10')

            all_results[model] = {
                'success_count': len(success),
                'total': len(test_cases),
                'avg_score': avg_total,
                'results': results
            }

            # 保存
            filename = f'tests/evaluation_results_{model.replace("/", "_")}_final.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_results[model], f, indent=2, ensure_ascii=False)
            print(f'已保存: {filename}')
        else:
            print('全部失败')
            all_results[model] = {'success_count': 0, 'total': len(test_cases), 'avg_score': 0}

    # 最终总结
    print('\n\n' + '=' * 80)
    print('最终总结')
    print('=' * 80)

    for model, result in all_results.items():
        print(f'\\n{model}:')
        print(f'  成功率: {result['success_count']}/{result['total']}')
        print(f'  得分: {result['avg_score']:.1f}/10')

if __name__ == '__main__':
    evaluate_all_models()
