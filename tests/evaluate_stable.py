"""
稳定版评测脚本 - 单线程，带重试机制

解决之前批量测试失败的问题
"""
import requests
import time
import json

def evaluate_model_stable(model, test_cases, output_file):
    """稳定评测（单线程+重试）"""
    url = 'http://localhost:11434/v1/chat/completions'

    print(f'稳定评测: {model}')
    print(f'测试用例: {len(test_cases)}个')
    print('='*80)

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f'\\n[{i}/{len(test_cases)}] {test["query"]}')

        # 重试机制（最多3次）
        for attempt in range(3):
            try:
                start = time.time()
                response = requests.post(
                    url,
                    json={
                        'model': model,
                        'messages': [{'role': 'user', 'content': test['query']}],
                        'max_tokens': 300,
                        'temperature': 0.3
                    },
                    timeout=30
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']

                    # 简单评分
                    from tests.evaluate_llm_quality import evaluate_response
                    scores = evaluate_response(
                        test['query'],
                        content,
                        test['reference'],
                        test['key_points']
                    )

                    print(f'  ✅ {elapsed:.1f}s | 得分:{scores["total"]:.1f}')

                    results.append({
                        'query': test['query'],
                        'response': content,
                        'scores': scores,
                        'latency': elapsed,
                        'success': True
                    })
                    break
                else:
                    print(f'  ⚠️ 失败 (尝试{attempt+1}/3)')
                    if attempt < 2:
                        time.sleep(2)
            except Exception as e:
                print(f'  ⚠️ 异常 (尝试{attempt+1}/3): {str(e)[:40]}')
                if attempt < 2:
                    time.sleep(2)
        else:
            print(f'  ❌ 全部失败')
            results.append({
                'query': test['query'],
                'success': False
            })

    # 统计
    success = [r for r in results if r.get('success')]
    if success:
        avg_scores = {
            'total': sum(r['scores']['total'] for r in success) / len(success),
            'accuracy': sum(r["scores"]["accuracy"] for r in success) / len(success),
        }

        print('\\n' + '='*80)
        print(f'成功率: {len(success)}/{len(test_cases)} ({len(success)/len(test_cases)*100:.0f}%)')
        print(f'综合得分: {avg_scores["total"]:.1f}/10')

        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model,
                'success_count': len(success),
                'test_cases': len(test_cases),
                'avg_scores': avg_scores,
                'results': results
            }, f, indent=2, ensure_ascii=False)

        return avg_scores['total']
    else:
        print('\\n❌ 全部失败')
        return 0

if __name__ == '__main__':
    # 导入测试用例
    import sys
    sys.path.insert(0, 'tests')
    from evaluate_llm_quality import test_cases

    # 评测nemotron
    score = evaluate_model_stable(
        'nvidia/nemotron-3-nano-4b',
        test_cases,
        'tests/evaluation_results_nvidia-nemotron-stable.json'
    )

    print(f'\\n最终得分: {score:.1f}/10')
