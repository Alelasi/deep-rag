"""
本地LLM效果评测脚本

评测维度：
1. 准确性（回答是否正确）
2. 完整性（是否回答全面）
3. 相关性（是否切题）
4. 专业性（术语使用是否准确）
5. 可读性（表达是否清晰）

使用方式：
python tests/evaluate_llm_quality.py --model gemma-3-1b-it --queries 10
"""
import requests
import json
import time

# 测试查询+标准答案
test_cases = [
    {
        "query": "什么是RAG？",
        "reference": "RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的技术，通过检索外部知识库来增强大语言模型的生成能力。",
        "key_points": ["检索", "生成", "知识库", "增强"]
    },
    {
        "query": "Corrective RAG和Self-RAG有什么区别？",
        "reference": "Corrective RAG在检索后评估文档相关性，不相关则改写查询重试。Self-RAG在生成后检查答案是否基于检索文档，有幻觉则重新生成。",
        "key_points": ["文档评分", "查询改写", "幻觉检测", "重新生成"]
    },
    {
        "query": "如何优化RAG检索准确率？",
        "reference": "1.混合检索(BM25+向量) 2.查询改写 3.多路召回 4.重排序 5.文档切块优化",
        "key_points": ["混合检索", "查询改写", "重排序", "文档切块"]
    },
    {
        "query": "什么是幻觉？如何规避？",
        "reference": "幻觉指模型生成的内容不基于检索文档或事实错误。规避方法：1.事实校验 2.引用标注 3.置信度阈值 4.检索质量控制",
        "key_points": ["事实错误", "事实校验", "引用", "置信度"]
    },
    {
        "query": "向量数据库有哪些？",
        "reference": "常见向量数据库：Qdrant、Milvus、Weaviate、Pinecone、ChromaDB、LanceDB、Faiss等",
        "key_points": ["Qdrant", "Milvus", "ChromaDB", "LanceDB"]
    },
    {
        "query": "混合检索的原理是什么？",
        "reference": "混合检索结合稀疏检索(BM25关键词匹配)和密集检索(向量相似度)，通过RRF等融合策略综合两者结果，提高召回率和准确率。",
        "key_points": ["BM25", "向量检索", "RRF", "融合"]
    },
    {
        "query": "如何实现查询改写？",
        "reference": "1.使用LLM扩展查询 2.提取关键词 3.生成同义查询 4.HyDE(假设文档) 5.多查询生成",
        "key_points": ["LLM", "关键词", "同义词", "HyDE"]
    },
    {
        "query": "Agent的工作原理是什么？",
        "reference": "Agent基于ReAct模式：Reasoning(推理分析)->Action(选择工具)->Observation(观察结果)->循环直到完成任务。",
        "key_points": ["ReAct", "推理", "工具", "循环"]
    },
    {
        "query": "如何配置LangChain的API Key？",
        "reference": "1.创建.env文件 2.添加OPENAI_API_KEY=sk-xxx 3.使用python-dotenv加载 4.或设置环境变量export OPENAI_API_KEY=xxx",
        "key_points": [".env", "环境变量", "python-dotenv", "OPENAI_API_KEY"]
    },
    {
        "query": "LangGraph和LangChain有什么区别？",
        "reference": "LangChain是链式调用框架，LangGraph是状态图框架。LangGraph支持循环、条件分支、并行执行，更适合复杂Agent。",
        "key_points": ["链式", "状态图", "循环", "分支"]
    }
]


def call_llm(query, model="gemma-3-1b-it"):
    """调用LM Studio"""
    url = 'http://localhost:11434/v1/chat/completions'

    try:
        response = requests.post(
            url,
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': query}],
                'max_tokens': 300,
                'temperature': 0.3
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def evaluate_response(query, response, reference, key_points):
    """评估回答质量（1-10分）"""
    if not response:
        return {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "professionalism": 0,
            "readability": 0,
            "total": 0
        }

    response_lower = response.lower()

    # 1. 准确性（关键点覆盖率）
    covered = sum(1 for kp in key_points if kp.lower() in response_lower)
    accuracy = min(10, (covered / len(key_points)) * 10)

    # 2. 完整性（长度合理性，50-500字符）
    length = len(response)
    if 50 <= length <= 500:
        completeness = 10
    elif length < 50:
        completeness = max(1, (length / 50) * 10)
    else:
        completeness = max(6, 10 - (length - 500) / 100)

    # 3. 相关性（包含查询关键词）
    query_words = set(query.replace("？", "").replace("什么", "").replace("如何", "").split())
    relevance_count = sum(1 for w in query_words if len(w) > 1 and w in response)
    relevance = min(10, (relevance_count / max(1, len(query_words))) * 15)

    # 4. 专业性（包含技术术语）
    tech_terms = ["RAG", "LLM", "向量", "检索", "生成", "模型", "数据库", "API", "Agent", "框架"]
    tech_count = sum(1 for term in tech_terms if term in response)
    professionalism = min(10, tech_count * 2)

    # 5. 可读性（结构化，有数字、列表等）
    has_structure = any(x in response for x in ["1.", "2.", "：", "、", "\n"])
    readability = 8 if has_structure else 5

    # 总分（加权平均）
    total = (
        accuracy * 0.35 +        # 准确性最重要
        completeness * 0.20 +
        relevance * 0.20 +
        professionalism * 0.15 +
        readability * 0.10
    )

    return {
        "accuracy": round(accuracy, 1),
        "completeness": round(completeness, 1),
        "relevance": round(relevance, 1),
        "professionalism": round(professionalism, 1),
        "readability": round(readability, 1),
        "total": round(total, 1)
    }


def run_evaluation(model="gemma-3-1b-it"):
    """运行完整评测"""
    print(f"开始评测: {model}")
    print(f"测试用例: {len(test_cases)}个")
    print("="*80)

    results = []
    total_time = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test['query']}")

        # 调用LLM
        start = time.time()
        response = call_llm(test['query'], model)
        elapsed = time.time() - start
        total_time += elapsed

        if response:
            # 评估
            scores = evaluate_response(
                test['query'],
                response,
                test['reference'],
                test['key_points']
            )

            print(f"  ⏱️  延迟: {elapsed:.2f}s")
            print(f"  📊 评分: 准确{scores['accuracy']} 完整{scores['completeness']} "
                  f"相关{scores['relevance']} 专业{scores['professionalism']} "
                  f"可读{scores['readability']} | 总分{scores['total']}")

            results.append({
                'query': test['query'],
                'response': response,
                'scores': scores,
                'latency': elapsed,
                'success': True
            })
        else:
            print(f"  ❌ 调用失败")
            results.append({
                'query': test['query'],
                'response': None,
                'scores': None,
                'latency': elapsed,
                'success': False
            })

    # 汇总统计
    print("\n" + "="*80)
    print("评测总结")
    print("="*80)

    success_results = [r for r in results if r['success']]

    if success_results:
        # 平均分数
        avg_scores = {
            'accuracy': sum(r['scores']['accuracy'] for r in success_results) / len(success_results),
            'completeness': sum(r['scores']['completeness'] for r in success_results) / len(success_results),
            'relevance': sum(r['scores']['relevance'] for r in success_results) / len(success_results),
            'professionalism': sum(r['scores']['professionalism'] for r in success_results) / len(success_results),
            'readability': sum(r['scores']['readability'] for r in success_results) / len(success_results),
            'total': sum(r['scores']['total'] for r in success_results) / len(success_results)
        }

        print(f"\n成功率: {len(success_results)}/{len(test_cases)} ({len(success_results)/len(test_cases)*100:.1f}%)")
        print(f"平均延迟: {total_time/len(test_cases):.2f}s")
        print(f"\n平均评分（满分10分）:")
        print(f"  准确性: {avg_scores['accuracy']:.1f}/10")
        print(f"  完整性: {avg_scores['completeness']:.1f}/10")
        print(f"  相关性: {avg_scores['relevance']:.1f}/10")
        print(f"  专业性: {avg_scores['professionalism']:.1f}/10")
        print(f"  可读性: {avg_scores['readability']:.1f}/10")
        print(f"  ─────────────────")
        print(f"  综合得分: {avg_scores['total']:.1f}/10")

        # 等级评定
        total_score = avg_scores['total']
        if total_score >= 9:
            grade = "A+ 优秀"
        elif total_score >= 8:
            grade = "A  良好"
        elif total_score >= 7:
            grade = "B+ 中上"
        elif total_score >= 6:
            grade = "B  中等"
        elif total_score >= 5:
            grade = "C  及格"
        else:
            grade = "D  不及格"

        print(f"\n等级评定: {grade}")

        # 保存结果
        with open(f'tests/evaluation_results_{model}.json', 'w', encoding='utf-8') as f:
            json.dump({
                'model': model,
                'test_cases': len(test_cases),
                'success_count': len(success_results),
                'avg_scores': avg_scores,
                'grade': grade,
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: tests/evaluation_results_{model}.json")
    else:
        print("❌ 全部失败")


if __name__ == "__main__":
    run_evaluation("gemma-3-1b-it")
