"""RAGAS评测报告生成器 — 生成完整的评测报告"""
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.graph import query, get_indexer
from src.retrieval.indexer import Indexer
import json
from datetime import datetime

DOCS_DIR = str(Path(PROJECT_ROOT) / "data/sample_docs")


def generate_evaluation_report():
    """生成完整的RAGAS评测报告"""
    print("\n" + "="*60)
    print("📊 DeepRAG RAGAS评测报告生成器")
    print("="*60 + "\n")

    # 1. 索引文档
    print("步骤1: 索引文档...")
    indexer = Indexer("test_kb")
    indexer.clear()
    count = indexer.index_directory(DOCS_DIR)
    print(f"  ✅ 已索引 {count} 个文档块\n")

    # 2. 准备测试用例
    print("步骤2: 准备测试用例...")
    test_questions = [
        "INTJ的主导功能是什么？",
        "什么是恐贪指数？",
        "九型5号的核心恐惧是什么？",
        "INTJ和ENFP的关系如何？",
        "如何使用RSI指标判断市场情绪？",
    ]
    print(f"  ✅ 准备了 {len(test_questions)} 个测试问题\n")

    # 3. 运行查询并评估
    print("步骤3: 运行查询并评估...")
    evaluator = RAGASEvaluator()
    test_cases = []

    for i, question in enumerate(test_questions, 1):
        print(f"  [{i}/{len(test_questions)}] 查询: {question}")

        # 运行查询
        state = query(question, "test_kb")

        # 准备评测数据
        test_case = {
            "question": question,
            "answer": state.get("answer", ""),
            "contexts": state.get("graded_docs", []),
            "hallucination_score": state.get("hallucination_score", 0),
            "retry_count": state.get("retry_count", 0),
            "relevant_count": state.get("relevant_count", 0),
            "citations": state.get("citations", []),
        }
        test_cases.append(test_case)

    print(f"  ✅ 完成所有查询\n")

    # 4. 批量评估
    print("步骤4: 批量RAGAS评估...")
    result = evaluator.evaluate_batch(test_cases)
    print("  ✅ 评估完成\n")

    # 5. 生成报告
    print("步骤5: 生成评测报告...")

    # 打印报告
    evaluator.print_report(result)

    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(PROJECT_ROOT) / "evaluation_reports"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"ragas_report_{timestamp}.json"

    # 准备完整报告数据
    report_data = {
        "timestamp": timestamp,
        "test_info": {
            "total_questions": len(test_questions),
            "knowledge_base": "test_kb",
            "indexed_chunks": count,
        },
        "average_metrics": result["average_metrics"],
        "individual_results": result["individual_results"],
        "summary": {
            "best_score": max(r["ragas_score"] for r in result["individual_results"]),
            "worst_score": min(r["ragas_score"] for r in result["individual_results"]),
            "avg_retry_count": sum(tc["retry_count"] for tc in test_cases) / len(test_cases),
            "avg_relevant_docs": sum(tc["relevant_count"] for tc in test_cases) / len(test_cases),
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 详细报告已保存到: {output_file}\n")

    # 6. 生成Markdown报告
    md_file = output_dir / f"ragas_report_{timestamp}.md"
    generate_markdown_report(report_data, md_file)
    print(f"  ✅ Markdown报告已保存到: {md_file}\n")

    print("="*60)
    print("✅ 评测报告生成完成！")
    print("="*60 + "\n")

    return report_data


def generate_markdown_report(data: dict, output_file: Path):
    """生成Markdown格式的评测报告"""

    md_content = f"""# DeepRAG RAGAS评测报告

**生成时间**: {data['timestamp']}

---

## 📊 测试概况

- **测试问题数**: {data['test_info']['total_questions']}
- **知识库**: {data['test_info']['knowledge_base']}
- **索引文档块数**: {data['test_info']['indexed_chunks']}

---

## 🎯 平均指标

| 指标 | 得分 | 说明 |
|------|------|------|
| Answer Relevancy (答案相关性) | {data['average_metrics']['answer_relevancy']:.3f} | 答案是否直接回答问题 |
| Context Precision (上下文精确度) | {data['average_metrics']['context_precision']:.3f} | 检索文档的相关性 |
| Context Recall (上下文召回率) | {data['average_metrics']['context_recall']:.3f} | 是否包含所需信息 |
| Faithfulness (忠实度) | {data['average_metrics']['faithfulness']:.3f} | 答案是否忠实于上下文 |
| **RAGAS Score (综合得分)** | **{data['average_metrics']['ragas_score']:.3f}** | **4个指标的平均值** |

---

## 📈 性能统计

- **最高得分**: {data['summary']['best_score']:.3f}
- **最低得分**: {data['summary']['worst_score']:.3f}
- **平均重试次数**: {data['summary']['avg_retry_count']:.2f}
- **平均相关文档数**: {data['summary']['avg_relevant_docs']:.2f}

---

## 📝 各问题详情

"""

    for i, result in enumerate(data['individual_results'], 1):
        md_content += f"""
### {i}. {result['question']}

**RAGAS Score**: {result['ragas_score']:.3f}

| 指标 | 得分 |
|------|------|
| Answer Relevancy | {result['answer_relevancy']:.3f} |
| Context Precision | {result['context_precision']:.3f} |
| Context Recall | {result['context_recall']:.3f} |
| Faithfulness | {result['faithfulness']:.3f} |

---
"""

    md_content += """
## 🔍 评分标准

### Answer Relevancy (答案相关性)
- **1.0**: 完全回答问题，无冗余信息
- **0.7-0.9**: 回答了问题，但有少量冗余
- **0.4-0.6**: 部分回答，有较多无关内容
- **0.0-0.3**: 基本没回答问题

### Context Precision (上下文精确度)
- **1.0**: 所有检索到的文档都相关
- **0.7-0.9**: 大部分文档相关
- **0.4-0.6**: 一半文档相关
- **0.0-0.3**: 大部分文档不相关

### Context Recall (上下文召回率)
- **1.0**: 包含所有必要信息
- **0.7-0.9**: 包含大部分必要信息
- **0.4-0.6**: 包含部分必要信息
- **0.0-0.3**: 缺少关键信息

### Faithfulness (忠实度)
- **1.0**: 完全基于上下文，无幻觉
- **0.7-0.9**: 基本忠实，有轻微推断
- **0.4-0.6**: 有明显的无根据推断
- **0.0-0.3**: 大量幻觉内容

---

## 📚 关于RAGAS

RAGAS (Retrieval Augmented Generation Assessment) 是专门用于评估RAG系统的框架，通过4个核心指标全面评估检索和生成质量。

**项目地址**: https://github.com/explodinggradients/ragas

---

*本报告由 DeepRAG 自动生成*
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)


if __name__ == "__main__":
    generate_evaluation_report()
