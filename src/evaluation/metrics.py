"""RAG评估指标 — 量化检索和生成质量"""


def evaluate_retrieval(graded_docs: list, total_relevant_in_kb: int = None) -> dict:
    """
    检索质量评估
    - precision: 检索到的文档中相关的比例
    - recall@k: 如果知道总相关数，计算召回率
    """
    if not graded_docs:
        return {"precision": 0, "recall": None, "total": 0}

    relevant = sum(1 for d in graded_docs if d.get("grade") == "relevant")
    total = len(graded_docs)
    precision = relevant / total if total > 0 else 0

    recall = None
    if total_relevant_in_kb:
        recall = relevant / total_relevant_in_kb

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3) if recall is not None else None,
        "relevant_count": relevant,
        "total_retrieved": total,
    }


def evaluate_generation(hallucination_score: float, citations: list,
                       answer: str, conflicts: list) -> dict:
    """
    生成质量评估
    - faithfulness: 1 - hallucination_score（忠实度）
    - citation_density: 引用数/答案长度
    - completeness: 基于答案长度的完成度估算
    """
    faithfulness = 1.0 - hallucination_score
    answer_len = len(answer) if answer else 0
    citation_density = len(citations) / max(answer_len / 200, 1)  # 每200字期望1个引用

    # 完成度估算：太短可能没回答完
    if answer_len > 500:
        completeness = 1.0
    elif answer_len > 200:
        completeness = 0.7
    elif answer_len > 50:
        completeness = 0.4
    else:
        completeness = 0.1

    return {
        "faithfulness": round(faithfulness, 3),
        "citation_density": round(min(1.0, citation_density), 3),
        "completeness": completeness,
        "has_conflicts": len(conflicts) > 0,
        "answer_length": answer_len,
    }


def evaluate_pipeline(state: dict) -> dict:
    """综合评估一次RAG查询的质量"""
    retrieval_eval = evaluate_retrieval(state.get("graded_docs", []))
    generation_eval = evaluate_generation(
        state.get("hallucination_score", 0),
        state.get("citations", []),
        state.get("answer", ""),
        state.get("conflicts", []),
    )

    # 综合得分 (0-100)
    overall = (
        retrieval_eval["precision"] * 30 +
        generation_eval["faithfulness"] * 40 +
        generation_eval["citation_density"] * 15 +
        generation_eval["completeness"] * 15
    )

    return {
        "retrieval": retrieval_eval,
        "generation": generation_eval,
        "overall_score": round(overall, 1),
        "retries": state.get("retry_count", 0),
        "steps": len(state.get("history", [])),
    }
