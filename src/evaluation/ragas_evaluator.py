"""离线启发式 RAG 指标（RAGAS 风格四指标，非官方 ragas 包）

指标名与官方 RAGAS 对齐，便于对照：
1. Answer Relevancy（答案相关性）
2. Context Precision（上下文精确度）
3. Context Recall（上下文召回率）
4. Faithfulness（忠实度）

说明：
- 本实现为无 API Key 的启发式打分，**不是** `pip install ragas` 的官方实现
- 中文场景使用 jieba / 字符 bigram，避免英文 split() 导致 Relevancy 恒为 0
"""

from typing import List, Dict, Optional, Set
import json
import re
from pathlib import Path


def _tokenize_zh(text: str) -> Set[str]:
    """中英混合分词：优先 jieba，失败则字符 bigram + 英文词"""
    if not text:
        return set()
    text = text.lower().strip()
    tokens: Set[str] = set()
    try:
        import jieba
        tokens.update(t.strip() for t in jieba.cut(text) if t and t.strip() and t.strip() not in "，。！？、；：""''（）【】《》 \t\n")
    except Exception:
        pass
    # 英文词
    tokens.update(re.findall(r"[a-z0-9_]+", text))
    # 中文 bigram 兜底（提升短问句重叠）
    hans = re.findall(r"[一-鿿]+", text)
    for chunk in hans:
        if len(chunk) == 1:
            tokens.add(chunk)
        else:
            for i in range(len(chunk) - 1):
                tokens.add(chunk[i : i + 2])
            tokens.add(chunk)
    # 过滤单字符噪音（保留中文单字）
    return {t for t in tokens if t and (len(t) > 1 or "一" <= t <= "鿿")}


class RAGASEvaluator:
    """离线启发式评测器（RAGAS 风格，非官方 ragas 包）"""

    def __init__(self):
        """初始化评测器"""
        self.results = []

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        context: List[str]
    ) -> float:
        """
        答案相关性评估（中英启发式）

        评分标准：
        - 1.0: 完全回答问题，无冗余信息
        - 0.7-0.9: 回答了问题，但有少量冗余
        - 0.4-0.6: 部分回答，有较多无关内容
        - 0.0-0.3: 基本没回答问题
        """
        if not answer or not question:
            return 0.0

        # 明确拒答：低相关但非 0（允许 no_knowledge 场景）
        refuse_words = [
            "不知道", "无法回答", "没有相关", "无法确定",
            "未找到可靠依据", "无法基于证据", "知识库与外部检索均未找到",
        ]
        if any(word in answer for word in refuse_words):
            return 0.25

        q_tokens = _tokenize_zh(question)
        a_tokens = _tokenize_zh(answer)
        if not q_tokens:
            return 0.0

        overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)

        # 答案过短且几乎无重叠
        if len(answer.strip()) < 8 and overlap < 0.1:
            return 0.0

        length_penalty = 1.0
        if len(answer) > 1000:
            length_penalty = 0.8
        elif len(answer) > 500:
            length_penalty = 0.9

        # 中文问题 token 少时，轻度抬升可解释重叠（避免永远 <0.1）
        if overlap > 0:
            score = min(1.0, overlap * 1.2) * length_penalty
        else:
            score = 0.0
        return round(min(1.0, score), 3)

    def evaluate_context_precision(
        self,
        question: str,
        contexts: List[Dict],
        ground_truth: Optional[str] = None
    ) -> float:
        """
        上下文精确度评估

        评估检索到的上下文中有多少是真正相关的

        评分标准：
        - 1.0: 所有检索到的文档都相关
        - 0.7-0.9: 大部分文档相关
        - 0.4-0.6: 一半文档相关
        - 0.0-0.3: 大部分文档不相关
        """
        if not contexts:
            return 0.0

        # 使用已有的grade字段
        relevant_count = sum(
            1 for ctx in contexts
            if ctx.get("grade") == "relevant"
        )

        precision = relevant_count / len(contexts)
        return round(precision, 3)

    def evaluate_context_recall(
        self,
        contexts: List[Dict],
        ground_truth: Optional[str] = None,
        answer: Optional[str] = None
    ) -> float:
        """
        上下文召回率评估

        评估检索到的上下文是否包含了回答问题所需的所有信息

        评分标准：
        - 1.0: 包含所有必要信息
        - 0.7-0.9: 包含大部分必要信息
        - 0.4-0.6: 包含部分必要信息
        - 0.0-0.3: 缺少关键信息
        """
        if not contexts:
            return 0.0

        # 如果有ground_truth，检查contexts是否包含
        if ground_truth:
            gt_words = _tokenize_zh(ground_truth)
            context_text = " ".join(ctx.get("content", "") for ctx in contexts)
            context_words = _tokenize_zh(context_text)
            overlap = len(gt_words & context_words) / max(len(gt_words), 1)
            return round(overlap, 3)

        # 如果有 answer，检查答案 token 有多少能在上下文中找到
        if answer:
            answer_words = _tokenize_zh(answer)
            context_text = " ".join(ctx.get("content", "") for ctx in contexts)
            context_words = _tokenize_zh(context_text)
            supported = len(answer_words & context_words) / max(len(answer_words), 1)
            return round(supported, 3)

        # 默认：基于relevant文档的比例
        relevant_count = sum(
            1 for ctx in contexts
            if ctx.get("grade") == "relevant"
        )

        # 如果有相关文档，假设召回率较高
        if relevant_count > 0:
            return round(min(1.0, relevant_count / 3), 3)  # 假设需要3个相关文档

        return 0.0

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[Dict],
        hallucination_score: Optional[float] = None
    ) -> float:
        """
        忠实度评估

        评估答案是否忠实于检索到的上下文，没有幻觉

        评分标准：
        - 1.0: 完全基于上下文，无幻觉
        - 0.7-0.9: 基本忠实，有轻微推断
        - 0.4-0.6: 有明显的无根据推断
        - 0.0-0.3: 大量幻觉内容
        """
        if not answer or not contexts:
            return 0.0

        # 如果已有hallucination_score，直接使用
        if hallucination_score is not None:
            return round(1.0 - hallucination_score, 3)

        # 否则，简单评估：答案中的词有多少能在上下文中找到
        answer_words = set(answer.lower().split())
        context_text = " ".join(ctx.get("content", "") for ctx in contexts)
        context_words = set(context_text.lower().split())

        # 过滤停用词
        stop_words = {"的", "是", "在", "了", "和", "与", "或", "等", "有", "为", "以", "及"}
        answer_words = answer_words - stop_words

        if not answer_words:
            return 0.5

        supported = len(answer_words & context_words) / len(answer_words)
        return round(supported, 3)

    def evaluate_single_query(
        self,
        question: str,
        answer: str,
        contexts: List[Dict],
        ground_truth: Optional[str] = None,
        hallucination_score: Optional[float] = None
    ) -> Dict:
        """
        评估单个查询的完整指标

        Args:
            question: 用户问题
            answer: 生成的答案
            contexts: 检索到的上下文（包含grade字段）
            ground_truth: 标准答案（可选）
            hallucination_score: 幻觉评分（可选）

        Returns:
            包含4个RAGAS指标的字典
        """
        metrics = {
            "question": question,
            "answer_relevancy": self.evaluate_answer_relevancy(question, answer, contexts),
            "context_precision": self.evaluate_context_precision(question, contexts, ground_truth),
            "context_recall": self.evaluate_context_recall(contexts, ground_truth, answer),
            "faithfulness": self.evaluate_faithfulness(answer, contexts, hallucination_score),
        }

        # 计算综合得分（4个指标平均）
        metrics["ragas_score"] = round(
            (metrics["answer_relevancy"] +
             metrics["context_precision"] +
             metrics["context_recall"] +
             metrics["faithfulness"]) / 4,
            3
        )

        self.results.append(metrics)
        return metrics

    def evaluate_batch(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        批量评估多个测试用例

        Args:
            test_cases: 测试用例列表，每个包含：
                - question: 问题
                - answer: 答案
                - contexts: 上下文
                - ground_truth: 标准答案（可选）
                - hallucination_score: 幻觉评分（可选）

        Returns:
            汇总统计结果
        """
        results = []
        for case in test_cases:
            result = self.evaluate_single_query(
                question=case["question"],
                answer=case["answer"],
                contexts=case["contexts"],
                ground_truth=case.get("ground_truth"),
                hallucination_score=case.get("hallucination_score")
            )
            results.append(result)

        # 计算平均指标
        avg_metrics = {
            "answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / len(results), 3),
            "context_precision": round(sum(r["context_precision"] for r in results) / len(results), 3),
            "context_recall": round(sum(r["context_recall"] for r in results) / len(results), 3),
            "faithfulness": round(sum(r["faithfulness"] for r in results) / len(results), 3),
            "ragas_score": round(sum(r["ragas_score"] for r in results) / len(results), 3),
        }

        return {
            "individual_results": results,
            "average_metrics": avg_metrics,
            "total_cases": len(results)
        }

    def save_results(self, output_path: str):
        """保存评测结果到JSON文件"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"✅ 评测结果已保存到: {output_path}")

    def print_report(self, metrics: Dict):
        """打印评测报告"""
        print("\n" + "="*60)
        print("📊 RAGAS 评测报告")
        print("="*60)

        if "individual_results" in metrics:
            # 批量评测结果
            print(f"\n总测试用例数: {metrics['total_cases']}")
            print("\n平均指标:")
            avg = metrics["average_metrics"]
            print(f"  • Answer Relevancy (答案相关性):    {avg['answer_relevancy']:.3f}")
            print(f"  • Context Precision (上下文精确度):  {avg['context_precision']:.3f}")
            print(f"  • Context Recall (上下文召回率):     {avg['context_recall']:.3f}")
            print(f"  • Faithfulness (忠实度):            {avg['faithfulness']:.3f}")
            print(f"  • RAGAS Score (综合得分):           {avg['ragas_score']:.3f}")

            # 显示每个用例的得分
            print("\n各用例详情:")
            for i, result in enumerate(metrics["individual_results"], 1):
                print(f"\n  [{i}] {result['question'][:50]}...")
                print(f"      RAGAS Score: {result['ragas_score']:.3f}")
        else:
            # 单个查询结果
            print(f"\n问题: {metrics['question']}")
            print(f"\n指标:")
            print(f"  • Answer Relevancy (答案相关性):    {metrics['answer_relevancy']:.3f}")
            print(f"  • Context Precision (上下文精确度):  {metrics['context_precision']:.3f}")
            print(f"  • Context Recall (上下文召回率):     {metrics['context_recall']:.3f}")
            print(f"  • Faithfulness (忠实度):            {metrics['faithfulness']:.3f}")
            print(f"  • RAGAS Score (综合得分):           {metrics['ragas_score']:.3f}")

        print("\n" + "="*60 + "\n")


def evaluate_from_state(state: Dict) -> Dict:
    """
    从RAGState直接评估

    Args:
        state: RAGState字典

    Returns:
        RAGAS评测结果
    """
    evaluator = RAGASEvaluator()

    return evaluator.evaluate_single_query(
        question=state.get("question", ""),
        answer=state.get("answer", ""),
        contexts=state.get("graded_docs", []),
        hallucination_score=state.get("hallucination_score")
    )
