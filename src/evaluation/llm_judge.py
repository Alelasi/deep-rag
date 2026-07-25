"""LLM-as-Judge评测 — v2.9.1新增

用LLM评估LLM输出质量，覆盖5个维度：
1. relevancy: 答案是否回答了问题（切题度）
2. faithfulness: 答案是否忠实于上下文文档（忠实度）
3. completeness: 答案是否完整覆盖参考答案要点（完整度）
4. conciseness: 答案是否简洁无冗余（简洁度）
5. citation_accuracy: 引用是否准确对应文档（引用准确度）

裁判模型使用 temperature=0 保证确定性评估。
"""
import json
import logging
import statistics
from typing import Optional
from pathlib import Path

log = logging.getLogger("deeprag")


class LLMJudge:
    """LLM-as-Judge 裁判

    用法：
        judge = LLMJudge()
        result = judge.evaluate(
            question="INTJ的主导功能是什么？",
            answer="INTJ的主导功能是Ni[1]...",
            reference="INTJ的主导功能是内倾直觉（Ni）...",
            context="[文档1] INTJ的 cognitive functions..."
        )
        print(result["overall"])  # 0-10分
        print(result["scores"])   # 5维度评分
    """

    # 评估维度
    DIMENSIONS = [
        "relevancy",
        "faithfulness",
        "completeness",
        "conciseness",
        "citation_accuracy",
    ]

    def __init__(self, judge_model: str = "glm-4-flash"):
        """初始化裁判

        Args:
            judge_model: 裁判模型名（默认用最快模型）
        """
        self.judge_model = judge_model
        self._llm = None

    def _get_llm(self):
        """获取裁判LLM实例（temperature=0保证确定性）"""
        if self._llm is None:
            from src.config import get_llm
            # 裁判始终用 temperature=0（确定性评估）
            self._llm = get_llm(temperature=0)
        return self._llm

    def _score(self, prompt: str) -> float:
        """调用LLM评分（0-10分）"""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="你是严格的评分专家。只输出0-10的数字，不要输出其他内容。"),
            HumanMessage(content=prompt),
        ]

        try:
            llm = self._get_llm()
            if llm is None:
                return 5.0  # LLM不可用时默认中等分

            response = llm.invoke(messages)
            content = response.content.strip()

            # 提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            if numbers:
                score = float(numbers[0])
                return min(10.0, max(0.0, score))
            return 5.0
        except Exception as e:
            log.warning(f"[LLMJudge] 评分失败: {e}")
            return 5.0

    def _eval_relevancy(self, question: str, answer: str) -> float:
        """评估切题度：答案是否回答了问题"""
        prompt = f"""请评估以下回答是否切题。

问题：{question}
回答：{answer[:500]}

评分标准（0-10）：
- 10分：完美回答了问题的核心
- 7-9分：基本回答了问题，有少量偏离
- 4-6分：部分回答，有较多无关内容
- 0-3分：完全未回答问题

只输出数字。"""
        return self._score(prompt)

    def _eval_faithfulness(self, answer: str, context: str) -> float:
        """评估忠实度：答案是否忠实于上下文文档"""
        prompt = f"""请评估回答是否忠实于参考文档。

参考文档：
{context[:800]}

回答：
{answer[:500]}

评分标准（0-10）：
- 10分：完全基于文档，无任何编造
- 7-9分：基本忠实，有轻微合理推断
- 4-6分：有部分未支撑的断言
- 0-3分：大量编造内容

只输出数字。"""
        return self._score(prompt)

    def _eval_completeness(self, answer: str, reference: str) -> float:
        """评估完整度：答案是否覆盖参考答案要点"""
        if not reference:
            return 7.0  # 无参考答案时默认较高分

        prompt = f"""请评估实际回答是否覆盖了参考答案的要点。

参考答案：
{reference[:500]}

实际回答：
{answer[:500]}

评分标准（0-10）：
- 10分：完全覆盖参考答案所有要点
- 7-9分：覆盖大部分要点
- 4-6分：覆盖部分要点
- 0-3分：几乎未覆盖

只输出数字。"""
        return self._score(prompt)

    def _eval_conciseness(self, answer: str) -> float:
        """评估简洁度：答案是否简洁无冗余"""
        prompt = f"""请评估回答是否简洁无冗余。

回答：
{answer[:500]}

评分标准（0-10）：
- 10分：精炼准确，无任何冗余
- 7-9分：基本简洁，有少量重复
- 4-6分：有较多冗余内容
- 0-3分：极度冗长

只输出数字。"""
        return self._score(prompt)

    def _eval_citations(self, answer: str, context: str) -> float:
        """评估引用准确度：引用标记是否准确对应文档"""
        import re
        refs = re.findall(r'\[(\d+)\]', answer)
        if not refs:
            # 无引用标记
            return 3.0 if len(context) > 50 else 7.0

        prompt = f"""请评估回答中的引用标记是否准确。

文档：
{context[:600]}

回答：
{answer[:500]}

引用标记：{refs}

评分标准（0-10）：
- 10分：所有引用准确对应文档内容
- 7-9分：大部分引用准确
- 4-6分：部分引用不准确
- 0-3分：引用完全错误或编造

只输出数字。"""
        return self._score(prompt)

    def evaluate(
        self,
        question: str,
        answer: str,
        reference: str = "",
        context: str = "",
    ) -> dict:
        """5维度评估

        Args:
            question: 用户问题
            answer: LLM生成的答案
            reference: 参考答案（标准答案）
            context: 检索到的文档上下文

        Returns:
            {"scores": {...}, "overall": float}
        """
        scores = {
            "relevancy": self._eval_relevancy(question, answer),
            "faithfulness": self._eval_faithfulness(answer, context),
            "completeness": self._eval_completeness(answer, reference),
            "conciseness": self._eval_conciseness(answer),
            "citation_accuracy": self._eval_citations(answer, context),
        }

        # 四舍五入
        scores = {k: round(v, 1) for k, v in scores.items()}
        overall = round(sum(scores.values()) / len(scores), 1)

        return {
            "scores": scores,
            "overall": overall,
            "judge_model": self.judge_model,
        }

    def batch_evaluate(self, test_cases: list[dict]) -> dict:
        """批量评估

        Args:
            test_cases: 测试用例列表，每个用例需包含query和actual_answer，
                       可选expected_answer和context

        Returns:
            {"individual": [...], "avg_overall": float, "avg_by_dimension": {...}}
        """
        results = []

        for i, case in enumerate(test_cases, 1):
            log.info(f"[LLMJudge] 评估 {i}/{len(test_cases)}: {case.get('query', '')[:30]}...")

            result = self.evaluate(
                question=case.get("query", ""),
                answer=case.get("actual_answer", case.get("answer", "")),
                reference=case.get("expected_answer", ""),
                context=case.get("context", ""),
            )
            result["id"] = case.get("id", i)
            result["query"] = case.get("query", "")
            result["category"] = case.get("category", "")
            result["difficulty"] = case.get("difficulty", "")
            results.append(result)

        # 聚合统计
        if not results:
            return {"individual": [], "avg_overall": 0, "avg_by_dimension": {}}

        avg_overall = statistics.mean(r["overall"] for r in results)
        avg_by_dimension = {
            dim: round(statistics.mean(r["scores"][dim] for r in results), 1)
            for dim in self.DIMENSIONS
        }

        # 按类别统计
        by_category = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r["overall"])
        avg_by_category = {
            cat: round(statistics.mean(scores), 1)
            for cat, scores in by_category.items()
        }

        # 按难度统计
        by_difficulty = {}
        for r in results:
            diff = r.get("difficulty", "unknown")
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(r["overall"])
        avg_by_difficulty = {
            diff: round(statistics.mean(scores), 1)
            for diff, scores in by_difficulty.items()
        }

        return {
            "individual": results,
            "avg_overall": round(avg_overall, 1),
            "avg_by_dimension": avg_by_dimension,
            "avg_by_category": avg_by_category,
            "avg_by_difficulty": avg_by_difficulty,
            "total_evaluated": len(results),
        }


# === 用户反馈记录 ===

FEEDBACK_LOG_PATH = Path(__file__).parent.parent.parent / "evaluation" / "feedback_log.json"


def record_feedback(query: str, answer: str, rating: str, mode: str = "enhanced"):
    """记录用户反馈

    Args:
        query: 用户问题
        answer: 系统回答
        rating: "up" 或 "down"
        mode: 检索模式
    """
    import time

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query[:200],
        "answer_length": len(answer),
        "rating": rating,
        "mode": mode,
    }

    # 确保文件存在
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 读取现有日志
    logs = []
    if FEEDBACK_LOG_PATH.exists():
        try:
            with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            logs = []

    # 追加新条目
    logs.append(entry)

    # 写回文件
    try:
        with open(FEEDBACK_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        log.info(f"[Feedback] 已记录反馈: {rating} for '{query[:30]}...'")
    except IOError as e:
        log.warning(f"[Feedback] 写入失败: {e}")


def get_feedback_stats() -> dict:
    """获取用户反馈统计"""
    if not FEEDBACK_LOG_PATH.exists():
        return {"total": 0, "up": 0, "down": 0, "satisfaction_rate": 0}

    try:
        with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"total": 0, "up": 0, "down": 0, "satisfaction_rate": 0}

    total = len(logs)
    up = sum(1 for l in logs if l.get("rating") == "up")
    down = sum(1 for l in logs if l.get("rating") == "down")
    rate = round(up / total * 100, 1) if total > 0 else 0

    return {
        "total": total,
        "up": up,
        "down": down,
        "satisfaction_rate": rate,
    }
