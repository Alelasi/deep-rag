"""Query分析Agent — 判断问题类型+查询改写"""
from src.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json

SYSTEM_PROMPT = """你是一个查询分析专家。分析用户问题并优化检索查询。

输出严格JSON：
{
  "question_type": "factual/reasoning/comparison/open_ended",
  "rewritten_query": "优化后的检索查询（去掉口语化表达，提取关键词）",
  "search_queries": ["子查询1", "子查询2"]
}

question_type判断标准：
- factual: 有明确答案的事实问题（"X是什么"、"Y的值是多少"）
- reasoning: 需要推理的问题（"为什么"、"怎么实现"）
- comparison: 对比类问题（"A和B的区别"、"哪个更好"）
- open_ended: 开放性问题（"你怎么看"、"有什么建议"）

search_queries: 将复杂问题拆分为1-3个独立可检索的子查询
"""


def analyze_query(question: str) -> dict:
    """分析用户查询，改写优化"""
    llm = get_llm(temperature=0)
    if llm is None:
        # 降级到离线模式
        return analyze_query_offline(question)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"分析这个问题：{question}"),
    ]
    response = llm.invoke(messages)

    try:
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content
        data = json.loads(json_str.strip())
    except (json.JSONDecodeError, IndexError):
        data = {
            "question_type": "factual",
            "rewritten_query": question,
            "search_queries": [question],
        }

    return {
        "question_type": data.get("question_type", "factual"),
        "rewritten_query": data.get("rewritten_query", question),
        "search_queries": data.get("search_queries", [question]),
    }


def analyze_query_offline(question: str) -> dict:
    """离线版查询分析（不调LLM，用规则）"""
    import re
    q = question.strip()

    # 判断类型
    if any(kw in q for kw in ["为什么", "怎么", "如何", "原因", "why", "how"]):
        qtype = "reasoning"
    elif any(kw in q for kw in ["区别", "对比", "和", "vs", "比较", "哪个"]):
        qtype = "comparison"
    elif any(kw in q for kw in ["建议", "看法", "你觉得", "推荐"]):
        qtype = "open_ended"
    else:
        qtype = "factual"

    # 简单查询改写：去掉口语词
    rewritten = re.sub(r"(请问|请|帮我|我想知道|能不能告诉我)", "", q).strip()

    return {
        "question_type": qtype,
        "rewritten_query": rewritten or q,
        "search_queries": [rewritten or q],
    }
