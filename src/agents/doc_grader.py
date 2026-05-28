"""Corrective RAG文档评分Agent — 逐文档评估相关性"""
from src.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
from src.state import Document, GradedDocument

SYSTEM_PROMPT = """你是一个文档相关性评估专家。判断检索到的文档片段是否与用户问题相关。

对每个文档输出JSON：
{
  "grade": "relevant/ambiguous/irrelevant",
  "relevance_score": 0.85,
  "reasoning": "一句话解释"
}

评分标准：
- relevant (>=0.7): 文档直接包含回答问题所需的信息
- ambiguous (0.3-0.7): 部分相关但不足以回答，或需要结合其他信息
- irrelevant (<0.3): 与问题无关
"""


def grade_documents(question: str, documents: list[Document]) -> list[GradedDocument]:
    """逐文档评分（LLM版）"""
    if not documents:
        return []

    llm = get_llm(temperature=0)
    if llm is None:
        # 降级到离线模式
        return grade_documents_offline(question, documents)
    graded = []

    for doc in documents:
        prompt = f"""问题：{question}

文档内容（来源: {doc['source']}，第{doc['page']}块）：
---
{doc['content'][:1000]}
---

评估该文档与问题的相关性。"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = llm.invoke(messages)
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            data = json.loads(json_str.strip())
        except Exception:
            data = {"grade": "ambiguous", "relevance_score": 0.5, "reasoning": "评分失败"}

        graded.append(GradedDocument(
            doc_id=doc["doc_id"],
            content=doc["content"],
            source=doc["source"],
            page=doc["page"],
            grade=data.get("grade", "ambiguous"),
            relevance_score=float(data.get("relevance_score", 0.5)),
            reasoning=data.get("reasoning", ""),
        ))

    return graded


def grade_documents_offline(question: str, documents: list[Document]) -> list[GradedDocument]:
    """离线版文档评分（关键词匹配+简单规则，不调LLM）"""
    import jieba

    question_tokens = set(jieba.cut(question))
    graded = []

    for doc in documents:
        doc_tokens = set(jieba.cut(doc["content"]))
        # 计算词汇重叠度
        overlap = question_tokens & doc_tokens
        coverage = len(overlap) / max(len(question_tokens), 1)

        if coverage >= 0.3:
            grade = "relevant"
            score = min(1.0, 0.5 + coverage)
        elif coverage >= 0.1:
            grade = "ambiguous"
            score = 0.3 + coverage
        else:
            grade = "irrelevant"
            score = coverage

        graded.append(GradedDocument(
            doc_id=doc["doc_id"],
            content=doc["content"],
            source=doc["source"],
            page=doc["page"],
            grade=grade,
            relevance_score=round(score, 3),
            reasoning=f"keyword overlap: {len(overlap)}/{len(question_tokens)} tokens",
        ))

    return graded
