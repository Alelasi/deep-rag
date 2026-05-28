"""多源冲突解决Agent — 检测和标注文档间的矛盾"""
from src.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
from src.state import GradedDocument, ConflictInfo

SYSTEM_PROMPT = """你是一个信息冲突分析专家。检查多个文档之间是否存在事实矛盾。

如果存在矛盾，输出JSON：
{
  "has_conflict": true,
  "conflicts": [
    {
      "topic": "冲突主题",
      "positions": [
        {"source": "来源A", "claim": "A的说法", "evidence": "原文片段", "confidence": 0.8},
        {"source": "来源B", "claim": "B的说法", "evidence": "原文片段", "confidence": 0.6}
      ],
      "resolution": "基于证据强度，建议采纳A的说法，因为..."
    }
  ]
}

如果无矛盾：
{"has_conflict": false, "conflicts": []}
"""


def resolve_conflicts(question: str, docs: list[GradedDocument]) -> list[ConflictInfo]:
    """检测和解决多源冲突（LLM版）"""
    if len(docs) < 2:
        return []

    llm = get_llm(temperature=0)
    if llm is None:
        # 降级到离线模式
        return resolve_conflicts_offline(question, docs)

    docs_text = "\n---\n".join(
        f"[来源: {doc['source']}, 第{doc['page']}块]\n{doc['content'][:500]}"
        for doc in docs[:6]
    )

    prompt = f"""问题：{question}

以下多个文档片段可能包含矛盾信息，请检查：

{docs_text}

是否存在文档间的事实矛盾？"""

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
        return []

    if not data.get("has_conflict"):
        return []

    conflicts = []
    for c in data.get("conflicts", []):
        conflicts.append(ConflictInfo(
            topic=c.get("topic", ""),
            positions=c.get("positions", []),
            resolution=c.get("resolution", ""),
        ))
    return conflicts


def resolve_conflicts_offline(question: str, docs: list[GradedDocument]) -> list[ConflictInfo]:
    """离线版冲突检测（简单：检查不同来源对同一关键词的不同描述）"""
    # 简化实现：如果多个来源有同一个关键词但数值不同，标记冲突
    if len(docs) < 2:
        return []

    import re
    # 提取每个文档中的数值断言（如 "准确率95%"）
    number_claims = {}  # {source: [(keyword, value)]}
    for doc in docs:
        numbers = re.findall(r'([一-鿿]+)\s*[:：]?\s*(\d+\.?\d*)\s*%?', doc["content"])
        for keyword, value in numbers:
            if keyword not in number_claims:
                number_claims[keyword] = []
            number_claims[keyword].append({
                "source": doc["source"],
                "value": value,
                "context": doc["content"][:100],
            })

    # 找有冲突的（同一关键词不同数值）
    conflicts = []
    for keyword, claims in number_claims.items():
        values = set(c["value"] for c in claims)
        sources = set(c["source"] for c in claims)
        if len(values) > 1 and len(sources) > 1:
            positions = [
                {"source": c["source"], "claim": f"{keyword}: {c['value']}",
                 "evidence": c["context"], "confidence": 0.5}
                for c in claims
            ]
            conflicts.append(ConflictInfo(
                topic=keyword,
                positions=positions,
                resolution=f"多个来源对'{keyword}'给出不同数值，需人工确认",
            ))

    return conflicts[:3]
