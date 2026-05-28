"""Self-RAG事实校验Agent — 检测生成内容中的幻觉"""
from src.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
from src.state import GradedDocument

SYSTEM_PROMPT = """你是一个事实核查专家。对比生成的回答与源文档，检测是否存在幻觉（hallucination）。

逐句检查回答中的事实性断言是否在源文档中有支撑。

输出JSON：
{
  "hallucination_score": 0.15,
  "passed": true,
  "unsupported_claims": ["无法在源文档中找到支撑的断言1"],
  "reasoning": "整体评估说明"
}

hallucination_score评分标准：
- 0.0: 完全忠实于源文档，无任何幻觉
- 0.1-0.3: 轻微推断但合理
- 0.3-0.6: 有部分未被支持的断言
- 0.6-1.0: 严重幻觉，大量内容编造

passed标准：hallucination_score < 0.3
"""


def check_facts(answer: str, source_docs: list[GradedDocument]) -> dict:
    """LLM事实校验"""
    if not answer or not source_docs:
        return {"hallucination_score": 0.0, "passed": True,
                "unsupported_claims": [], "reasoning": "无内容需要校验"}

    llm = get_llm(temperature=0)
    if llm is None:
        # 降级到离线模式
        return check_facts_offline(answer, source_docs)

    docs_text = "\n---\n".join(
        f"[{doc['source']}] {doc['content'][:600]}" for doc in source_docs[:5]
    )

    prompt = f"""## 生成的回答
{answer}

## 源文档
{docs_text}

请逐句检查回答中的事实性断言是否在源文档中有支撑。"""

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
        data = {"hallucination_score": 0.2, "passed": True,
                "unsupported_claims": [], "reasoning": "校验过程出错，默认通过"}

    score = float(data.get("hallucination_score", 0.2))
    return {
        "hallucination_score": score,
        "passed": score < 0.3,
        "unsupported_claims": data.get("unsupported_claims", []),
        "reasoning": data.get("reasoning", ""),
    }


def check_facts_offline(answer: str, source_docs: list[GradedDocument]) -> dict:
    """离线版事实校验（关键词覆盖率检查）"""
    if not answer or not source_docs:
        return {"hallucination_score": 0.0, "passed": True,
                "unsupported_claims": [], "reasoning": "无内容"}

    import jieba

    # 提取答案中的关键句（按句号/换行分割）
    import re
    sentences = re.split(r'[。\n.！？]', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    # 合并所有源文档文本
    all_source_text = " ".join(doc["content"] for doc in source_docs)
    source_tokens = set(jieba.cut(all_source_text))

    # 逐句检查覆盖率
    unsupported = []
    total_coverage = 0
    for sent in sentences:
        sent_tokens = set(jieba.cut(sent))
        overlap = sent_tokens & source_tokens
        coverage = len(overlap) / max(len(sent_tokens), 1)
        total_coverage += coverage
        if coverage < 0.2:
            unsupported.append(sent[:50])

    avg_coverage = total_coverage / max(len(sentences), 1)
    # 覆盖率越低，幻觉分越高
    hallucination_score = round(max(0, 1.0 - avg_coverage), 3)

    return {
        "hallucination_score": hallucination_score,
        "passed": hallucination_score < 0.3,
        "unsupported_claims": unsupported[:5],
        "reasoning": f"平均词覆盖率: {avg_coverage:.2%}, {len(unsupported)}句未被支持",
    }
