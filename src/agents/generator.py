"""答案生成Agent — 带引用溯源的生成"""
from src.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from src.state import GradedDocument, Citation

SYSTEM_PROMPT = """你是一个知识库问答专家。根据提供的文档生成准确的回答。

关键规则：
1. 只基于提供的文档回答，不使用外部知识
2. 每个关键事实后标注引用 [来源:文件名, 第N块]
3. 如果文档信息不足以回答，明确说"根据现有知识库资料，无法完整回答此问题"
4. 如果多个文档说法矛盾，标注"注意：来源间存在分歧"

输出格式：
先回答问题（带引用标注），最后列出引用清单。
"""


def generate_answer(question: str, relevant_docs: list[GradedDocument]) -> dict:
    """基于相关文档生成带引用的回答"""
    if not relevant_docs:
        return {
            "answer": "根据现有知识库资料，未找到与此问题相关的信息。",
            "citations": [],
        }

    llm = get_llm(temperature=0.3)
    if llm is None:
        # 降级到离线模式
        return generate_answer_offline(question, relevant_docs)

    # 构造文档上下文
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        context_parts.append(
            f"[文档{i}] 来源: {doc['source']}, 第{doc['page']}块\n"
            f"相关度: {doc['relevance_score']:.0%}\n"
            f"内容:\n{doc['content']}\n"
        )
    context = "\n---\n".join(context_parts)

    prompt = f"""问题：{question}

参考文档（{len(relevant_docs)}篇）：
{context}

请根据以上文档回答问题，每个关键事实标注引用 [来源:文件名, 第N块]。"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    answer = response.content

    # 提取引用
    citations = []
    for doc in relevant_docs:
        if doc["source"] in answer or f"第{doc['page']}块" in answer:
            citations.append(Citation(
                text=doc["content"][:200],
                source=doc["source"],
                page=doc["page"],
            ))

    return {"answer": answer, "citations": citations}


def generate_answer_offline(question: str, relevant_docs: list[GradedDocument]) -> dict:
    """离线版生成（基于规则的摘要式回答）"""
    if not relevant_docs:
        return {"answer": "未找到相关信息。", "citations": []}

    # 按相关度排序取top3
    sorted_docs = sorted(relevant_docs, key=lambda d: -d["relevance_score"])[:3]

    # 提取关键信息（简单规则：取每个文档的前2句话）
    key_points = []
    citations = []

    for i, doc in enumerate(sorted_docs, 1):
        content = doc['content']
        # 简单分句（按句号、问号、感叹号分割）
        sentences = [s.strip() for s in content.replace('。', '.|').replace('？', '?|').replace('！', '!|').split('|') if s.strip()]
        # 取前2句作为关键点
        summary = '。'.join(sentences[:2])
        if summary and not summary.endswith('。'):
            summary += '。'

        key_points.append(f"{i}. {summary} [来源:{doc['source']}, 第{doc['page']}块]")
        citations.append(Citation(text=doc["content"][:200], source=doc["source"], page=doc["page"]))

    # 构造摘要式回答
    answer_parts = [
        f"根据知识库资料，关于「{question}」的回答如下：\n",
        "\n".join(key_points),
        f"\n\n以上信息来自 {len(sorted_docs)} 个相关文档片段。"
    ]

    return {"answer": "".join(answer_parts), "citations": citations}
