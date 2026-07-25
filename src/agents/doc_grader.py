"""Corrective RAG文档评分Agent — 批量评估相关性（v2.8：合并LLM调用, v2.9.2: Constrained Decoding增强）"""
from src.config import get_llm, get_temperature
from langchain_core.messages import HumanMessage, SystemMessage
import json
import logging
from src.state import Document, GradedDocument
from src.llm.rate_limiter import retry_with_backoff

log = logging.getLogger("deeprag.grader")


def _get_doc_field(doc: dict, field: str, default=None):
    """兼容获取文档字段 — 同时支持 content/text, doc_id/id 等不同键名"""
    aliases = {
        "content": ["content", "text", "document", "page_content"],
        "doc_id": ["doc_id", "id", "document_id"],
        "source": ["source", "file", "filename", "file_path"],
        "page": ["page", "chunk", "chunk_id", "page_number"],
    }
    for key in aliases.get(field, [field]):
        if key in doc:
            return doc[key]
    return default

SYSTEM_PROMPT = """你是一个文档相关性评估专家。判断检索到的文档片段是否与用户问题相关。

对每个文档输出JSON数组，每个元素格式：
{
  "index": 1,
  "grade": "relevant/ambiguous/irrelevant",
  "relevance_score": 0.85,
  "reasoning": "一句话解释"
}

评分标准：
- relevant (>=0.7): 文档直接包含回答问题所需的信息
- ambiguous (0.3-0.7): 部分相关但不足以回答，或需要结合其他信息
- irrelevant (<0.3): 与问题无关

输出格式：```json
[{"index": 1, "grade": "relevant", "relevance_score": 0.9, "reasoning": "..."}, ...]
```
"""


def grade_documents(question: str, documents: list[Document]) -> list[GradedDocument]:
    """批量文档评分（v2.8：5篇文档合并为1次LLM调用，避免429）

    v2.7：每篇文档单独调用LLM，5篇=5次API调用，容易429
    v2.8：所有文档合并到一个prompt，1次API调用完成全部评分
    """
    if not documents:
        return []

    llm = get_llm(temperature=get_temperature("doc_grading"))
    if llm is None:
        return grade_documents_offline(question, documents)

    # v2.8: 预提取所有文档信息
    doc_infos = []
    for i, doc in enumerate(documents):
        doc_infos.append({
            "index": i + 1,
            "content": _get_doc_field(doc, "content", "")[:400],  # v2.8.1: 减少到400字
            "source": _get_doc_field(doc, "source", "unknown"),
            "page": _get_doc_field(doc, "page", 0),
            "doc_id": _get_doc_field(doc, "doc_id", f"doc_{i}"),
        })

    # v2.8: 构造批量评分prompt
    doc_sections = []
    for info in doc_infos:
        doc_sections.append(
            f"[文档{info['index']}] 来源: {info['source']}, 第{info['page']}块\n"
            f"{info['content']}"
        )
    all_docs_text = "\n---\n".join(doc_sections)

    prompt = f"""问题：{question}

以下共{len(doc_infos)}篇文档，请逐篇评估与问题的相关性：

{all_docs_text}

请输出JSON数组，为每篇文档评分。"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # v2.8.2: 单次LLM调用评分所有文档（think模式由全局开关控制，评分始终禁用思考）
    @retry_with_backoff(max_retries=3, base_delay=1.5)
    def _invoke_batch():
        from src.config import LLM_BACKEND, LLM_MODEL
        if LLM_BACKEND == "ollama":
            from src.llm.ollama_helper import ollama_chat
            ollama_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            content, _ = ollama_chat(
                ollama_messages, LLM_MODEL,
                temperature=get_temperature("doc_grading"), num_predict=400, think=False,  # 评分始终禁用思考
            )
            return content
        else:
            response = llm.invoke(messages)
            return response.content

    try:
        content = _invoke_batch()
        # v2.9.1: 使用safe_parse_json替代手动split解析
        from src.llm.constrained_decoder import safe_parse_json
        grades_data = safe_parse_json(content)

        if grades_data is None:
            raise json.JSONDecodeError("safe_parse_json返回None", content[:100], 0)

        # 如果返回的是单个对象而非数组，转为数组
        if isinstance(grades_data, dict):
            grades_data = [grades_data]

        log.info(f"[Grader v2.8] Batch graded {len(documents)} docs in 1 LLM call")
    except Exception as e:
        log.warning(f"[Grader v2.8] Batch grading failed: {e}, falling back to offline")
        return grade_documents_offline(question, documents)

    # 构造结果列表
    graded = []
    # 按 index 匹配结果
    grade_map = {}
    for item in grades_data:
        if isinstance(item, dict) and "index" in item:
            grade_map[int(item["index"])] = item

    for i, info in enumerate(doc_infos):
        item = grade_map.get(i + 1, {})
        if not item:
            # 尝试按顺序匹配
            if i < len(grades_data) and isinstance(grades_data[i], dict):
                item = grades_data[i]
            else:
                item = {}

        graded.append(GradedDocument(
            doc_id=info["doc_id"],
            content=info["content"],
            source=info["source"],
            page=info["page"],
            grade=item.get("grade", "ambiguous"),
            relevance_score=float(item.get("relevance_score", 0.5)),
            reasoning=item.get("reasoning", ""),
        ))

    return graded


def grade_documents_offline(question: str, documents: list[Document]) -> list[GradedDocument]:
    """离线版文档评分（关键词匹配+简单规则，不调LLM）

    返回：GradedDocument列表，每个文档包含评分和置信度判断
    """
    import jieba

    question_tokens = set(jieba.cut(question))
    graded = []

    for doc in documents:
        doc_content = _get_doc_field(doc, "content", "")
        doc_source = _get_doc_field(doc, "source", "unknown")
        doc_page = _get_doc_field(doc, "page", 0)
        doc_id = _get_doc_field(doc, "doc_id", f"{doc_source}_{doc_page}")

        doc_tokens = set(jieba.cut(doc_content))
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
            doc_id=doc_id,
            content=doc_content,
            source=doc_source,
            page=doc_page,
            grade=grade,
            relevance_score=round(score, 3),
            reasoning=f"keyword overlap: {len(overlap)}/{len(question_tokens)} tokens",
        ))

    return graded


def check_confidence(graded_docs: list[GradedDocument]) -> bool:
    """置信度判断 — 判断是否需要人工审核

    规则：如果所有文档的最高相关度评分 < 0.5，标记需要人工审核
    这意味着没有文档被明确判定为相关，检索结果质量可能不足。

    参数：
        graded_docs: 评分后的文档列表

    返回：
        True = 需要人工审核（置信度不足）
        False = 置信度足够，继续正常Pipeline
    """
    if not graded_docs:
        return True  # 无文档时也需要人工介入

    # 取所有文档中的最高相关度评分
    max_score = max(doc["relevance_score"] for doc in graded_docs)
    # 如果最高评分低于0.5，说明没有高质量匹配，需要人工审核
    return max_score < 0.5
