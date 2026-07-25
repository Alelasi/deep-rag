"""Prompt压缩器 — v2.9.1新增

智能压缩文档上下文，替代粗暴截断。
- extractive（默认）：jieba分词 + 句子级关键词重叠排序，零成本
- llm（可选）：用小模型摘要，高质量但有延迟
"""
import re
import logging
from typing import Optional

log = logging.getLogger("deeprag")

# jieba 懒加载
_jieba = None

def _get_jieba():
    """获取jieba分词器（懒加载）"""
    global _jieba
    if _jieba is None:
        try:
            import jieba
            _jieba = jieba
        except ImportError:
            log.warning("[PromptCompressor] jieba未安装，降级为简单空格分词")
            _jieba = False
    return _jieba


def _tokenize(text: str) -> set:
    """分词，返回去停用词后的词集合"""
    jieba = _get_jieba()
    if jieba:
        words = jieba.cut(text)
        # 过滤单字和常见停用词
        stop_words = {"的", "是", "了", "在", "和", "与", "或", "也", "都", "但",
                      "而", "如", "为", "对", "从", "到", "被", "把", "让", "使",
                      "这", "那", "一", "个", "些", "么", "什么", "如何", "怎么"}
        return {w for w in words if len(w) > 1 and w not in stop_words}
    else:
        # 降级：简单空格分词
        return {w for w in text.split() if len(w) > 1}


class PromptCompressor:
    """Prompt上下文压缩器

    用法：
        compressor = PromptCompressor()
        compressed = compressor.compress(question, docs, max_tokens=400)
    """

    def __init__(self, strategy: str = "extractive"):
        self.strategy = strategy

    def compress(
        self,
        question: str,
        docs: list[dict],
        max_tokens: int = 400,
        strategy: Optional[str] = None,
    ) -> str:
        """压缩文档上下文

        Args:
            question: 用户问题（用于提取关键词）
            docs: 文档列表，每个doc有content/text字段
            max_tokens: 最大token数（约等于字符数）
            strategy: 压缩策略（覆盖默认）

        Returns:
            压缩后的上下文字符串
        """
        strat = strategy or self.strategy

        if not docs:
            return ""

        if strat == "extractive":
            return self._extractive_compress(question, docs, max_tokens)
        elif strat == "llm":
            return self._llm_compress(question, docs, max_tokens)
        else:
            return self._extractive_compress(question, docs, max_tokens)

    def _extractive_compress(
        self,
        question: str,
        docs: list[dict],
        max_tokens: int,
    ) -> str:
        """抽取式压缩（默认，零成本）

        1. jieba分词提取问题关键词
        2. 按句子分割文档
        3. 计算每个句子与问题的关键词重叠率
        4. 按重叠率排序，取top-N句子
        5. 去重（前20字相同则视为重复）
        """
        # 1. 提取问题关键词
        q_keywords = _tokenize(question)
        if not q_keywords:
            # 无法提取关键词，降级到截断
            return self._fallback_truncate(docs, max_tokens)

        # 2. 收集所有文档的句子
        sentences = []  # (句子文本, 重叠得分, 来源标记)
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", doc.get("text", ""))
            source = doc.get("source", f"文档{i}")
            page = doc.get("page", 0)

            # 按句号、换行、问号、感叹号分割
            doc_sents = re.split(r'[。\n！？.!?\r]', content)
            for sent in doc_sents:
                sent = sent.strip()
                if len(sent) < 5:
                    continue

                # 3. 计算关键词重叠率
                sent_tokens = _tokenize(sent)
                if not sent_tokens:
                    continue
                overlap = len(q_keywords & sent_tokens) / len(q_keywords)

                # 加入来源标记
                sentences.append((sent, overlap, source, page, i))

        if not sentences:
            return self._fallback_truncate(docs, max_tokens)

        # 4. 按重叠率排序（稳定排序，保持原文顺序）
        # 先按文档编号和原始顺序排序，再按重叠率排序
        sentences.sort(key=lambda x: (-x[1], x[4]))

        # 5. 去重 + 组装结果
        result = []
        seen_keys = set()
        total_len = 0

        for sent, score, source, page, doc_idx in sentences:
            # 去重：前20字相同则跳过
            dedup_key = sent[:20]
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # 格式化输出
            entry = f"[文档{doc_idx}] {sent}。"
            if total_len + len(entry) > max_tokens:
                break
            result.append(entry)
            total_len += len(entry)

        if not result:
            # 没有匹配的句子，取第一个文档的开头
            return self._fallback_truncate(docs, max_tokens)

        return "\n".join(result)

    def _llm_compress(
        self,
        question: str,
        docs: list[dict],
        max_tokens: int,
    ) -> str:
        """LLM压缩（可选，高质量）：用小模型摘要

        使用GLM-4-Flash（最快模型）将文档摘要。
        """
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from ..config import get_llm, get_temperature

            # 拼接文档
            doc_text = "\n\n".join(
                doc.get("content", doc.get("text", ""))[:1000]
                for doc in docs[:3]
            )

            prompt = f"""请将以下文档压缩为不超过{max_tokens}字的摘要，保留与问题"{question}"最相关的信息：

{doc_text}"""

            llm = get_llm(temperature=get_temperature("generation"))
            response = llm.invoke([
                SystemMessage(content="你是文档摘要专家，只保留关键信息，去除冗余。"),
                HumanMessage(content=prompt),
            ])
            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            log.warning(f"[PromptCompressor] LLM压缩失败，降级到extractive: {e}")
            return self._extractive_compress(question, docs, max_tokens)

    @staticmethod
    def _fallback_truncate(docs: list[dict], max_tokens: int) -> str:
        """降级截断（当无法提取关键词时）"""
        parts = []
        total = 0
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", doc.get("text", ""))
            source = doc.get("source", f"文档{i}")
            remaining = max_tokens - total
            if remaining <= 0:
                break
            chunk = content[:remaining]
            parts.append(f"[文档{i}] {chunk}")
            total += len(chunk)
        return "\n".join(parts)


# 全局单例
_compressor_instance: Optional[PromptCompressor] = None

def get_compressor() -> PromptCompressor:
    """获取全局压缩器实例"""
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = PromptCompressor()
    return _compressor_instance
