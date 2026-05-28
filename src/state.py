"""LangGraph 状态定义 — 7层Pipeline的数据流"""
from __future__ import annotations
from typing import TypedDict, Optional, Literal


class Document(TypedDict):
    """检索到的文档片段"""
    doc_id: str
    content: str
    source: str           # 来源文件名
    page: int             # 页码/块号
    metadata: dict        # 额外元数据


class GradedDocument(TypedDict):
    """经过Corrective RAG评分的文档"""
    doc_id: str
    content: str
    source: str
    page: int
    grade: Literal["relevant", "ambiguous", "irrelevant"]
    relevance_score: float   # 0-1 相关度
    reasoning: str           # 评分理由


class Citation(TypedDict):
    """引用溯源"""
    text: str             # 引用的原文片段
    source: str           # 来源文件
    page: int             # 页码


class ConflictInfo(TypedDict):
    """多源冲突信息"""
    topic: str            # 冲突主题
    positions: list[dict] # [{source, claim, evidence, confidence}]
    resolution: str       # 解决方案/建议


class RAGState(TypedDict):
    """DeepRAG Pipeline全局状态"""

    # --- 输入 ---
    question: str                          # 用户提问
    collection_name: str                   # 知识库名称

    # --- 1.Query分析 ---
    question_type: Literal["factual", "reasoning", "comparison", "open_ended"]
    rewritten_query: str                   # 改写后的查询
    search_queries: list[str]             # 拆分的多个子查询

    # --- 2.检索 ---
    retrieved_docs: list[Document]         # 原始检索结果

    # --- 3.Corrective RAG文档评分 ---
    graded_docs: list[GradedDocument]      # 评分后文档
    relevant_count: int
    irrelevant_count: int

    # --- 4.路由决策 ---
    retrieval_decision: Literal["generate", "rewrite", "web_search"]

    # --- 5.生成 ---
    answer: str                            # 生成的答案
    citations: list[Citation]              # 引用列表

    # --- 6.事实校验(Self-RAG) ---
    hallucination_score: float             # 幻觉评分 0-1 (0=无幻觉)
    fact_check_passed: bool
    unsupported_claims: list[str]          # 未被文档支持的断言

    # --- 7.冲突解决 ---
    conflicts: list[ConflictInfo]

    # --- Web Fallback ---
    web_results: list[Document]

    # --- 流程控制 ---
    current_step: str
    retry_count: int
    max_retries: int
    errors: list[str]
    history: list[str]
