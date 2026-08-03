"""L1 单元测试 — 状态定义与图节点（无外部依赖）

测试 src/state.py 的类型定义和 src/llm/prompt_templates.py 的模板逻辑
"""
import pytest
from typing import get_type_hints


@pytest.mark.L1
class TestStateDefinitions:
    """RAGState 类型定义测试"""

    def test_rag_state_has_question(self):
        """RAGState 包含 question 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "question" in hints

    def test_rag_state_has_retrieved_docs(self):
        """RAGState 包含 retrieved_docs 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "retrieved_docs" in hints

    def test_rag_state_has_graded_docs(self):
        """RAGState 包含 graded_docs 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "graded_docs" in hints

    def test_rag_state_has_answer(self):
        """RAGState 包含 answer 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "answer" in hints

    def test_rag_state_has_citations(self):
        """RAGState 包含 citations 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "citations" in hints

    def test_rag_state_has_hallucination_score(self):
        """RAGState 包含 hallucination_score 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "hallucination_score" in hints

    def test_rag_state_has_retry_count(self):
        """RAGState 包含 retry_count 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "retry_count" in hints

    def test_rag_state_has_errors(self):
        """RAGState 包含 errors 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "errors" in hints

    def test_rag_state_has_agent_fields(self):
        """RAGState 包含 ReAct Agent 字段"""
        from src.state import RAGState
        hints = get_type_hints(RAGState)
        assert "next_action" in hints
        assert "agent_reason" in hints
        assert "retrieval_round" in hints
        assert "used_tools" in hints

    def test_document_type_fields(self):
        """Document 类型有正确字段"""
        from src.state import Document
        hints = get_type_hints(Document)
        assert "doc_id" in hints
        assert "content" in hints
        assert "source" in hints
        assert "page" in hints
        assert "metadata" in hints

    def test_graded_document_fields(self):
        """GradedDocument 类型有正确字段"""
        from src.state import GradedDocument
        hints = get_type_hints(GradedDocument)
        assert "grade" in hints
        assert "relevance_score" in hints
        assert "reasoning" in hints

    def test_citation_fields(self):
        """Citation 类型有正确字段"""
        from src.state import Citation
        hints = get_type_hints(Citation)
        assert "text" in hints
        assert "source" in hints
        assert "page" in hints


@pytest.mark.L1
class TestPromptTemplates:
    """Prompt 模板系统测试"""

    def test_few_shot_example_to_text_with_description(self):
        """FewShotExample 带描述转文本"""
        from src.llm.prompt_templates import FewShotExample
        example = FewShotExample(
            input="什么是RAG",
            output="RAG是检索增强生成",
            description="基础概念"
        )
        text = example.to_text()
        assert "基础概念" in text
        assert "什么是RAG" in text
        assert "RAG是检索增强生成" in text

    def test_few_shot_example_to_text_without_description(self):
        """FewShotExample 不带描述转文本"""
        from src.llm.prompt_templates import FewShotExample
        example = FewShotExample(
            input="什么是RAG",
            output="RAG是检索增强生成"
        )
        text = example.to_text()
        assert "什么是RAG" in text
        assert "RAG是检索增强生成" in text
        assert "【" not in text  # 无描述标记

    def test_prompt_builder_chain(self):
        """PromptBuilder 链式调用"""
        from src.llm.prompt_templates import PromptBuilder, PromptTemplate
        template = (PromptBuilder()
                  .role("专家")
                  .task("回答问题")
                  .context("上下文")
                  .build())
        assert isinstance(template, PromptTemplate)
        prompt_text = template.render()
        assert isinstance(prompt_text, str)
        assert "专家" in prompt_text
        assert "回答问题" in prompt_text

    def test_prompt_builder_with_example(self):
        """PromptBuilder 添加示例"""
        from src.llm.prompt_templates import PromptBuilder
        template = (PromptBuilder()
                  .role("专家")
                  .task("回答")
                  .example("问题", "答案")
                  .build())
        prompt_text = template.render()
        assert "问题" in prompt_text
        assert "答案" in prompt_text
