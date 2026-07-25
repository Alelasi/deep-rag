"""
集成测试 - v2.2增强检索模块
测试3种检索模式：enhanced / agentic / hybrid
"""
import pytest
import os
from unittest.mock import Mock, patch
from src.graph import query, get_enhanced_retriever


class TestEnhancedRetrievalIntegration:
    """测试增强检索模块集成到主Pipeline"""

    @patch('src.graph.get_enhanced_retriever')
    @patch('src.retrieval.unified_retriever.UnifiedRetriever')
    def test_enhanced_mode_valid_query(self, mock_unified, mock_get_enhanced):
        """测试Enhanced模式 - 有效查询"""
        # Mock增强检索器
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = {
            'query': 'test query',
            'validation': {'is_valid': True, 'intent': 'knowledge', 'confidence': 0.9},
            'results': [
                {
                    'text': 'LangChain是一个框架',
                    'source': 'doc1.md',
                    'page': 1,
                    'metadata': {},
                    'similarity': 0.85
                }
            ],
            'confidence': 'high',
            'used_web_fallback': False,
            'metadata': {}
        }
        mock_get_enhanced.return_value = mock_retriever

        # 设置环境变量
        os.environ['RETRIEVAL_MODE'] = 'enhanced'

        # 执行查询（Mock大部分节点避免依赖）
        with patch('src.graph.analyze_query') as mock_analyze, \
             patch('src.graph.grade_documents') as mock_grade, \
             patch('src.graph.generate_answer') as mock_generate, \
             patch('src.graph.check_facts') as mock_facts, \
             patch('src.graph.resolve_conflicts') as mock_conflicts:

            mock_analyze.return_value = {
                'question_type': 'factual',
                'rewritten_query': 'test query',
                'search_queries': ['test query']
            }
            mock_grade.return_value = [
                {'grade': 'relevant', 'relevance_score': 0.8, 'reasoning': 'test'}
            ]
            mock_generate.return_value = {
                'answer': 'Test answer',
                'citations': []
            }
            mock_facts.return_value = {
                'hallucination_score': 0.1,
                'passed': True,
                'unsupported_claims': []
            }
            mock_conflicts.return_value = []

            result = query("什么是LangChain？", collection_name="test_kb")

            # 验证
            assert 'Enhanced v2.2' in str(result.get('history', [])), \
                f"应显示Enhanced模式，实际history: {result.get('history')}"
            assert result['current_step'] == 'completed'

    @patch('src.graph.get_enhanced_retriever')
    def test_enhanced_mode_invalid_query(self, mock_get_enhanced):
        """测试Enhanced模式 - 无效查询（闲聊应拒识）"""
        # Mock增强检索器返回拒识结果
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = {
            'query': '你好',
            'validation': {
                'is_valid': False,
                'intent': 'chitchat',
                'confidence': 0.8,
                'reason': '检测到闲聊意图',
                'suggestion': '请提出技术相关的问题'
            },
            'results': [],
            'confidence': 'rejected',
            'used_web_fallback': False,
            'metadata': {'rejection_reason': '检测到闲聊意图'}
        }
        mock_get_enhanced.return_value = mock_retriever

        os.environ['RETRIEVAL_MODE'] = 'enhanced'

        with patch('src.graph.analyze_query') as mock_analyze:
            mock_analyze.return_value = {
                'question_type': 'factual',
                'rewritten_query': '你好',
                'search_queries': ['你好']
            }

            result = query("你好", collection_name="test_kb")

            # 验证拒识
            assert result['current_step'] == 'query_rejected', \
                f"应拒识查询，实际step: {result['current_step']}"
            assert len(result['retrieved_docs']) == 0
            assert 'rejected' in str(result.get('history', [])).lower()


    def test_config_retrieval_mode(self):
        """测试配置项是否正确加载"""
        from src.config import RETRIEVAL_MODE

        # 默认应该是enhanced
        assert RETRIEVAL_MODE in ['enhanced', 'agentic', 'hybrid'], \
            f"RETRIEVAL_MODE应为3种模式之一，实际: {RETRIEVAL_MODE}"


@pytest.mark.skip(reason="需要真实向量数据库和LLM")
def test_end_to_end_enhanced_mode():
    """端到端测试（需要真实环境）"""
    os.environ['RETRIEVAL_MODE'] = 'enhanced'

    result = query("什么是LangChain？", collection_name="test_kb")

    assert result['answer'], "应返回答案"
    assert 'Enhanced v2.2' in str(result['history']), "应使用Enhanced模式"
    assert result['current_step'] == 'completed'


@pytest.mark.skip(reason="需要真实环境")
def test_mode_comparison():
    """对比3种检索模式（需要真实环境）"""
    test_question = "什么是LangChain的LCEL？"
    results = {}

    for mode in ['enhanced', 'agentic', 'hybrid']:
        os.environ['RETRIEVAL_MODE'] = mode
        result = query(test_question, collection_name="test_kb")
        results[mode] = {
            'answer_length': len(result.get('answer', '')),
            'citations': len(result.get('citations', [])),
            'retrieval_steps': [h for h in result.get('history', []) if 'Retrieved' in h]
        }

    # 验证3种模式都能返回结果
    for mode, data in results.items():
        assert data['answer_length'] > 0, f"{mode}模式应返回答案"
        assert len(data['retrieval_steps']) > 0, f"{mode}模式应有检索步骤"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
