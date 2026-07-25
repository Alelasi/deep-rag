"""Web Fallback 模块测试"""
import pytest
from unittest.mock import Mock, patch
from src.retrieval.web_fallback import (
    web_search_fallback,
    _search_duckduckgo,
    _search_tavily,
    _search_serper,
    _mock_results,
    DDGS_AVAILABLE,
    REQUESTS_AVAILABLE
)


class TestWebFallback:
    """Web Fallback 测试套件"""

    def test_mock_results(self):
        """测试：Mock 结果生成"""
        results = _mock_results("test query", max_results=3)

        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all("doc_id" in r for r in results)
        assert all("content" in r for r in results)
        assert all("source" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all(r["metadata"]["is_web"] for r in results)
        assert all(r["metadata"]["engine"] == "mock" for r in results)

    def test_web_search_fallback_default(self):
        """测试：默认使用 DuckDuckGo"""
        # Mock DuckDuckGo to avoid network dependency
        with patch('src.retrieval.web_fallback._search_duckduckgo', return_value=_mock_results("test query", 2)):
            results = web_search_fallback("test query", max_results=2)

            # 应该返回结果（即使DuckDuckGo失败，也会fallback到mock）
            assert len(results) == 2
            assert all(isinstance(r, dict) for r in results)

    def test_web_search_fallback_unknown_engine(self):
        """测试：未知搜索引擎使用 mock"""
        results = web_search_fallback("test query", max_results=2, engine="unknown")

        assert len(results) == 2
        assert all(r["metadata"]["engine"] == "mock" for r in results)

    @pytest.mark.skipif(not DDGS_AVAILABLE, reason="duckduckgo-search not installed")
    def test_search_duckduckgo_structure(self):
        """测试：DuckDuckGo 返回结构"""
        # 即使搜索失败，也应该返回mock结果
        results = _search_duckduckgo("test query", max_results=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        if results:
            assert "doc_id" in results[0]
            assert "content" in results[0]
            assert "source" in results[0]
            assert "metadata" in results[0]

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_search_tavily_no_api_key(self):
        """测试：Tavily 无 API Key 时使用 mock"""
        with patch.dict('os.environ', {}, clear=True):
            results = _search_tavily("test query", max_results=2)

            assert len(results) == 2
            assert all(r["metadata"]["engine"] == "mock" for r in results)

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_search_serper_no_api_key(self):
        """测试：Serper 无 API Key 时使用 mock"""
        with patch.dict('os.environ', {}, clear=True):
            results = _search_serper("test query", max_results=2)

            assert len(results) == 2
            assert all(r["metadata"]["engine"] == "mock" for r in results)

    def test_result_format(self):
        """测试：结果格式一致性"""
        results = web_search_fallback("test query", max_results=3)

        for result in results:
            # 必须字段
            assert "doc_id" in result
            assert "content" in result
            assert "source" in result
            assert "page" in result
            assert "metadata" in result

            # 类型检查
            assert isinstance(result["doc_id"], str)
            assert isinstance(result["content"], str)
            assert isinstance(result["source"], str)
            assert isinstance(result["page"], int)
            assert isinstance(result["metadata"], dict)

            # 元数据字段
            assert "is_web" in result["metadata"]
            assert "engine" in result["metadata"]
            assert result["metadata"]["is_web"] is True

    def test_max_results_limit(self):
        """测试：最大结果数限制"""
        # Mock to avoid network dependency
        for max_results in [1, 3, 5]:
            with patch('src.retrieval.web_fallback._search_duckduckgo', return_value=_mock_results("test query", max_results)):
                results = web_search_fallback("test query", max_results=max_results)
                assert len(results) == max_results

    def test_empty_query(self):
        """测试：空查询"""
        results = web_search_fallback("", max_results=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_chinese_query(self):
        """测试：中文查询"""
        # Mock to avoid network dependency
        with patch('src.retrieval.web_fallback._search_duckduckgo', return_value=_mock_results("什么是向量数据库", 2)):
            results = web_search_fallback("什么是向量数据库", max_results=2)
            assert isinstance(results, list)
            assert len(results) == 2

    def test_long_query(self):
        """测试：长查询"""
        long_query = "这是一个非常长的查询 " * 20
        # Mock to avoid network dependency
        with patch('src.retrieval.web_fallback._search_duckduckgo', return_value=_mock_results(long_query, 2)):
            results = web_search_fallback(long_query, max_results=2)
            assert isinstance(results, list)
            assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
