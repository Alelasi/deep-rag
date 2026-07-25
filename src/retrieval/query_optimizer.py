"""
查询优化器
查询重写 + 扩展 + 分解
"""
from typing import List, Dict
import re


class QueryOptimizer:
    """查询优化器"""

    def __init__(self):
        # 同义词映射
        self.synonyms = {
            "安装": ["安装", "配置", "部署", "setup"],
            "优化": ["优化", "加速", "提升", "改进"],
            "问题": ["问题", "错误", "bug", "异常"],
            "方法": ["方法", "方案", "策略", "技巧"],
            "gpu": ["gpu", "显卡", "cuda"],
            "内存": ["内存", "memory", "ram"],
        }

        # 缩写扩展
        self.abbreviations = {
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "gpu": "graphics processing unit",
            "cpu": "central processing unit",
        }

    def optimize(self, query: str) -> Dict:
        """
        优化查询

        Returns:
            {
                'original': 原始查询,
                'rewritten': 重写后的查询,
                'expanded': 扩展后的查询列表,
                'keywords': 关键词列表,
            }
        """
        # 清理查询
        cleaned = self._clean_query(query)

        # 提取关键词
        keywords = self._extract_keywords(cleaned)

        # 查询重写
        rewritten = self._rewrite_query(cleaned, keywords)

        # 查询扩展
        expanded = self._expand_query(rewritten, keywords)

        return {
            'original': query,
            'cleaned': cleaned,
            'rewritten': rewritten,
            'expanded': expanded,
            'keywords': keywords,
        }

    def _clean_query(self, query: str) -> str:
        """清理查询"""
        # 去除多余空格
        query = re.sub(r'\s+', ' ', query).strip()

        # 去除特殊字符（保留中文、英文、数字）
        query = re.sub(r'[^\w\s一-鿿]', ' ', query)

        return query

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单分词（按空格）
        words = query.lower().split()

        # 过滤停用词
        stopwords = {'的', '了', '是', '在', '有', '和', '与', '或', '等',
                     'the', 'a', 'an', 'is', 'are', 'was', 'were'}

        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        return keywords

    def _rewrite_query(self, query: str, keywords: List[str]) -> str:
        """查询重写"""
        rewritten = query

        # 扩展缩写
        for abbr, full in self.abbreviations.items():
            if abbr in query.lower():
                rewritten = re.sub(
                    rf'\b{abbr}\b',
                    f'{abbr} {full}',
                    rewritten,
                    flags=re.IGNORECASE
                )

        return rewritten

    def _expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """查询扩展"""
        expanded = [query]

        # 同义词扩展
        for keyword in keywords:
            if keyword in self.synonyms:
                for synonym in self.synonyms[keyword]:
                    if synonym != keyword:
                        expanded_query = query.replace(keyword, synonym)
                        if expanded_query not in expanded:
                            expanded.append(expanded_query)

        return expanded[:5]  # 最多5个扩展

    def decompose_query(self, query: str) -> List[str]:
        """分解复杂查询"""
        # 检测是否是复合查询
        if '和' in query or 'and' in query.lower():
            # 按"和"分割
            parts = re.split(r'[和&]', query)
            return [p.strip() for p in parts if p.strip()]

        if '，' in query or ',' in query:
            # 按逗号分割
            parts = re.split(r'[，,]', query)
            return [p.strip() for p in parts if p.strip()]

        # 单一查询
        return [query]

    def suggest_alternatives(self, query: str, no_results: bool = False) -> List[str]:
        """建议替代查询"""
        suggestions = []

        if no_results:
            # 简化查询
            keywords = self._extract_keywords(query)
            if len(keywords) > 2:
                # 只保留前2个关键词
                suggestions.append(' '.join(keywords[:2]))

            # 更通用的查询
            if 'gpu' in query.lower():
                suggestions.append('GPU 加速')
            if '安装' in query:
                suggestions.append('安装指南')
            if '优化' in query:
                suggestions.append('性能优化')

        return suggestions[:3]
