"""Agentic RAG工具箱
基于理论文档中的"Agent的高级检索工具箱"实现

包含4种专业检索工具：
1. 精确查询工具（Exact Match Tool）
2. 向量/语义检索工具（Vector/Semantic Search Tool）
3. 图检索工具（Graph RAG Tool）
4. 网络搜索工具（Web Search Tool）

Agent可以根据问题动态选择使用哪些工具
"""
from typing import List, Dict, Any, Optional
from src.state import Document
from abc import ABC, abstractmethod


class RetrievalTool(ABC):
    """检索工具基类"""

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Document]:
        """执行检索"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """返回工具描述（供Agent理解）"""
        pass


class ExactMatchTool(RetrievalTool):
    """精确查询工具

    场景：
    - 查特定用户ID
    - 查特定订单号
    - 查特定版本号 version == 'v3.5'
    """

    def __init__(self, database):
        """
        Args:
            database: 可以是MySQL、Redis或Elasticsearch
        """
        self.db = database

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """精确匹配检索

        Args:
            query: 查询文本（可能包含精确匹配的关键词）
            filters: 精确过滤条件，如 {"user_id": "12345", "status": "active"}
        """
        # 这里是示例实现，实际需要连接真实数据库
        results = []

        # 示例：从数据库查询
        # results = self.db.query(filters)

        return results

    def get_description(self) -> str:
        return """精确查询工具：用于查找特定ID、订单号、版本号等精确匹配的信息。
适用场景：用户问"查询用户ID为12345的信息"、"版本v3.5的文档"等。"""


class VectorSearchTool(RetrievalTool):
    """向量/语义检索工具

    场景：
    - 用户提问很模糊（如"系统老是报错连不上"）
    - 用来捞取原理相似的排障文档
    """

    def __init__(self, retriever):
        """
        Args:
            retriever: 向量检索器（ChromaDB、Qdrant等）
        """
        self.retriever = retriever

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
        """
        # 调用向量检索器
        results = self.retriever.retrieve(query, top_k=top_k)
        return results

    def get_description(self) -> str:
        return """向量检索工具：基于语义相似度检索，适合模糊查询。
适用场景：用户问"如何解决连接超时问题"、"INTJ的性格特点"等语义查询。"""


class GraphSearchTool(RetrievalTool):
    """图检索工具（Graph RAG）

    场景：
    - 用户问"Spring Security 的认证流程和 WebFlux 有什么关系？"
    - 图检索能精准顺着"实体-关系-实体"的链条把跨模块的架构抓出来
    """

    def __init__(self, graph_db):
        """
        Args:
            graph_db: 知识图谱数据库（Neo4j等）
        """
        self.graph_db = graph_db

    def search(self, entity: str, relation: Optional[str] = None,
               max_depth: int = 2) -> List[Document]:
        """图检索

        Args:
            entity: 实体名称（如"Spring Security"）
            relation: 关系类型（如"依赖"、"继承"）
            max_depth: 最大搜索深度
        """
        # 这里是示例实现，实际需要连接知识图谱
        results = []

        # 示例：查询知识图谱
        # query = f"MATCH (a)-[{relation}]->(b) WHERE a.name='{entity}'"
        # results = self.graph_db.query(query)

        return results

    def get_description(self) -> str:
        return """图检索工具：基于知识图谱的实体关系检索。
适用场景：用户问"A和B有什么关系"、"A依赖哪些模块"等关系查询。"""


class WebSearchTool(RetrievalTool):
    """网络搜索工具

    场景：
    - 当本地知识库查不到
    - 或者需要最新的时效性信息（比如 2026 年某个框架的最新特性）
    """

    def __init__(self, search_api: str = "tavily"):
        """
        Args:
            search_api: 搜索API类型（tavily、google、bing等）
        """
        self.search_api = search_api

    def search(self, query: str, max_results: int = 3) -> List[Document]:
        """网络搜索

        Args:
            query: 查询文本
            max_results: 最大结果数量
        """
        # 这里是示例实现，实际需要调用搜索API
        results = []

        # 示例：调用Tavily API
        # from tavily import TavilyClient
        # client = TavilyClient(api_key="...")
        # search_results = client.search(query, max_results=max_results)
        # results = [Document(...) for r in search_results]

        return results

    def get_description(self) -> str:
        return """网络搜索工具：从互联网搜索最新信息。
适用场景：本地知识库没有答案，或需要最新的时效性信息。"""


class AgenticRAGToolbox:
    """Agentic RAG工具箱

    Agent可以根据问题动态选择使用哪些工具
    """

    def __init__(self):
        self.tools: Dict[str, RetrievalTool] = {}

    def register_tool(self, name: str, tool: RetrievalTool):
        """注册工具"""
        self.tools[name] = tool
        print(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[RetrievalTool]:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """列出所有工具（供Agent选择）"""
        return [
            {
                "name": name,
                "description": tool.get_description()
            }
            for name, tool in self.tools.items()
        ]

    def execute_tool(self, tool_name: str, query: str, **kwargs) -> List[Document]:
        """执行工具"""
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")

        return tool.search(query, **kwargs)


# 示例：如何使用Agentic RAG工具箱
def create_toolbox(retriever) -> AgenticRAGToolbox:
    """创建工具箱并注册所有工具"""
    toolbox = AgenticRAGToolbox()

    # 注册向量检索工具
    toolbox.register_tool("vector_search", VectorSearchTool(retriever))

    # 注册精确查询工具（需要数据库连接）
    # toolbox.register_tool("exact_match", ExactMatchTool(database))

    # 注册图检索工具（需要知识图谱）
    # toolbox.register_tool("graph_search", GraphSearchTool(graph_db))

    # 注册网络搜索工具
    # toolbox.register_tool("web_search", WebSearchTool())

    return toolbox
