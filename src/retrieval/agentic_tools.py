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


try:
    from src.logging_config import get_logger
except Exception:
    import logging

    def get_logger(n):  # type: ignore
        return logging.getLogger(n)

logger = get_logger(__name__)

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
    """精确查询工具（SQLite实现，零成本）

    场景：
    - 查特定用户ID
    - 查特定订单号
    - 查特定版本号 version == 'v3.5'
    """

    def __init__(self, db_path: str = None):
        """初始化SQLite精确查询

        Args:
            db_path: SQLite数据库路径，默认使用项目data目录
        """
        import sqlite3
        if db_path is None:
            from src.config import DATA_DIR
            db_path = str(DATA_DIR / "knowledge.db")
        self.db_path = db_path
        # 建表（首次运行时自动创建）
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    source TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, **kwargs) -> List[Document]:
        """精确匹配检索

        Args:
            query: 查询文本（从中提取ID/编号）
            filters: 精确过滤条件
        """
        import re
        import sqlite3
        import json

        results = []

        # 从query中提取可能的ID（字母+数字组合，至少3字符）
        id_patterns = [
            re.compile(r"(?:ID|id|编号|版本|文档)[\s:：]*([A-Za-z0-9\-_]{2,})"),
            re.compile(r"\b([A-Z]{2,}[\-_]\d{2,})\b"),  # 如 DOC-001, NX-100
        ]

        doc_ids = []
        for pattern in id_patterns:
            matches = pattern.findall(query)
            doc_ids.extend(matches)

        if not doc_ids:
            return results

        with sqlite3.connect(self.db_path) as conn:
            for doc_id in doc_ids:
                row = conn.execute(
                    "SELECT content, source, metadata FROM documents WHERE id = ?",
                    (doc_id,)
                ).fetchone()
                if row:
                    results.append(Document(
                        content=row[0],
                        source=row[1] or "sqlite",
                        page=0,
                        metadata=json.loads(row[2]) if row[2] else {},
                    ))

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
    """图检索工具（NetworkX实现，无需Neo4j）

    用轻量级NetworkX有向图替代Neo4j，数据持久化到pickle文件。
    适合毕设/个人项目，零基础设施成本。

    场景：
    - 用户问"Spring Security 的认证流程和 WebFlux 有什么关系？"
    - 图检索能精准顺着"实体-关系-实体"的链条把跨模块的架构抓出来
    """

    def __init__(self, graph_path: str = None):
        """初始化知识图谱

        Args:
            graph_path: 图谱pickle文件路径，默认使用项目data目录
        """
        import pickle
        if graph_path is None:
            from src.config import DATA_DIR
            graph_path = str(DATA_DIR / "knowledge_graph.pkl")
        self.graph_path = graph_path

        try:
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
        except (FileNotFoundError, EOFError):
            import networkx as nx
            self.graph = nx.DiGraph()
            self._save()

    def _save(self):
        """持久化图谱到文件"""
        import pickle
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)

    def search(self, query: str, relation: Optional[str] = None,
               max_depth: int = 2, **kwargs) -> List[Document]:
        """图检索

        Args:
            query: 查询文本（从中提取实体名称）
            relation: 关系类型（如"依赖"、"继承"）
            max_depth: 最大搜索深度
        """
        import re
        import networkx as nx

        results = []

        # 简单实体提取：取query中的名词性词汇（2-20字符的中文/英文词组）
        entity = None
        # 尝试匹配已知实体
        for node in self.graph.nodes:
            if node in query:
                entity = node
                break

        if entity is None:
            # 提取可能的实体名（中文2-10字 或 英文单词）
            match = re.search(r"[\u4e00-\u9fa5]{2,10}|[A-Za-z]{2,20}", query)
            entity = match.group() if match else None

        if entity is None or entity not in self.graph:
            return results

        # BFS遍历找关联节点（同时检查出边和入边）
        try:
            # 检查出边：entity → neighbor
            for neighbor in self.graph.successors(entity):
                edge_data = self.graph.get_edge_data(entity, neighbor)
                if not edge_data:
                    continue
                rel = edge_data.get("relation", "关联")
                if relation and rel != relation:
                    continue
                results.append(Document(
                    content=f"{entity} --{rel}--> {neighbor}",
                    source="knowledge_graph",
                    page=0,
                    metadata={"entity": entity, "relation": rel, "target": neighbor, "direction": "out"},
                ))
            # 检查入边：neighbor → entity
            for neighbor in self.graph.predecessors(entity):
                edge_data = self.graph.get_edge_data(neighbor, entity)
                if not edge_data:
                    continue
                rel = edge_data.get("relation", "关联")
                if relation and rel != relation:
                    continue
                results.append(Document(
                    content=f"{neighbor} --{rel}--> {entity}",
                    source="knowledge_graph",
                    page=0,
                    metadata={"entity": neighbor, "relation": rel, "target": entity, "direction": "in"},
                ))
        except Exception:
            pass

        return results

    def add_relation(self, source: str, target: str, relation: str = "关联"):
        """添加实体关系（建图时用）"""
        self.graph.add_edge(source, target, relation=relation)
        self._save()

    def get_description(self) -> str:
        return """图检索工具：基于知识图谱的实体关系检索。
适用场景：用户问"A和B有什么关系"、"A依赖哪些模块"等关系查询。"""


class WebSearchTool(RetrievalTool):
    """网络搜索工具（DuckDuckGo实现，完全免费无需API Key）

    场景：
    - 当本地知识库查不到
    - 或者需要最新的时效性信息（比如 2026 年某个框架的最新特性）
    """

    def __init__(self, search_api: str = "duckduckgo"):
        """
        Args:
            search_api: 搜索引擎（duckduckgo=免费默认 / tavily=需Key）
        """
        self.search_api = search_api

    def search(self, query: str, max_results: int = 3, **kwargs) -> List[Document]:
        """网络搜索

        Args:
            query: 查询文本
            max_results: 最大结果数量
        """
        results = []

        if self.search_api == "duckduckgo":
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append(Document(
                            content=r.get("body", ""),
                            source=r.get("href", "web"),
                            page=0,
                            metadata={"title": r.get("title", ""), "engine": "duckduckgo"},
                        ))
            except ImportError:
                logger.warning(
                    "duckduckgo-search not installed. Run: pip install duckduckgo-search"
                )
            except Exception as e:
                logger.warning("WebSearch failed: %s", e)

        return results

    def get_description(self) -> str:
        return """网络搜索工具：从互联网搜索最新信息。
适用场景：本地知识库没有答案，或需要最新的时效性信息。"""


class KBStatsTool(RetrievalTool):
    """系统全量统计工具 — 知识库、模型、数据库、GPU等元数据"""

    def _collect_stats(self) -> str:
        import os
        sections = []

        # 1. ChromaDB 知识库
        try:
            from src.config import get_chroma_client
            client = get_chroma_client()
            collections = client.list_collections()
            col_details = []
            total = 0
            for col in collections:
                try:
                    count = col.count()
                    total += count
                    col_details.append(f"  - {col.name}: {count} 篇")
                except Exception:
                    col_details.append(f"  - {col.name}: 读取失败")
            sections.append(f"【知识库 ChromaDB】\n  集合数: {len(collections)}\n  总文档数: {total}\n" + "\n".join(col_details))
        except Exception as e:
            sections.append(f"【知识库 ChromaDB】连接失败: {e}")

        # 2. Embedding 模型
        try:
            from src.config import EMBEDDING_MODEL, DEVICE, EMBEDDING_MODE
            sections.append(f"【Embedding 模型】\n  模型: {EMBEDDING_MODEL}\n  模式: {EMBEDDING_MODE}\n  设备: {DEVICE}")
        except Exception:
            sections.append("【Embedding 模型】未配置")

        # 3. LLM 模型
        try:
            from src.config import LLM_BACKEND, LLM_MODEL
            sections.append(f"【LLM 模型】\n  后端: {LLM_BACKEND}\n  模型: {LLM_MODEL}")
        except Exception:
            sections.append("【LLM 模型】未配置")

        # 4. GPU 信息
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                vram_used = torch.cuda.memory_allocated(0) / 1024**3
                sections.append(f"【GPU】\n  设备: {gpu_name}\n  显存: {vram_used:.1f}/{vram_total:.1f} GB\n  PyTorch: {torch.__version__}")
            else:
                sections.append(f"【GPU】不可用 (CPU模式)\n  PyTorch: {torch.__version__}")
        except Exception:
            sections.append("【GPU】PyTorch 未安装")

        # 5. 文档目录
        try:
            from src.config import DOCS_DIR
            if DOCS_DIR and os.path.isdir(DOCS_DIR):
                file_count = sum(1 for f in os.rglob(DOCS_DIR) if f.is_file())
                sections.append(f"【文档目录】\n  路径: {DOCS_DIR}\n  文件数: {file_count}")
        except Exception:
            pass

        # 6. BM25 / 缓存状态
        try:
            from src.retrieval.cache import get_cache_stats
            stats = get_cache_stats()
            sections.append(f"【缓存】\n  文档缓存: {stats['doc_cache_size']} 项\n  BM25缓存: {stats['bm25_cache_size']} 项\n  LLM缓存: {stats['llm_cache_size']} 项\n  TTL: {stats['ttl_seconds']}秒")
        except Exception:
            pass

        return "\n\n".join(sections)

    def search(self, query: str, **kwargs) -> List[Document]:
        summary = self._collect_stats()
        return [Document(page_content=summary, metadata={"source": "系统元数据", "is_stats": True})]

    def get_description(self) -> str:
        return """系统全量统计工具：查询知识库数据量、Embedding模型、LLM模型、GPU显存、文档目录、缓存状态等系统元数据。
适用场景：用户询问系统数据量、模型信息、GPU状态、知识库大小等元信息。"""


class AgenticRAGToolbox:
    """Agentic RAG工具箱

    Agent可以根据问题动态选择使用哪些工具
    """

    def __init__(self):
        self.tools: Dict[str, RetrievalTool] = {}

    def register_tool(self, name: str, tool: RetrievalTool) -> None:
        """注册工具"""
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")

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
def create_toolbox(retriever) -> "AgenticRAGToolbox":
    """创建工具箱并注册全部4个工具

    全部工具零成本实现：
    - vector_search: 向量语义检索（ChromaDB）
    - exact_match: SQLite精确查询（本地文件）
    - graph_search: NetworkX图谱检索（本地文件）
    - web_search: DuckDuckGo网络搜索（免费）
    """
    toolbox = AgenticRAGToolbox()

    # 1. 向量检索（已有）
    toolbox.register_tool("vector_search", VectorSearchTool(retriever))

    # 2. 精确查询（SQLite，零成本）
    toolbox.register_tool("exact_match", ExactMatchTool())

    # 3. 图检索（NetworkX，零成本）
    toolbox.register_tool("graph_search", GraphSearchTool())

    # 4. 网络搜索（DuckDuckGo，免费）
    toolbox.register_tool("web_search", WebSearchTool())

    # 5. 知识库统计（查询数据量等元信息）
    toolbox.register_tool("kb_stats", KBStatsTool())

    return toolbox
