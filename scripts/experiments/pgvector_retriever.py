"""pgvector向量数据库检索器
基于PostgreSQL + pgvector扩展实现
- 支持向量检索（Cosine/L2/Inner Product距离）
- 支持元数据过滤
- 支持HNSW索引加速

依赖：pip install psycopg2-binary pgvector
"""
from typing import List, Optional, Dict, Any
import json
from src.state import Document

# 延迟导入：未安装时仍可import本模块
try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None  # type: ignore


class PgvectorRetriever:
    """pgvector向量检索器 - PostgreSQL扩展"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "deep_rag",
        user: str = "postgres",
        password: str = "postgres",
        table_name: str = "documents",
        conn=None
    ):
        """初始化pgvector客户端

        Args:
            host: PostgreSQL服务地址
            port: PostgreSQL服务端口
            database: 数据库名称
            user: 用户名
            password: 密码
            table_name: 表名
            conn: 可选，注入已有连接（便于测试）
        """
        if not PSYCOPG2_AVAILABLE and conn is None:
            raise ImportError(
                "psycopg2 not installed. Run: pip install psycopg2-binary pgvector"
            )

        self.table_name = table_name
        self.embedding_dim = 768  # 默认768维

        if conn is not None:
            self.conn = conn
        else:
            self.conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )

    def create_table(self, embedding_dim: int = 768):
        """创建表并启用pgvector扩展"""
        self.embedding_dim = embedding_dim

        with self.conn.cursor() as cur:
            # 1. 启用pgvector扩展
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 2. 创建表
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({embedding_dim}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. 创建HNSW索引（加速向量检索）
            # 使用cosine距离（vector_cosine_ops）
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops);
            """)

            self.conn.commit()
            print(f"✅ Table {self.table_name} created with pgvector support")

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """批量添加文档

        Args:
            documents: 文档列表
            embeddings: 向量列表
        """
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have same length")

        # 准备数据
        data = [
            (
                doc.get("page_content") or doc.get("content", ""),
                embeddings[i],
                json.dumps(doc.get("metadata", {}))
            )
            for i, doc in enumerate(documents)
        ]

        # 批量插入
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {self.table_name} (content, embedding, metadata)
                VALUES %s
                """,
                data,
                template="(%s, %s::vector, %s::jsonb)"
            )
            self.conn.commit()

        print(f"✅ Added {len(documents)} documents to pgvector")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine"
    ) -> List[Document]:
        """向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回Top-K结果
            metadata_filter: 元数据过滤（例如：{"source": "doc1.md"}）
            distance_metric: 距离度量（cosine/l2/inner）

        Returns:
            检索到的文档列表
        """
        # 选择距离运算符
        distance_ops = {
            "cosine": "<=>",  # cosine distance
            "l2": "<->",      # L2 distance
            "inner": "<#>",   # inner product (negative)
        }
        op = distance_ops.get(distance_metric, "<=>")

        # 构建查询
        query = f"""
            SELECT id, content, metadata, embedding {op} %s::vector AS distance
            FROM {self.table_name}
        """

        # 添加元数据过滤
        params = [query_vector]
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(f"metadata->>{key!r} = %s")
                params.append(str(value))
            query += " WHERE " + " AND ".join(conditions)

        # 排序和限制
        query += f" ORDER BY distance LIMIT {top_k};"

        # 执行查询
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        # 转换为Document对象
        documents = []
        for row in rows:
            doc_id, content, metadata, distance = row
            doc: Document = {
                "doc_id": str(doc_id),
                "content": content,
                "source": metadata.get("source", "unknown") if metadata else "unknown",
                "page": metadata.get("page", 0) if metadata else 0,
                "metadata": {
                    **(metadata or {}),
                    "id": doc_id,
                    "distance": float(distance)
                }
            }
            documents.append(doc)

        return documents

    def delete_collection(self):
        """删除表"""
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")
            self.conn.commit()
        print(f"✅ Table {self.table_name} deleted")

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
