"""Qdrant 索引器 — 替代 ChromaDB Indexer

使用 Qdrant 向量数据库，索引持久化到磁盘，重启零损耗。
"""
import hashlib
import logging
import time
from pathlib import Path
from typing import List

import jieba
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .qdrant_retriever import QdrantRetriever, get_qdrant_retriever

log = logging.getLogger(__name__)

# 支持的文件类型
SUPPORTED_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html"}

# 分块配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _token_length(text: str) -> int:
    """Token 长度估算"""
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except ImportError:
        cjk_count = sum(1 for c in text if '一' <= c <= '鿿')
        ascii_count = len(text) - cjk_count
        return int(cjk_count * 1.5 + ascii_count * 0.25)


class QdrantIndexer:
    """Qdrant 索引管理器"""

    def __init__(self, collection_name: str = "general_kb"):
        self.collection_name = collection_name
        self.retriever = get_qdrant_retriever(collection_name)
        self._embedder = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "],
            length_function=_token_length,
        )

    def _get_embedder(self):
        """获取 SentenceTransformer"""
        if self._embedder is None:
            from src.ui.model_cache import get_embedding_model
            from src.config import EMBEDDING_MODEL, DEVICE
            self._embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)
        return self._embedder

    def is_already_indexed(self) -> bool:
        """检查是否已索引"""
        return self.retriever.is_ready()

    def index_directory(self, dir_path: str) -> int:
        """索引整个目录"""
        if self.is_already_indexed():
            count = self.retriever.count()
            log.info(f"[QdrantIndexer] 已有 {count} 条记录，跳过索引")
            return 0

        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        t0 = time.time()
        all_chunks = []
        for filepath in sorted(path.rglob("*")):
            if filepath.suffix.lower() not in SUPPORTED_EXTS:
                continue
            if filepath.name.startswith("."):
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if len(content) < 20:
                    continue
                chunks = self._splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc_id = hashlib.md5(f"{filepath}:{i}".encode()).hexdigest()[:12]
                    all_chunks.append({
                        "doc_id": doc_id,
                        "content": chunk,
                        "source": str(filepath.relative_to(path)),
                        "page": i + 1,
                    })
            except Exception:
                continue

        if not all_chunks:
            return 0

        # 批量 embedding
        embedder = self._get_embedder()
        batch_size = 100
        total_added = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = embedder.encode([c["content"] for c in batch]).tolist()
            self.retriever.add_documents(
                documents=[c["content"] for c in batch],
                embeddings=embeddings,
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
                ids=[c["doc_id"] for c in batch],
            )
            total_added += len(batch)

        # 构建 BM25 索引
        if all_chunks:
            self._build_bm25(all_chunks)

        elapsed = time.time() - t0
        log.info(f"[QdrantIndexer] 索引完成: {total_added} 块, {elapsed:.1f}s")
        return total_added

    def get_collection(self):
        """返回检索器（兼容旧接口）"""
        return self.retriever

    def get_all_collections(self):
        """返回所有集合（Qdrant 只有一个集合，兼容旧接口）"""
        return [self.retriever]

    def get_bm25(self):
        """返回 BM25 索引（需要先调用 index_directory）"""
        return getattr(self, '_bm25', None), getattr(self, '_bm25_docs', [])

    def _build_bm25(self, chunks: list[dict]):
        """构建 BM25 索引"""
        from rank_bm25 import BM25Okapi
        self._bm25_docs = chunks
        corpus = [list(jieba.cut(c["content"])) for c in chunks]
        self._bm25 = BM25Okapi(corpus)


# 全局索引器缓存
_indexers = {}


def get_qdrant_indexer(collection_name: str = "general_kb") -> QdrantIndexer:
    """获取 Qdrant 索引器"""
    if collection_name not in _indexers:
        _indexers[collection_name] = QdrantIndexer(collection_name)
    return _indexers[collection_name]
