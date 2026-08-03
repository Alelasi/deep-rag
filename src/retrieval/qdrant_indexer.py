"""Qdrant 索引器 — 替代 ChromaDB Indexer

使用 Qdrant 向量数据库，索引持久化到磁盘，重启零损耗。

索引构建加固（子 agent L）：
- 批量化 upsert（batch_size 可配置，默认 100），减少网络往返。
- 有界并发文件读取（ThreadPoolExecutor，max_workers 可配置，默认 4）；
  嵌入与写入保持单线程顺序执行（SentenceTransformer 模型本身非线程安全）。
- 持久化安全：本项目默认生产后端为 Qdrant（Server 模式）。如需本地向量库，请用
  Qdrant Server 模式或 ChromaDB 服务器模式 + HttpClient；严禁对已存在库路径创建
  chromadb.PersistentClient（会导致 HNSW 索引大规模损坏，见 CLAUDE.md / DEVELOPMENT.md）。
"""
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import jieba
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from src.logging_config import get_logger
except Exception:
    import logging
    def get_logger(n):  # type: ignore
        return logging.getLogger(n)
logger = get_logger(__name__)

from .qdrant_retriever import QdrantRetriever, get_qdrant_retriever

# 支持的文件类型
SUPPORTED_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html"}

# 分块配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 索引构建加固默认值
DEFAULT_BATCH_SIZE = 100          # 单批写入条数（减少网络往返）
DEFAULT_MAX_WORKERS = 4           # 文件读取/分块的最大并发线程数（有界）
_MAX_RETRIES = 3                  # 单批写入失败重试上限


def _token_length(text: str) -> int:
    """Token 长度估算"""
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except ImportError:
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
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

    def _read_file_chunks(self, filepath: Path, base_path: Path) -> List[dict]:
        """读取单个文件并分块（线程安全：仅做 I/O 与无状态分块）。"""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if len(content) < 20:
                return []
            chunks = self._splitter.split_text(content)
            result = []
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{filepath}:{i}".encode()).hexdigest()[:12]
                result.append({
                    "doc_id": doc_id,
                    "content": chunk,
                    "source": str(filepath.relative_to(base_path)),
                    "page": i + 1,
                })
            return result
        except Exception as e:
            logger.warning(f"[QdrantIndexer] 读取/分块失败，跳过 {filepath}: {e}")
            return []

    def _collect_chunks(self, path: Path, max_workers: int) -> List[dict]:
        """遍历目录收集所有分块；max_workers>1 时有界并发读取文件。"""
        file_list = [
            fp for fp in sorted(path.rglob("*"))
            if fp.suffix.lower() in SUPPORTED_EXTS and not fp.name.startswith(".")
        ]
        all_chunks: List[dict] = []
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for res in ex.map(lambda fp: self._read_file_chunks(fp, path), file_list):
                    all_chunks.extend(res)
        else:
            for fp in file_list:
                all_chunks.extend(self._read_file_chunks(fp, path))
        return all_chunks

    def _add_chunks(self, chunks: List[dict], batch_size: int) -> int:
        """按批次写入分块到 Qdrant，单批失败按 _MAX_RETRIES 重试。

        嵌入与写入保持单线程顺序执行（SentenceTransformer 非线程安全）。
        返回成功写入的块数。
        """
        total = len(chunks)
        added = 0
        embedder = self._get_embedder()
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            try:
                embeddings = embedder.encode([c["content"] for c in batch]).tolist()
            except Exception as e:
                logger.error(f"[QdrantIndexer] 批次嵌入失败（跳过 {len(batch)} 块）: {e}")
                raise
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    self.retriever.add_documents(
                        documents=[c["content"] for c in batch],
                        embeddings=embeddings,
                        metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
                        ids=[c["doc_id"] for c in batch],
                    )
                    added += len(batch)
                    break
                except Exception as e:
                    if attempt >= _MAX_RETRIES:
                        logger.error(f"[QdrantIndexer] 批次写入失败，重试耗尽，已写 {added}/{total}: {e}")
                        raise
                    logger.warning(f"[QdrantIndexer] 批次写入失败，第 {attempt} 次重试: {e}")
                    time.sleep(min(2 ** attempt, 10))
            logger.info(f"[QdrantIndexer] 已提交 {min(i + batch_size, total)}/{total} 块")
        return added

    def index_directory(
        self,
        dir_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> int:
        """索引整个目录

        Args:
            dir_path: 待索引目录。
            batch_size: 单批写入条数（默认 100）。
            max_workers: 文件读取/分块的最大并发线程数（默认 4，有界）。
        """
        if self.is_already_indexed():
            count = self.retriever.count()
            logger.info(f"[QdrantIndexer] 已有 {count} 条记录，跳过索引")
            return 0

        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        logger.info(f"[QdrantIndexer] 开始索引目录 {dir_path} (batch={batch_size}, workers={max_workers})")
        all_chunks = self._collect_chunks(path, max_workers)

        if not all_chunks:
            logger.info(f"[QdrantIndexer] 目录 {dir_path} 无可用分块，跳过")
            return 0

        total_added = self._add_chunks(all_chunks, batch_size)

        # 构建 BM25 索引
        if all_chunks:
            self._build_bm25(all_chunks)

        logger.info(f"[QdrantIndexer] 索引完成: {total_added} 块")
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

    def _build_bm25(self, chunks: List[dict]) -> None:
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
