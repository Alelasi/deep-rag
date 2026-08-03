"""文档索引器 — 分块+向量化+BM25索引（v2.9.1: Token感知分块）

索引构建加固（子 agent L）：
- 批量化 upsert（batch_size 可配置，默认 100），减少网络往返。
- 有界并发文件读取（ThreadPoolExecutor，max_workers 可配置，默认 4）；
  嵌入与写入保持单线程顺序执行（SentenceTransformer 模型本身非线程安全）。
- ChromaDB 持久化安全：本文件只通过 src.config.get_chroma_client() 取得 HttpClient，
  严禁在已有库路径创建 chromadb.PersistentClient（会导致 HNSW 索引大规模损坏，见 CLAUDE.md）。
  请勿在本文件或任何位置引入 PersistentClient 直连。
"""
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from src.logging_config import get_logger
except Exception:
    import logging
    def get_logger(n):  # type: ignore
        return logging.getLogger(n)
logger = get_logger(__name__)

from ..config import CHROMA_DB_PATH, EMBEDDING_MODEL, DEVICE, get_chroma_client


# 支持的文件类型
SUPPORTED_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html"}

# 分块配置（v2.9.1: 从字符级改为token级）
# 500 token ≈ 750-1000 中文字符，预留上下文空间给System Prompt和用户问题
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 索引构建加固默认值
DEFAULT_BATCH_SIZE = 100          # 单批写入条数（减少网络往返）
DEFAULT_MAX_WORKERS = 4           # 文件读取/分块的最大并发线程数（有界）
_MAX_RETRIES = 3                  # 单批写入失败重试上限

# Token计数器：优先用tiktoken，降级到字符计数
_tiktoken_encoder = None

# 保护元数据拼接等共享状态的锁（嵌入/写入为单线程，此处仅防御性）
_meta_lock = threading.Lock()

def _get_token_encoder():
    """获取token编码器（懒加载，全局缓存）"""
    global _tiktoken_encoder
    if _tiktoken_encoder is not None:
        return _tiktoken_encoder
    try:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")  # GPT/Qwen通用编码
        logger.info("[Indexer] 使用 tiktoken 进行 token 级分块")
    except ImportError:
        _tiktoken_encoder = False  # 标记为不可用
        logger.warning("[Indexer] tiktoken 未安装，降级为字符级分块（1字符≈1token估算）")
    return _tiktoken_encoder

def _token_length(text: str) -> int:
    """Token长度计算函数（供 RecursiveCharacterTextSplitter 使用）"""
    encoder = _get_token_encoder()
    if encoder:
        return len(encoder.encode(text))
    # 降级：中文字符≈1.5token，ASCII≈0.25token，取加权平均
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_count = len(text) - cjk_count
    return int(cjk_count * 1.5 + ascii_count * 0.25)


class Indexer:
    """文档索引管理（v2.6: 支持子集合）"""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.client = get_chroma_client()
        self.collection_name = collection_name
        self._collection = None
        self._embedder = None
        self._bm25 = None
        self._bm25_docs = []   # BM25对应的原始文档列表
        self._sub_collections = None  # v2.6: 子集合缓存
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "],
            length_function=_token_length,  # v2.9.1: token级分块
        )

    def _get_embedder(self):
        """获取 SentenceTransformer（通过全局model_cache缓存）"""
        if self._embedder is None:
            from src.ui.model_cache import get_embedding_model
            self._embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)
        return self._embedder

    def _get_collection(self):
        """获取或创建ChromaDB集合（v2.6: 支持子集合）

        注意：self.client 来自 get_chroma_client()（HttpClient 服务器模式），
        严禁在此处或任何位置对已存在库路径创建 PersistentClient。
        """
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except Exception:
                # v2.6: 尝试子集合 (如 work_1, work_2, work_3)
                subs = self._find_sub_collections()
                if subs:
                    self._collection = subs[0]  # 用第一个子集合作为主集合
                    self._sub_collections = subs
                    logger.info(f"[Indexer] 使用 {len(subs)} 个子集合: {[s.name for s in subs]}")
                else:
                    self._collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
        return self._collection

    def _find_sub_collections(self):
        """v2.6: 查找子集合 (如 work_1, work_2, work_3)"""
        subs = []
        idx = 1
        while idx <= 20:  # 最多查找20个子集合
            try:
                col = self.client.get_collection(f"{self.collection_name}_{idx}")
                subs.append(col)
                idx += 1
            except Exception:
                break
        return subs if subs else None

    def get_all_collections(self) -> list:
        """v2.6: 获取所有集合（主集合 + 子集合）

        用于向量检索时查询所有子集合并合并结果。
        """
        if self._sub_collections is not None:
            return self._sub_collections

        # 确保主集合已初始化
        self._get_collection()

        if self._sub_collections is not None:
            return self._sub_collections
        else:
            # 没有子集合，返回主集合
            return [self._collection] if self._collection else []

    def is_already_indexed(self) -> bool:
        """检查集合是否已有数据（避免重复索引）"""
        try:
            collection = self._get_collection()
            return collection.count() > 0
        except Exception:
            return False

    def _read_file_chunks(self, filepath: Path, base_path: Path) -> List[dict]:
        """读取单个文件并分块（线程安全：仅做 I/O 与无状态分块）。

        配合有界线程池用于并行文件读取/分块；嵌入与写入不在此函数内。
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
            logger.warning(f"[Indexer] 读取/分块失败，跳过 {filepath}: {e}")
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

    def _add_chunks(self, collection, chunks: List[dict], batch_size: int) -> int:
        """按批次写入分块到 ChromaDB，单批失败按 _MAX_RETRIES 重试。

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
                logger.error(f"[Indexer] 批次嵌入失败（跳过 {len(batch)} 块）: {e}")
                raise
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    collection.add(
                        documents=[c["content"] for c in batch],
                        ids=[c["doc_id"] for c in batch],
                        metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
                        embeddings=embeddings,
                    )
                    added += len(batch)
                    break
                except Exception as e:
                    if attempt >= _MAX_RETRIES:
                        logger.error(f"[Indexer] 批次写入失败，重试耗尽，已写 {added}/{total}: {e}")
                        raise
                    logger.warning(f"[Indexer] 批次写入失败，第 {attempt} 次重试: {e}")
                    time.sleep(min(2 ** attempt, 10))
            logger.info(f"[Indexer] 已提交 {min(i + batch_size, total)}/{total} 块")
        return added

    def index_directory(
        self,
        dir_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> int:
        """索引整个目录，返回索引的块数

        Args:
            dir_path: 待索引目录。
            batch_size: 单批写入条数（默认 100）。
            max_workers: 文件读取/分块的最大并发线程数（默认 4，有界）。
        """
        # 跳过已索引的集合
        if self.is_already_indexed():
            count = self._get_collection().count()
            logger.info(f"[Indexer] 集合 '{self.collection_name}' 已有 {count} 条记录，跳过索引")
            return 0

        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        logger.info(f"[Indexer] 开始索引目录 {dir_path} (batch={batch_size}, workers={max_workers})")
        all_chunks = self._collect_chunks(path, max_workers)

        if not all_chunks:
            logger.info(f"[Indexer] 目录 {dir_path} 无可用分块，跳过")
            return 0

        # ChromaDB向量索引（显式传入 embedding，确保与检索器一致）
        collection = self._get_collection()
        added = self._add_chunks(collection, all_chunks, batch_size)

        # BM25索引
        self._build_bm25(all_chunks)

        logger.info(f"[Indexer] 索引完成: {added} 块")
        return added

    def index_texts(
        self,
        texts: List[dict],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> int:
        """索引文本列表 [{content, source, page}]

        Args:
            texts: 文本字典列表。
            batch_size: 单批写入条数（默认 100）。
        """
        collection = self._get_collection()
        chunks = []
        for i, t in enumerate(texts):
            doc_id = hashlib.md5(f"{t.get('source', 'text')}:{i}".encode()).hexdigest()[:12]
            chunks.append({
                "doc_id": doc_id,
                "content": t["content"],
                "source": t.get("source", "unknown"),
                "page": t.get("page", i + 1),
            })

        if chunks:
            self._add_chunks(collection, chunks, batch_size)
            self._build_bm25(chunks)

        return len(chunks)

    def update_document(
        self,
        doc_id: str,
        new_text: str,
        source: str = "updated",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """更新文档：先删后增（RAG 标准更新流程）"""
        collection = self._get_collection()
        try:
            collection.delete(ids=[doc_id])
        except Exception:
            pass
        embedder = self._get_embedder()
        embedding = embedder.encode([new_text]).tolist()
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                collection.add(
                    documents=[new_text], ids=[doc_id],
                    metadatas=[{"source": source, "page": 0}],
                    embeddings=embedding,
                )
                logger.info(f"[Indexer] 文档更新完成: {doc_id}")
                break
            except Exception as e:
                if attempt >= _MAX_RETRIES:
                    logger.error(f"[Indexer] 文档更新失败（重试耗尽）: {doc_id}: {e}")
                    raise
                logger.warning(f"[Indexer] 文档更新失败，第 {attempt} 次重试: {doc_id}: {e}")
                time.sleep(min(2 ** attempt, 10))

    def index_directory_with_hash(
        self,
        dir_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict:
        """带 hash 去重的索引，返回 {added, updated, skipped}

        Args:
            dir_path: 待索引目录。
            batch_size: 单批写入条数（默认 100）。
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        collection = self._get_collection()
        # 获取已有文档的 hash 映射
        existing = {}
        try:
            stored = collection.get(include=["metadatas"])
            for i, mid in enumerate(stored["ids"]):
                meta = stored["metadatas"][i] if i < len(stored["metadatas"]) else {}
                if "content_hash" in meta:
                    existing[mid] = meta["content_hash"]
        except Exception:
            pass

        stats = {"added": 0, "updated": 0, "skipped": 0}
        embedder = self._get_embedder()

        # 累积待写入批次（删除仍按块即时执行，开销低）
        pending_docs: List[str] = []
        pending_embeddings: List[list] = []
        pending_metas: List[dict] = []
        pending_ids: List[str] = []

        def _flush_pending() -> None:
            if not pending_ids:
                return
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    collection.add(
                        documents=list(pending_docs),
                        ids=list(pending_ids),
                        metadatas=list(pending_metas),
                        embeddings=list(pending_embeddings),
                    )
                    break
                except Exception as e:
                    if attempt >= _MAX_RETRIES:
                        logger.error(f"[Indexer] 批次写入失败（去重索引），重试耗尽: {e}")
                        raise
                    logger.warning(f"[Indexer] 批次写入失败（去重索引），第 {attempt} 次重试: {e}")
                    time.sleep(min(2 ** attempt, 10))
            logger.info(f"[Indexer] 已提交去重批次 {len(pending_ids)} 块")

        for filepath in sorted(path.rglob("*")):
            if filepath.suffix.lower() not in SUPPORTED_EXTS:
                continue
            if filepath.name.startswith("."):
                continue
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if len(content) < 20:
                    continue

                content_hash = hashlib.md5(content.encode()).hexdigest()
                chunks = self._splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc_id = hashlib.md5(f"{filepath}:{i}".encode()).hexdigest()[:12]
                    old_hash = existing.get(doc_id)
                    if old_hash == content_hash:
                        stats["skipped"] += 1
                        continue
                    if old_hash:
                        try:
                            collection.delete(ids=[doc_id])
                        except Exception:
                            pass
                        stats["updated"] += 1
                    else:
                        stats["added"] += 1
                    with _meta_lock:
                        pending_docs.append(chunk)
                        pending_embeddings.append(embedder.encode([chunk]).tolist()[0])
                        pending_metas.append({
                            "source": str(filepath.relative_to(path)),
                            "page": i + 1,
                            "content_hash": content_hash,
                        })
                        pending_ids.append(doc_id)
                    if len(pending_ids) >= batch_size:
                        _flush_pending()
                        pending_docs, pending_embeddings, pending_metas, pending_ids = [], [], [], []
            except Exception:
                continue

        _flush_pending()
        logger.info(f"[Indexer] 去重索引完成: {stats}")
        return stats

    def _build_bm25(self, chunks: List[dict]) -> None:
        """构建BM25索引"""
        from rank_bm25 import BM25Okapi
        import jieba

        self._bm25_docs = chunks
        corpus = [list(jieba.cut(c["content"])) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def get_bm25(self):
        return self._bm25, self._bm25_docs

    def get_collection(self):
        return self._get_collection()

    def clear(self):
        """清空索引"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._bm25 = None
        self._bm25_docs = []
