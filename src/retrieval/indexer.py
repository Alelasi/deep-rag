"""文档索引器 — 分块+向量化+BM25索引（v2.9.1: Token感知分块）"""
import hashlib
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

from ..config import CHROMA_DB_PATH, EMBEDDING_MODEL, DEVICE, get_chroma_client


# 支持的文件类型
SUPPORTED_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html"}

# 分块配置（v2.9.1: 从字符级改为token级）
# 500 token ≈ 750-1000 中文字符，预留上下文空间给System Prompt和用户问题
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Token计数器：优先用tiktoken，降级到字符计数
_tiktoken_encoder = None

def _get_token_encoder():
    """获取token编码器（懒加载，全局缓存）"""
    global _tiktoken_encoder
    if _tiktoken_encoder is not None:
        return _tiktoken_encoder
    try:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")  # GPT/Qwen通用编码
        log.info("[Indexer] 使用 tiktoken 进行 token 级分块")
    except ImportError:
        _tiktoken_encoder = False  # 标记为不可用
        log.warning("[Indexer] tiktoken 未安装，降级为字符级分块（1字符≈1token估算）")
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
        """获取或创建ChromaDB集合（v2.6: 支持子集合）"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except Exception:
                # v2.6: 尝试子集合 (如 work_1, work_2, work_3)
                subs = self._find_sub_collections()
                if subs:
                    self._collection = subs[0]  # 用第一个子集合作为主集合
                    self._sub_collections = subs
                    log.info(f"[Indexer] 使用 {len(subs)} 个子集合: {[s.name for s in subs]}")
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

    def index_directory(self, dir_path: str) -> int:
        """索引整个目录，返回索引的块数"""
        # 跳过已索引的集合
        if self.is_already_indexed():
            count = self._get_collection().count()
            log.info(f"[Indexer] 集合 '{self.collection_name}' 已有 {count} 条记录，跳过索引")
            return 0

        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_chunks = []
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

        # ChromaDB向量索引（显式传入 embedding，确保与检索器一致）
        collection = self._get_collection()
        embedder = self._get_embedder()
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = embedder.encode([c["content"] for c in batch]).tolist()
            collection.add(
                documents=[c["content"] for c in batch],
                ids=[c["doc_id"] for c in batch],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
                embeddings=embeddings,
            )

        # BM25索引
        self._build_bm25(all_chunks)

        return len(all_chunks)

    def index_texts(self, texts: list[dict]) -> int:
        """索引文本列表 [{content, source, page}]"""
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
            embedder = self._get_embedder()
            embeddings = embedder.encode([c["content"] for c in chunks]).tolist()
            collection.add(
                documents=[c["content"] for c in chunks],
                ids=[c["doc_id"] for c in chunks],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
                embeddings=embeddings,
            )
            self._build_bm25(chunks)

        return len(chunks)

    def update_document(self, doc_id: str, new_text: str, source: str = "updated"):
        """更新文档：先删后增（RAG 标准更新流程）"""
        collection = self._get_collection()
        try:
            collection.delete(ids=[doc_id])
        except Exception:
            pass
        embedder = self._get_embedder()
        embedding = embedder.encode([new_text]).tolist()
        collection.add(
            documents=[new_text], ids=[doc_id],
            metadatas=[{"source": source, "page": 0}],
            embeddings=embedding,
        )

    def index_directory_with_hash(self, dir_path: str) -> dict:
        """带 hash 去重的索引，返回 {added, updated, skipped}"""
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
                        collection.delete(ids=[doc_id])
                        stats["updated"] += 1
                    else:
                        stats["added"] += 1
                    embedding = embedder.encode([chunk]).tolist()
                    collection.add(
                        documents=[chunk], ids=[doc_id],
                        metadatas=[{
                            "source": str(filepath.relative_to(path)),
                            "page": i + 1,
                            "content_hash": content_hash,
                        }],
                        embeddings=embedding,
                    )
            except Exception:
                continue

        return stats

    def _build_bm25(self, chunks: list[dict]):
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
