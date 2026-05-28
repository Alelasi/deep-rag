"""文档索引器 — 分块+向量化+BM25索引"""
import hashlib
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 支持的文件类型
SUPPORTED_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html"}

# 分块配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


class Indexer:
    """文档索引管理"""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self._collection = None
        self._bm25 = None
        self._bm25_docs = []   # BM25对应的原始文档列表
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "],
        )

    def _get_collection(self):
        """获取或创建ChromaDB集合"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except Exception:
                self._collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        return self._collection

    def index_directory(self, dir_path: str) -> int:
        """索引整个目录，返回索引的块数"""
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

        # ChromaDB向量索引
        collection = self._get_collection()
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            collection.add(
                documents=[c["content"] for c in batch],
                ids=[c["doc_id"] for c in batch],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
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
            collection.add(
                documents=[c["content"] for c in chunks],
                ids=[c["doc_id"] for c in chunks],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
            )
            self._build_bm25(chunks)

        return len(chunks)

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
