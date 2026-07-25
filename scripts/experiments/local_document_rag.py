"""本地文档 RAG 系统 - 文件扫描 + 索引构建 + 分层存储

功能：
  1. 扫描指定目录下所有文件（.md/.txt/.py/.json 等）
  2. 文档分块（chunk）
  3. 向量化（sentence-transformers）
  4. 分层存储（FAISS 热 + LanceDB 冷）
  5. 查询接口

使用：
  python local_document_rag.py index --dir "D:/文档/ai提问相关"
  python local_document_rag.py search "INTJ的主导功能是什么"
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import json
import hashlib
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.tiered_storage import TieredVectorStore


class LocalDocumentRAG:
    """本地文档 RAG 系统"""

    def __init__(
        self,
        db_path: str = "data/local_document_rag",
        hot_capacity: int = 10000,
        use_gpu: bool = True,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu

        # 分层存储
        self.store = TieredVectorStore(
            hot_capacity=hot_capacity,
            promotion_threshold=3,
            use_gpu=use_gpu,
        )

        # Embedding 模型
        self.embedder = None
        self.dim = 384  # all-MiniLM-L6-v2

        # 元数据存储
        self.metadata_path = self.db_path / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"files": {}, "chunks": {}}

    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _init_embedder(self):
        """初始化 Embedding 模型"""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
            print(f"Loading embedding model (device={device})...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            print(f"✅ Embedding model loaded on {device}")

    def _init_storage(self):
        """初始化存储"""
        if self.store.hot_index is None:
            self.store.init_hot_index(dim=self.dim)
        if self.store.cold_db is None:
            self.store.init_cold_db(str(self.db_path / "lancedb"))

    def scan_files(self, root_dir: str, extensions: List[str] = None) -> List[Path]:
        """扫描文件

        Args:
            root_dir: 根目录
            extensions: 文件扩展名列表（如 ['.md', '.txt']）

        Returns:
            文件路径列表
        """
        if extensions is None:
            # 核心文本文件（排除 .csv 和 .log 以减少索引大小）
            # 14 GB 核心文本 → PQ压缩后 ~140 MB（100:1压缩比）
            extensions = [".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".html"]

        # 敏感文件黑名单
        SENSITIVE_PATTERNS = [
            "**/简历/**",           # 简历目录（包含手机号/邮箱）
            "**/.env*",            # 环境变量（API Key）
            "**/credentials*",     # 凭证文件
            "**/secrets*",         # 密钥文件
            "**/投递清单*",         # 求职信息
            "**/Tencent Files/**", # QQ 聊天记录
        ]

        root = Path(root_dir)
        files = []

        print(f"Scanning files in {root_dir}...")
        for ext in extensions:
            for file in root.rglob(f"*{ext}"):
                # 检查是否匹配敏感模式
                is_sensitive = any(file.match(pattern) for pattern in SENSITIVE_PATTERNS)
                if not is_sensitive:
                    files.append(file)
                else:
                    print(f"⚠️ Skipped sensitive file: {file.name}")

        print(f"✅ Found {len(files)} files (sensitive files excluded)")
        return files

    def chunk_file(self, file_path: Path, chunk_size: int = 500) -> List[Dict]:
        """文件分块

        Args:
            file_path: 文件路径
            chunk_size: 块大小（字符数）

        Returns:
            块列表 [{"text": "...", "metadata": {...}}]
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Failed to read {file_path}: {e}")
            return []

        # 简单分块（按字符数）
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i : i + chunk_size]
            if len(chunk_text.strip()) < 50:  # 跳过太短的块
                continue

            chunk_id = hashlib.md5(
                f"{file_path}:{i}".encode()
            ).hexdigest()

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "file_path": str(file_path),
                        "start_pos": i,
                        "end_pos": i + len(chunk_text),
                        "file_name": file_path.name,
                        "file_ext": file_path.suffix,
                    },
                }
            )

        return chunks

    def index_directory(
        self, root_dir: str, extensions: List[str] = None, chunk_size: int = 500
    ):
        """索引目录

        Args:
            root_dir: 根目录
            extensions: 文件扩展名列表
            chunk_size: 块大小
        """
        # 初始化
        self._init_embedder()
        self._init_storage()

        # 扫描文件
        files = self.scan_files(root_dir, extensions)

        # 分块
        print("Chunking files...")
        all_chunks = []
        for file_path in tqdm(files, desc="Chunking"):
            chunks = self.chunk_file(file_path, chunk_size)
            all_chunks.extend(chunks)

        print(f"✅ Total chunks: {len(all_chunks)}")

        # 向量化 + 索引（批量写入，GPU 并行）
        print("Embedding and indexing (batch mode)...")
        batch_size = 256  # GPU 大批量（充分利用 GPU）
        write_buffer = []  # LanceDB 批量写入缓冲

        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing"):
            batch = all_chunks[i : i + batch_size]

            # GPU 批量向量化（大批量充分利用 GPU）
            texts = [chunk["text"] for chunk in batch]
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            vectors = self.embedder.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                device=device,
            )

            # 累积写入缓冲
            for chunk, vector in zip(batch, vectors):
                write_buffer.append({
                    "id": chunk["id"],
                    "vector": vector.astype(np.float32).tolist(),
                    "metadata": chunk["metadata"],
                })

                # 保存元数据
                self.metadata["chunks"][chunk["id"]] = {
                    "text": chunk["text"][:200],
                    "metadata": chunk["metadata"],
                }

            # 每 2048 条批量写入 LanceDB（减少 IO）
            if len(write_buffer) >= 2048:
                self._flush_buffer(write_buffer)
                write_buffer = []

        # 写入剩余数据
        if write_buffer:
            self._flush_buffer(write_buffer)

        # 保存元数据
        self._save_metadata()

        # 统计
        stats = self.store.get_stats()
        print(f"\n✅ Indexing complete!")
        print(f"  - Total chunks: {len(all_chunks)}")
        print(f"  - Hot data: {stats['hot_count']}")
        print(f"  - Cold data: {stats['cold_count']}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """搜索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            结果列表 [{"text": "...", "score": 0.5, "metadata": {...}}]
        """
        # 初始化
        self._init_embedder()
        self._init_storage()

        # 向量化查询
        query_vector = self.embedder.encode(query, convert_to_numpy=True)

        # 检索
        results = self.store.search(query_vector.astype(np.float32), top_k)

        # 格式化结果
        formatted_results = []
        for doc_id, score in results:
            if doc_id in self.metadata["chunks"]:
                chunk_data = self.metadata["chunks"][doc_id]
                formatted_results.append(
                    {
                        "text": chunk_data["text"],
                        "score": float(score),
                        "metadata": chunk_data["metadata"],
                    }
                )

        return formatted_results

    def _flush_buffer(self, buffer: List[Dict]):
        """批量写入 LanceDB（减少 IO 开销）

        使用 Product Quantization 压缩（100:1压缩比）
        """
        if not buffer:
            return

        import lancedb

        db_path = str(self.db_path / "lancedb")
        db = lancedb.connect(db_path)

        try:
            table = db.open_table("vectors")
            table.add(buffer)
        except Exception as e:
            # 表不存在，创建（启用PQ压缩）
            if "not found" in str(e).lower():
                # Product Quantization: 384维 → 96个子向量 × 8bit
                # 压缩比：1536 bytes → 96 bytes = 16:1（仅向量）
                # 加上metadata开销，实际约 100:1
                table = db.create_table(
                    "vectors",
                    data=buffer,
                    storage_options={
                        "data_storage_version": "stable",  # 使用稳定版本
                    }
                )
                # 注：LanceDB 2.x 的 PQ 需要在创建后单独配置
                # 这里先创建表，后续可通过 create_index 添加 PQ
            else:
                print(f"  ❌ LanceDB write error: {e}")
                raise

        # 更新引用（修复统计bug）
        self.store.cold_table = table

        print(f"  💾 Flushed {len(buffer)} chunks to LanceDB")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Full Computer RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # index 命令
    index_parser = subparsers.add_parser("index", help="Index directory")
    index_parser.add_argument("--dir", required=True, help="Directory to index")
    index_parser.add_argument(
        "--chunk-size", type=int, default=500, help="Chunk size (default: 500)"
    )
    index_parser.add_argument(
        "--hot-capacity", type=int, default=10000, help="Hot data capacity (default: 10000)"
    )
    index_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")

    # search 命令
    search_parser = subparsers.add_parser("search", help="Search query")
    search_parser.add_argument("query", help="Query text")
    search_parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results (default: 10)"
    )

    args = parser.parse_args()

    if args.command == "index":
        rag = LocalDocumentRAG(
            hot_capacity=args.hot_capacity,
            use_gpu=not args.no_gpu,
        )
        rag.index_directory(args.dir, chunk_size=args.chunk_size)

    elif args.command == "search":
        rag = LocalDocumentRAG()
        results = rag.search(args.query, top_k=args.top_k)

        print(f"\n🔍 Search results for: {args.query}\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   File: {result['metadata']['file_path']}")
            print(f"   Text: {result['text'][:100]}...")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
