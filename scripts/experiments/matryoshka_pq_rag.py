"""Matryoshka + PQ RAG - 极致压缩方案

组合方案：
1. Matryoshka: 384维 → 64维（6:1）
2. PQ量化: 64维 → 8 bytes（32:1）
3. 总压缩比: 6 × 32 = 192:1

15 GB 文本 → ~80 MB 索引
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.pq_compressor import PQCompressor


class MatryoshkaPQRAG:
    """Matryoshka + PQ 极致压缩RAG"""

    def __init__(
        self,
        db_path: str = "data/matryoshka_pq_rag",
        matryoshka_dim: int = 64,
        num_subvectors: int = 8,
    ):
        """初始化

        Args:
            db_path: 数据库路径
            matryoshka_dim: Matryoshka目标维度
            num_subvectors: PQ子向量数量
        """
        self.db_path = Path(db_path)
        self.matryoshka_dim = matryoshka_dim
        self.num_subvectors = num_subvectors

        self.embedder = None
        self.pq = PQCompressor(
            dim=matryoshka_dim, num_subvectors=num_subvectors, num_clusters=256
        )
        self.codes = None  # PQ codes
        self.metadata = {"chunks": {}, "id_to_index": {}}

    def _init_embedder(self):
        """初始化embedding模型"""
        if self.embedder is None:
            print("Loading embedding model (device=cuda)...")
            self.embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cuda"
            )
            print("✅ Embedding model loaded on cuda")

    def scan_files(
        self, root_dir: str, extensions: List[str] = None
    ) -> List[Path]:
        """扫描文件"""
        if extensions is None:
            extensions = [
                ".md",
                ".txt",
                ".py",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".html",
            ]

        SENSITIVE_PATTERNS = [
            "**/简历/**",
            "**/.env*",
            "**/credentials*",
            "**/secrets*",
            "**/投递清单*",
            "**/Tencent Files/**",
        ]

        root = Path(root_dir)
        files = []

        print(f"Scanning files in {root_dir}...")
        for ext in extensions:
            for file in root.rglob(f"*{ext}"):
                is_sensitive = any(
                    file.match(pattern) for pattern in SENSITIVE_PATTERNS
                )
                if not is_sensitive:
                    files.append(file)

        print(f"✅ Found {len(files)} files")
        return files

    def chunk_file(
        self, file_path: Path, chunk_size: int = 500
    ) -> List[Dict]:
        """分块文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Failed to read {file_path}: {e}")
            return []

        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i : i + chunk_size]
            if len(chunk_text.strip()) < 50:
                continue

            chunk_id = hashlib.md5(f"{file_path}:{i}".encode()).hexdigest()

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "file_path": str(file_path),
                        "chunk_index": i // chunk_size,
                    },
                }
            )

        return chunks

    def index_directory(
        self,
        root_dir: str,
        extensions: List[str] = None,
        chunk_size: int = 500,
    ):
        """索引目录（Matryoshka + PQ）"""
        self._init_embedder()
        self.db_path.mkdir(parents=True, exist_ok=True)

        # 扫描文件
        files = self.scan_files(root_dir, extensions)

        # 分块
        print("Chunking files...")
        all_chunks = []
        for file_path in tqdm(files, desc="Chunking"):
            chunks = self.chunk_file(file_path, chunk_size)
            all_chunks.extend(chunks)

        print(f"✅ Total chunks: {len(all_chunks)}")

        # Step 1: Embedding + Matryoshka降维
        print(
            f"Step 1: Embedding + Matryoshka ({self.matryoshka_dim}D)..."
        )
        batch_size = 512  # 平衡：RTX 4060 8GB 安全值
        all_vectors = []

        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
            batch = all_chunks[i : i + batch_size]
            texts = [chunk["text"] for chunk in batch]

            # GPU批量向量化
            vectors_384 = self.embedder.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                device="cuda",
            )

            # Matryoshka降维
            vectors_reduced = vectors_384[:, : self.matryoshka_dim]
            all_vectors.append(vectors_reduced)

        all_vectors = np.vstack(all_vectors).astype(np.float32)
        print(f"✅ Vectors shape: {all_vectors.shape}")

        # Step 2: 训练PQ
        print(f"\nStep 2: Training PQ ({self.num_subvectors} subvectors)...")
        # 使用前10万个向量训练（足够）
        train_size = min(100000, len(all_vectors))
        self.pq.train(all_vectors[:train_size])

        # Step 3: PQ编码
        print("\nStep 3: PQ encoding...")
        self.codes = self.pq.encode(all_vectors)
        print(f"✅ PQ codes shape: {self.codes.shape}")

        # Step 4: 保存metadata
        print("\nStep 4: Saving metadata...")
        for idx, chunk in enumerate(all_chunks):
            self.metadata["chunks"][chunk["id"]] = {
                "text": chunk["text"][:200],  # 只存前200字符
                "metadata": chunk["metadata"],
            }
            self.metadata["id_to_index"][chunk["id"]] = idx

        self._save_all()

        # 统计
        original_size = len(all_chunks) * 384 * 4
        compressed_size = self.codes.nbytes

        print(f"\n✅ Indexing complete!")
        print(f"  - Total chunks: {len(all_chunks):,}")
        print(f"  - Dimension: 384 → {self.matryoshka_dim}")
        print(f"  - Original size: {original_size / 1024 / 1024:.1f} MB")
        print(f"  - Compressed size: {compressed_size / 1024:.1f} KB")
        print(
            f"  - Compression ratio: {original_size / compressed_size:.0f}:1"
        )

    def _save_all(self):
        """保存所有数据"""
        # 保存PQ codebooks
        pq_path = self.db_path / "pq_codebooks.pkl"
        with open(pq_path, "wb") as f:
            pickle.dump(self.pq.codebooks, f)

        # 保存PQ codes
        codes_path = self.db_path / "pq_codes.npy"
        np.save(codes_path, self.codes)

        # 保存metadata
        metadata_path = self.db_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ Data saved to {self.db_path}")

    def _load_all(self):
        """加载所有数据"""
        # 加载PQ codebooks
        pq_path = self.db_path / "pq_codebooks.pkl"
        with open(pq_path, "rb") as f:
            self.pq.codebooks = pickle.load(f)
        self.pq.trained = True

        # 加载PQ codes
        codes_path = self.db_path / "pq_codes.npy"
        self.codes = np.load(codes_path)

        # 加载metadata
        metadata_path = self.db_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"✅ Data loaded from {self.db_path}")
        print(f"  - Chunks: {len(self.metadata['chunks']):,}")
        print(f"  - PQ codes: {self.codes.shape}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """搜索"""
        self._init_embedder()
        self._load_all()

        # 向量化查询 + 降维
        query_vector_384 = self.embedder.encode(
            query, convert_to_numpy=True
        )
        query_vector = query_vector_384[: self.matryoshka_dim].astype(
            np.float32
        )

        # PQ搜索
        distances, indices = self.pq.search(query_vector, self.codes, top_k)

        # 格式化结果
        results = []
        for dist, idx in zip(distances, indices):
            # 通过索引找chunk_id
            chunk_id = None
            for cid, cidx in self.metadata["id_to_index"].items():
                if cidx == idx:
                    chunk_id = cid
                    break

            if chunk_id and chunk_id in self.metadata["chunks"]:
                chunk_data = self.metadata["chunks"][chunk_id]
                results.append(
                    {
                        "text": chunk_data["text"],
                        "score": float(dist),
                        "metadata": chunk_data["metadata"],
                    }
                )

        return results


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Matryoshka + PQ RAG - 极致压缩"
    )
    parser.add_argument(
        "command", choices=["index", "search"], help="命令"
    )
    parser.add_argument("--dir", help="索引目录")
    parser.add_argument("--query", help="搜索查询")
    parser.add_argument(
        "--matryoshka-dim", type=int, default=64, help="Matryoshka维度"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="块大小"
    )
    parser.add_argument("--top-k", type=int, default=10, help="返回结果数")

    args = parser.parse_args()

    rag = MatryoshkaPQRAG(matryoshka_dim=args.matryoshka_dim)

    if args.command == "index":
        if not args.dir:
            print("❌ --dir required for index command")
            return
        rag.index_directory(args.dir, chunk_size=args.chunk_size)

    elif args.command == "search":
        if not args.query:
            print("❌ --query required for search command")
            return
        results = rag.search(args.query, top_k=args.top_k)
        print(f"\n🔍 Search results for: {args.query}\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text'][:100]}...")
            print(f"   Score: {result['score']:.4f}")
            print(f"   File: {result['metadata']['file_path']}\n")


if __name__ == "__main__":
    main()
