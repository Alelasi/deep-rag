"""Binary Quantization（二值量化）实现

最极致的压缩方法：
- 压缩比：32:1（float32 → 1 bit）
- 召回率损失：10-15%
- 速度提升：10-50x（汉明距离）
- 内存节省：96.875%

适用场景：
- 第一阶段粗筛（召回 top-1000，再精排 top-10）
- 超大规模数据（>1B 向量）
"""

import numpy as np
from typing import Tuple, Optional
import logging

log = logging.getLogger("deeprag.binary_quantization")


class BinaryQuantizer:
    """二值量化器

    将 float32 向量压缩为 1 bit，节省 96.875% 内存

    原理：
    1. 每个维度 > 阈值 → 1，否则 → 0
    2. 打包为 uint8（每 8 维 = 1 字节）
    3. 使用汉明距离搜索（XOR + popcount）
    """

    def __init__(self, dim: int, threshold: float = 0.0):
        """初始化

        Args:
            dim: 向量维度（必须是 8 的倍数）
            threshold: 二值化阈值（默认 0）
        """
        if dim % 8 != 0:
            raise ValueError(f"dim must be multiple of 8, got {dim}")

        self.dim = dim
        self.threshold = threshold
        self.packed_dim = dim // 8  # 打包后的维度

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """编码为二值向量

        Args:
            vectors: 输入向量 (N, dim) float32

        Returns:
            二值向量 (N, packed_dim) uint8
        """
        # 二值化
        binary = (vectors > self.threshold).astype(np.uint8)

        # 打包为 uint8（每 8 位 = 1 字节）
        packed = np.packbits(binary, axis=1)

        return packed

    def decode(self, packed: np.ndarray) -> np.ndarray:
        """解码为 float32（0 或 1）

        Args:
            packed: 二值向量 (N, packed_dim) uint8

        Returns:
            解码向量 (N, dim) float32
        """
        # 解包
        binary = np.unpackbits(packed, axis=1, count=self.dim)

        return binary.astype(np.float32)

    def hamming_distance(self, query_packed: np.ndarray, db_packed: np.ndarray) -> np.ndarray:
        """计算汉明距离（超快）

        Args:
            query_packed: 查询向量 (packed_dim,) uint8
            db_packed: 数据库向量 (N, packed_dim) uint8

        Returns:
            汉明距离 (N,)
        """
        # XOR + popcount
        xor = query_packed ^ db_packed
        distances = np.unpackbits(xor, axis=1).sum(axis=1)

        return distances

    def cosine_similarity_approx(self, query_packed: np.ndarray, db_packed: np.ndarray) -> np.ndarray:
        """近似余弦相似度

        汉明距离可以近似余弦距离：
        cosine_sim ≈ 1 - 2 * hamming_distance / dim

        Args:
            query_packed: 查询向量 (packed_dim,) uint8
            db_packed: 数据库向量 (N, packed_dim) uint8

        Returns:
            近似余弦相似度 (N,)
        """
        hamming_dist = self.hamming_distance(query_packed, db_packed)
        cosine_sim = 1.0 - 2.0 * hamming_dist / self.dim

        return cosine_sim

    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        return 32.0  # float32 (4 bytes) → 1 bit (1/8 byte)

    def get_memory_saving(self, num_vectors: int) -> dict:
        """计算内存节省

        Args:
            num_vectors: 向量数量

        Returns:
            内存统计（MB）
        """
        original_mb = num_vectors * self.dim * 4 / 1024 / 1024
        binary_mb = num_vectors * self.packed_dim * 1 / 1024 / 1024

        saved_mb = original_mb - binary_mb

        return {
            "original_mb": original_mb,
            "binary_mb": binary_mb,
            "saved_mb": saved_mb,
            "saving_ratio": saved_mb / original_mb,
        }


class BinaryQuantizedIndex:
    """基于 Binary Quantization 的向量索引

    适合第一阶段粗筛：
    1. Binary 快速召回 top-1000（汉明距离）
    2. Float32 精确重排 top-10（余弦距离）
    """

    def __init__(self, dim: int):
        """初始化

        Args:
            dim: 向量维度（必须是 8 的倍数）
        """
        self.dim = dim
        self.quantizer = BinaryQuantizer(dim)

        self.binary_vectors = None  # (N, packed_dim) uint8
        self.original_vectors = None  # (N, dim) float32（可选，用于重排）
        self.ids = None  # (N,) 向量 ID

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None, keep_original: bool = True):
        """添加向量

        Args:
            vectors: 向量 (N, dim) float32
            ids: 向量 ID (N,)
            keep_original: 是否保留原始向量（用于重排）
        """
        if ids is None:
            start_id = 0 if self.ids is None else len(self.ids)
            ids = np.arange(start_id, start_id + len(vectors))

        # 二值化
        binary = self.quantizer.encode(vectors)

        # 存储
        if self.binary_vectors is None:
            self.binary_vectors = binary
            self.ids = ids
            if keep_original:
                self.original_vectors = vectors
        else:
            self.binary_vectors = np.vstack([self.binary_vectors, binary])
            self.ids = np.concatenate([self.ids, ids])
            if keep_original:
                if self.original_vectors is None:
                    self.original_vectors = vectors
                else:
                    self.original_vectors = np.vstack([self.original_vectors, vectors])

        log.info(f"Added {len(vectors)} vectors (binary quantized)")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        rerank: bool = True,
        rerank_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最近邻

        Args:
            query: 查询向量 (dim,) float32
            k: 返回 top-k
            rerank: 是否重排（需要 keep_original=True）
            rerank_k: 重排候选数（建议 10-100 倍 k）

        Returns:
            distances: 距离 (k,)
            ids: 向量 ID (k,)
        """
        if self.binary_vectors is None:
            return np.array([]), np.array([])

        # 阶段1：二值搜索（汉明距离）
        query_binary = self.quantizer.encode(query.reshape(1, -1))[0]

        if rerank and self.original_vectors is not None:
            # 召回 rerank_k 个候选
            hamming_distances = self.quantizer.hamming_distance(query_binary, self.binary_vectors)
            candidate_idx = np.argpartition(hamming_distances, min(rerank_k, len(hamming_distances)))[:rerank_k]

            # 阶段2：精确重排（余弦距离）
            candidate_vectors = self.original_vectors[candidate_idx]
            cosine_distances = 1 - np.dot(candidate_vectors, query)

            # 排序返回 top-k
            if len(cosine_distances) <= k:
                sorted_idx = np.argsort(cosine_distances)
                final_idx = candidate_idx[sorted_idx]
                return cosine_distances[sorted_idx], self.ids[final_idx]

            top_k_idx = np.argpartition(cosine_distances, k)[:k]
            top_k_idx = top_k_idx[np.argsort(cosine_distances[top_k_idx])]
            final_idx = candidate_idx[top_k_idx]

            return cosine_distances[top_k_idx], self.ids[final_idx]

        else:
            # 只用汉明距离
            hamming_distances = self.quantizer.hamming_distance(query_binary, self.binary_vectors)

            if len(hamming_distances) <= k:
                sorted_idx = np.argsort(hamming_distances)
                return hamming_distances[sorted_idx], self.ids[sorted_idx]

            top_k_idx = np.argpartition(hamming_distances, k)[:k]
            top_k_idx = top_k_idx[np.argsort(hamming_distances[top_k_idx])]

            return hamming_distances[top_k_idx], self.ids[top_k_idx]

    def get_stats(self) -> dict:
        """获取索引统计"""
        if self.binary_vectors is None:
            num_vectors = 0
        else:
            num_vectors = len(self.binary_vectors)

        memory_stats = self.quantizer.get_memory_saving(num_vectors)

        return {
            "dim": self.dim,
            "num_vectors": num_vectors,
            "has_original": self.original_vectors is not None,
            **memory_stats,
        }


def benchmark_binary_quantization(dim: int = 384, num_vectors: int = 100000):
    """Benchmark Binary Quantization"""
    import time

    print(f"\n{'='*80}")
    print(f"Binary Quantization Benchmark (dim={dim}, num_vectors={num_vectors:,})")
    print(f"{'='*80}")

    # 生成测试数据
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    # 创建量化器
    quantizer = BinaryQuantizer(dim)

    # 编码
    start = time.time()
    binary = quantizer.encode(vectors)
    encode_time = time.time() - start

    # 解码
    start = time.time()
    decoded = quantizer.decode(binary)
    decode_time = time.time() - start

    # 内存统计
    memory_stats = quantizer.get_memory_saving(num_vectors)

    # 打印结果
    print(f"\n编码时间: {encode_time:.3f}s ({num_vectors/encode_time:.0f} vectors/s)")
    print(f"解码时间: {decode_time:.3f}s ({num_vectors/decode_time:.0f} vectors/s)")
    print(f"\n内存统计:")
    print(f"  原始: {memory_stats['original_mb']:.1f} MB")
    print(f"  二值: {memory_stats['binary_mb']:.1f} MB")
    print(f"  节省: {memory_stats['saved_mb']:.1f} MB ({memory_stats['saving_ratio']:.1%})")
    print(f"  压缩比: {quantizer.get_compression_ratio():.1f}:1")

    # 搜索性能
    query_binary = quantizer.encode(query.reshape(1, -1))[0]

    # 原始余弦距离
    start = time.time()
    distances_cosine = 1 - np.dot(vectors, query)
    search_time_cosine = time.time() - start

    # 汉明距离
    start = time.time()
    distances_hamming = quantizer.hamming_distance(query_binary, binary)
    search_time_hamming = time.time() - start

    print(f"\n搜索性能:")
    print(f"  余弦距离 (float32): {search_time_cosine*1000:.2f}ms")
    print(f"  汉明距离 (binary): {search_time_hamming*1000:.2f}ms")
    print(f"  加速比: {search_time_cosine/search_time_hamming:.1f}x")

    # 召回率（汉明距离 vs 余弦距离）
    k = 10
    top_k_cosine = np.argpartition(distances_cosine, k)[:k]
    top_k_hamming = np.argpartition(distances_hamming, k)[:k]
    recall = len(set(top_k_cosine) & set(top_k_hamming)) / k

    print(f"\n召回率 @{k} (汉明 vs 余弦): {recall:.1%}")

    # 两阶段检索
    rerank_k = 100
    start = time.time()

    # 阶段1：汉明距离召回 top-100
    candidate_idx = np.argpartition(distances_hamming, rerank_k)[:rerank_k]

    # 阶段2：余弦距离重排 top-10
    candidate_vectors = vectors[candidate_idx]
    cosine_distances = 1 - np.dot(candidate_vectors, query)
    top_k_idx = np.argpartition(cosine_distances, k)[:k]
    final_idx = candidate_idx[top_k_idx]

    two_stage_time = time.time() - start

    recall_two_stage = len(set(top_k_cosine) & set(final_idx)) / k

    print(f"\n两阶段检索:")
    print(f"  时间: {two_stage_time*1000:.2f}ms")
    print(f"  召回率 @{k}: {recall_two_stage:.1%}")
    print(f"  加速比 vs 全量余弦: {search_time_cosine/two_stage_time:.1f}x")

    print(f"{'='*80}\n")


def compare_quantization_methods(dim: int = 384, num_vectors: int = 100000):
    """对比三种量化方法"""
    print(f"\n{'='*100}")
    print(f"量化方法对比 (dim={dim}, num_vectors={num_vectors:,})")
    print(f"{'='*100}")

    # 生成测试数据
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    # Ground truth
    distances_gt = 1 - np.dot(vectors, query)
    top_k_gt = set(np.argpartition(distances_gt, 10)[:10])

    results = []

    # 1. 无量化（Baseline）
    import time
    start = time.time()
    distances = 1 - np.dot(vectors, query)
    search_time = time.time() - start

    results.append({
        "method": "无量化 (Baseline)",
        "compression": "1:1",
        "memory_mb": num_vectors * dim * 4 / 1024 / 1024,
        "search_ms": search_time * 1000,
        "recall": 1.0,
    })

    # 2. Scalar Quantization
    from src.retrieval.scalar_quantization import ScalarQuantizer

    sq = ScalarQuantizer(dim)
    sq.train(vectors)
    vectors_sq = sq.encode(vectors)

    start = time.time()
    distances_sq = sq.compute_distance(query, vectors_sq, metric="cosine")
    search_time_sq = time.time() - start

    top_k_sq = set(np.argpartition(distances_sq, 10)[:10])
    recall_sq = len(top_k_gt & top_k_sq) / 10

    results.append({
        "method": "Scalar Quantization",
        "compression": "4:1",
        "memory_mb": num_vectors * dim * 1 / 1024 / 1024,
        "search_ms": search_time_sq * 1000,
        "recall": recall_sq,
    })

    # 3. Binary Quantization
    bq = BinaryQuantizer(dim)
    vectors_bq = bq.encode(vectors)
    query_bq = bq.encode(query.reshape(1, -1))[0]

    start = time.time()
    distances_bq = bq.hamming_distance(query_bq, vectors_bq)
    search_time_bq = time.time() - start

    top_k_bq = set(np.argpartition(distances_bq, 10)[:10])
    recall_bq = len(top_k_gt & top_k_bq) / 10

    results.append({
        "method": "Binary Quantization",
        "compression": "32:1",
        "memory_mb": num_vectors * dim / 8 / 1024 / 1024,
        "search_ms": search_time_bq * 1000,
        "recall": recall_bq,
    })

    # 打印表格
    print(f"{'方法':<25} {'压缩比':<10} {'内存(MB)':<12} {'搜索时间(ms)':<15} {'召回率@10':<12}")
    print(f"{'-'*100}")

    for r in results:
        print(f"{r['method']:<25} "
              f"{r['compression']:<10} "
              f"{r['memory_mb']:>10.1f} "
              f"{r['search_ms']:>13.2f} "
              f"{r['recall']:>10.1%}")

    print(f"{'='*100}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Binary Quantization 单独测试
    benchmark_binary_quantization(dim=384, num_vectors=100_000)

    # 三种方法对比
    compare_quantization_methods(dim=384, num_vectors=100_000)
