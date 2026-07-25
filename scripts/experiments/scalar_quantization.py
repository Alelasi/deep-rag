"""Scalar Quantization（标量量化）实现

最实用的量化方法：
- 压缩比：4:1（float32 → int8）
- 召回率损失：<2%
- 速度提升：1.5-2x
- 内存节省：75%

适用场景：生产环境首选，性价比最高
"""

import numpy as np
from typing import Tuple, Optional
import logging

log = logging.getLogger("deeprag.scalar_quantization")


class ScalarQuantizer:
    """标量量化器

    将 float32 向量压缩为 int8，节省 75% 内存

    原理：
    1. 计算每个维度的 min/max
    2. 线性映射到 [-127, 127]
    3. 存储 int8 + min/max 参数
    """

    def __init__(self, dim: int):
        """初始化

        Args:
            dim: 向量维度
        """
        self.dim = dim
        self.min_vals = None
        self.max_vals = None
        self.is_trained = False

    def train(self, vectors: np.ndarray):
        """训练量化器（计算 min/max）

        Args:
            vectors: 训练向量 (N, dim) float32
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vectors.shape[1]}")

        # 计算每个维度的 min/max
        self.min_vals = vectors.min(axis=0)
        self.max_vals = vectors.max(axis=0)

        # 避免除零
        mask = self.max_vals == self.min_vals
        self.max_vals[mask] = self.min_vals[mask] + 1e-6

        self.is_trained = True
        log.info(f"Scalar quantizer trained on {vectors.shape[0]} vectors")

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """编码为 int8

        Args:
            vectors: 输入向量 (N, dim) float32

        Returns:
            量化向量 (N, dim) int8
        """
        if not self.is_trained:
            raise ValueError("Quantizer not trained yet")

        # 线性映射到 [-127, 127]
        scale = 254.0 / (self.max_vals - self.min_vals)
        quantized = ((vectors - self.min_vals) * scale - 127).astype(np.int8)

        return quantized

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """解码为 float32

        Args:
            quantized: 量化向量 (N, dim) int8

        Returns:
            原始向量 (N, dim) float32
        """
        if not self.is_trained:
            raise ValueError("Quantizer not trained yet")

        # 反向映射
        scale = 254.0 / (self.max_vals - self.min_vals)
        decoded = (quantized.astype(np.float32) + 127) / scale + self.min_vals

        return decoded

    def compute_distance(
        self,
        query: np.ndarray,
        db_quantized: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """直接在量化空间计算距离（更快）

        Args:
            query: 查询向量 (dim,) float32
            db_quantized: 数据库向量 (N, dim) int8
            metric: 距离度量（"cosine" 或 "l2"）

        Returns:
            距离 (N,)
        """
        # 量化查询向量
        query_quantized = self.encode(query.reshape(1, -1))[0]

        if metric == "cosine":
            # 余弦距离 = 1 - 内积（归一化后）
            # 简化：直接计算 int8 内积
            dot_products = np.dot(db_quantized, query_quantized)

            # 归一化
            query_norm = np.linalg.norm(query_quantized)
            db_norms = np.linalg.norm(db_quantized, axis=1)

            cosine_sim = dot_products / (query_norm * db_norms + 1e-8)
            distances = 1 - cosine_sim

        elif metric == "l2":
            # L2 距离
            diff = db_quantized.astype(np.float32) - query_quantized.astype(np.float32)
            distances = np.linalg.norm(diff, axis=1)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return distances

    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        return 4.0  # float32 (4 bytes) → int8 (1 byte)

    def get_memory_saving(self, num_vectors: int) -> dict:
        """计算内存节省

        Args:
            num_vectors: 向量数量

        Returns:
            内存统计（MB）
        """
        original_mb = num_vectors * self.dim * 4 / 1024 / 1024
        quantized_mb = num_vectors * self.dim * 1 / 1024 / 1024
        params_mb = self.dim * 4 * 2 / 1024 / 1024  # min/max

        total_mb = quantized_mb + params_mb
        saved_mb = original_mb - total_mb

        return {
            "original_mb": original_mb,
            "quantized_mb": total_mb,
            "saved_mb": saved_mb,
            "saving_ratio": saved_mb / original_mb,
        }


class ScalarQuantizedIndex:
    """基于 Scalar Quantization 的向量索引

    结合 IVF + Scalar Quantization
    """

    def __init__(self, dim: int, nlist: int = 1024):
        """初始化

        Args:
            dim: 向量维度
            nlist: IVF 聚类数
        """
        self.dim = dim
        self.nlist = nlist
        self.quantizer = ScalarQuantizer(dim)

        # IVF 结构
        self.centroids = None  # (nlist, dim) float32
        self.inverted_lists = [[] for _ in range(nlist)]  # 每个桶的向量 ID
        self.quantized_vectors = None  # (N, dim) int8

        self.is_trained = False

    def train(self, vectors: np.ndarray):
        """训练索引

        Args:
            vectors: 训练向量 (N, dim) float32
        """
        from sklearn.cluster import MiniBatchKMeans

        # 1. 训练 IVF（KMeans 聚类）
        log.info(f"Training IVF with {self.nlist} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.nlist,
            random_state=42,
            batch_size=1000,
            max_iter=100
        )
        kmeans.fit(vectors)
        self.centroids = kmeans.cluster_centers_

        # 2. 训练 Scalar Quantizer
        log.info("Training scalar quantizer...")
        self.quantizer.train(vectors)

        self.is_trained = True
        log.info("✅ Index trained")

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """添加向量

        Args:
            vectors: 向量 (N, dim) float32
            ids: 向量 ID (N,)，如果为 None 则自动生成
        """
        if not self.is_trained:
            raise ValueError("Index not trained yet")

        if ids is None:
            start_id = 0 if self.quantized_vectors is None else len(self.quantized_vectors)
            ids = np.arange(start_id, start_id + len(vectors))

        # 1. 量化向量
        quantized = self.quantizer.encode(vectors)

        # 2. 分配到桶
        distances = np.linalg.norm(
            vectors[:, None, :] - self.centroids[None, :, :],
            axis=2
        )
        assignments = distances.argmin(axis=1)

        # 3. 存储
        if self.quantized_vectors is None:
            self.quantized_vectors = quantized
        else:
            self.quantized_vectors = np.vstack([self.quantized_vectors, quantized])

        for i, (vec_id, bucket_id) in enumerate(zip(ids, assignments)):
            self.inverted_lists[bucket_id].append(vec_id)

        log.info(f"Added {len(vectors)} vectors")

    def search(self, query: np.ndarray, k: int = 10, nprobe: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最近邻

        Args:
            query: 查询向量 (dim,) float32
            k: 返回 top-k
            nprobe: 搜索多少个桶

        Returns:
            distances: 距离 (k,)
            ids: 向量 ID (k,)
        """
        if not self.is_trained:
            raise ValueError("Index not trained yet")

        # 1. 找最近的 nprobe 个桶
        centroid_distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_buckets = np.argpartition(centroid_distances, nprobe)[:nprobe]

        # 2. 收集候选向量
        candidate_ids = []
        for bucket_id in nearest_buckets:
            candidate_ids.extend(self.inverted_lists[bucket_id])

        if len(candidate_ids) == 0:
            return np.array([]), np.array([])

        candidate_ids = np.array(candidate_ids)

        # 3. 在量化空间计算距离
        candidate_vectors = self.quantized_vectors[candidate_ids]
        distances = self.quantizer.compute_distance(query, candidate_vectors, metric="cosine")

        # 4. 排序返回 top-k
        if len(distances) <= k:
            sorted_idx = np.argsort(distances)
            return distances[sorted_idx], candidate_ids[sorted_idx]

        top_k_idx = np.argpartition(distances, k)[:k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        return distances[top_k_idx], candidate_ids[top_k_idx]

    def get_stats(self) -> dict:
        """获取索引统计"""
        if self.quantized_vectors is None:
            num_vectors = 0
        else:
            num_vectors = len(self.quantized_vectors)

        memory_stats = self.quantizer.get_memory_saving(num_vectors)

        return {
            "dim": self.dim,
            "nlist": self.nlist,
            "num_vectors": num_vectors,
            "is_trained": self.is_trained,
            **memory_stats,
        }


def benchmark_scalar_quantization(dim: int = 384, num_vectors: int = 100000):
    """Benchmark Scalar Quantization"""
    import time

    print(f"\n{'='*80}")
    print(f"Scalar Quantization Benchmark (dim={dim}, num_vectors={num_vectors:,})")
    print(f"{'='*80}")

    # 生成测试数据
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    # 训练量化器
    quantizer = ScalarQuantizer(dim)
    quantizer.train(vectors)

    # 编码
    start = time.time()
    quantized = quantizer.encode(vectors)
    encode_time = time.time() - start

    # 解码
    start = time.time()
    decoded = quantizer.decode(quantized)
    decode_time = time.time() - start

    # 计算误差
    mse = np.mean((vectors - decoded) ** 2)

    # 内存统计
    memory_stats = quantizer.get_memory_saving(num_vectors)

    # 打印结果
    print(f"\n编码时间: {encode_time:.3f}s ({num_vectors/encode_time:.0f} vectors/s)")
    print(f"解码时间: {decode_time:.3f}s ({num_vectors/decode_time:.0f} vectors/s)")
    print(f"重建误差 (MSE): {mse:.6f}")
    print(f"\n内存统计:")
    print(f"  原始: {memory_stats['original_mb']:.1f} MB")
    print(f"  量化: {memory_stats['quantized_mb']:.1f} MB")
    print(f"  节省: {memory_stats['saved_mb']:.1f} MB ({memory_stats['saving_ratio']:.1%})")
    print(f"  压缩比: {quantizer.get_compression_ratio():.1f}:1")

    # 搜索性能
    start = time.time()
    distances_original = 1 - np.dot(vectors, query)
    search_time_original = time.time() - start

    start = time.time()
    distances_quantized = quantizer.compute_distance(query, quantized, metric="cosine")
    search_time_quantized = time.time() - start

    print(f"\n搜索性能:")
    print(f"  原始 float32: {search_time_original*1000:.2f}ms")
    print(f"  量化 int8: {search_time_quantized*1000:.2f}ms")
    print(f"  加速比: {search_time_original/search_time_quantized:.2f}x")

    # 召回率
    k = 10
    top_k_original = np.argpartition(distances_original, k)[:k]
    top_k_quantized = np.argpartition(distances_quantized, k)[:k]
    recall = len(set(top_k_original) & set(top_k_quantized)) / k

    print(f"\n召回率 @{k}: {recall:.1%}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 小规模测试
    benchmark_scalar_quantization(dim=384, num_vectors=100_000)

    # 中等规模测试
    benchmark_scalar_quantization(dim=384, num_vectors=1_000_000)
