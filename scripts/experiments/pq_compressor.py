"""Product Quantization (PQ) 压缩器

手动实现PQ量化，无需FAISS GPU
- 384维 → 8个子向量 × 8bit = 8 bytes
- 压缩比：1536 bytes → 8 bytes = 192:1
"""

import numpy as np
from typing import Tuple
from sklearn.cluster import MiniBatchKMeans


class PQCompressor:
    """Product Quantization压缩器"""

    def __init__(self, dim: int = 384, num_subvectors: int = 8, num_clusters: int = 256):
        """初始化

        Args:
            dim: 向量维度
            num_subvectors: 子向量数量（8个子向量）
            num_clusters: 每个子向量的聚类中心数（256 = 8bit）
        """
        self.dim = dim
        self.num_subvectors = num_subvectors
        self.num_clusters = num_clusters
        self.subvector_dim = dim // num_subvectors

        # 每个子向量的codebook（聚类中心）
        self.codebooks = []
        self.trained = False

    def train(self, vectors: np.ndarray):
        """训练PQ codebook

        Args:
            vectors: 训练向量 (N, dim)
        """
        print(f"Training PQ: {self.num_subvectors} subvectors × {self.num_clusters} clusters...")

        self.codebooks = []

        for i in range(self.num_subvectors):
            # 提取第i个子向量
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]

            # KMeans聚类
            kmeans = MiniBatchKMeans(
                n_clusters=self.num_clusters,
                random_state=42,
                batch_size=1000,
                max_iter=100
            )
            kmeans.fit(subvectors)

            self.codebooks.append(kmeans.cluster_centers_)
            print(f"  Subvector {i+1}/{self.num_subvectors} trained")

        self.trained = True
        print("✅ PQ training complete")

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """编码向量为PQ codes

        Args:
            vectors: 输入向量 (N, dim)

        Returns:
            PQ codes (N, num_subvectors) uint8
        """
        if not self.trained:
            raise ValueError("PQ not trained yet")

        N = vectors.shape[0]
        codes = np.zeros((N, self.num_subvectors), dtype=np.uint8)

        for i in range(self.num_subvectors):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]

            # 找最近的聚类中心
            codebook = self.codebooks[i]
            distances = np.linalg.norm(
                subvectors[:, np.newaxis, :] - codebook[np.newaxis, :, :],
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """解码PQ codes为近似向量

        Args:
            codes: PQ codes (N, num_subvectors) uint8

        Returns:
            近似向量 (N, dim)
        """
        if not self.trained:
            raise ValueError("PQ not trained yet")

        N = codes.shape[0]
        vectors = np.zeros((N, self.dim), dtype=np.float32)

        for i in range(self.num_subvectors):
            start = i * self.subvector_dim
            end = start + self.subvector_dim

            # 从codebook查找
            codebook = self.codebooks[i]
            vectors[:, start:end] = codebook[codes[:, i]]

        return vectors

    def search(self, query: np.ndarray, codes: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最近邻

        Args:
            query: 查询向量 (dim,)
            codes: 所有PQ codes (N, num_subvectors)
            top_k: 返回top-k结果

        Returns:
            (distances, indices)
        """
        # 解码所有向量（近似）
        decoded_vectors = self.decode(codes)

        # 计算距离
        distances = np.linalg.norm(decoded_vectors - query, axis=1)

        # Top-K
        indices = np.argpartition(distances, top_k)[:top_k]
        indices = indices[np.argsort(distances[indices])]

        return distances[indices], indices


def test_pq():
    """测试PQ压缩"""
    print("=== Testing PQ Compression ===\n")

    # 生成测试数据
    dim = 384
    n_train = 10000
    n_test = 1000

    print(f"Generating {n_train} training vectors...")
    train_vectors = np.random.randn(n_train, dim).astype(np.float32)
    test_vectors = np.random.randn(n_test, dim).astype(np.float32)

    # 训练PQ
    pq = PQCompressor(dim=dim, num_subvectors=8, num_clusters=256)
    pq.train(train_vectors)

    # 编码
    print(f"\nEncoding {n_test} test vectors...")
    codes = pq.encode(test_vectors)
    print(f"✅ Encoded shape: {codes.shape}, dtype: {codes.dtype}")

    # 解码
    print("\nDecoding...")
    decoded = pq.decode(codes)

    # 计算误差
    mse = np.mean((test_vectors - decoded) ** 2)
    print(f"✅ MSE: {mse:.6f}")

    # 计算压缩比
    original_size = n_test * dim * 4  # float32
    compressed_size = n_test * 8  # 8 bytes per vector
    ratio = original_size / compressed_size

    print(f"\n📊 Compression:")
    print(f"  Original: {original_size / 1024:.1f} KB")
    print(f"  Compressed: {compressed_size / 1024:.1f} KB")
    print(f"  Ratio: {ratio:.0f}:1")

    # 测试搜索
    print("\n🔍 Testing search...")
    query = test_vectors[0]
    distances, indices = pq.search(query, codes, top_k=5)
    print(f"✅ Top-5 distances: {distances}")
    print(f"✅ Top-5 indices: {indices}")


if __name__ == "__main__":
    test_pq()
