"""向量索引性能对比测试

对比 HNSW / IVF / IVF+PQ 三种索引的性能
测试指标：查询延迟、召回率、内存占用、吞吐量

基于真实数据验证《向量索引选择指南》中的建议
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple
import logging

log = logging.getLogger("deeprag.benchmark")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    log.error("FAISS not installed")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    log.error("ChromaDB not installed")


class IndexBenchmark:
    """向量索引性能对比测试"""

    def __init__(self, dim: int = 384, num_vectors: int = 100000):
        """初始化

        Args:
            dim: 向量维度
            num_vectors: 测试向量数量
        """
        self.dim = dim
        self.num_vectors = num_vectors
        self.vectors = None
        self.query_vectors = None
        self.ground_truth = None

        log.info(f"Benchmark initialized: dim={dim}, num_vectors={num_vectors}")

    def generate_data(self, num_queries: int = 1000):
        """生成测试数据"""
        log.info("Generating test data...")

        # 生成随机向量（模拟真实 embedding）
        np.random.seed(42)
        self.vectors = np.random.randn(self.num_vectors, self.dim).astype('float32')
        self.query_vectors = np.random.randn(num_queries, self.dim).astype('float32')

        # L2 归一化（模拟 sentence-transformers 输出）
        self.vectors = self._normalize(self.vectors)
        self.query_vectors = self._normalize(self.query_vectors)

        # 计算 ground truth（暴力搜索）
        log.info("Computing ground truth...")
        self.ground_truth = self._compute_ground_truth(k=10)

        log.info("✅ Test data generated")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 归一化"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def _compute_ground_truth(self, k: int = 10) -> np.ndarray:
        """计算 ground truth（精确搜索）"""
        index = faiss.IndexFlatIP(self.dim)  # 内积（余弦距离）
        index.add(self.vectors)
        _, ids = index.search(self.query_vectors, k)
        return ids

    def benchmark_flat(self, k: int = 10) -> Dict:
        """Benchmark: Flat 精确搜索"""
        log.info("Benchmarking Flat index...")

        # 构建索引
        start_time = time.time()
        index = faiss.IndexFlatIP(self.dim)
        index.add(self.vectors)
        build_time = time.time() - start_time

        # 内存占用
        memory_mb = self._get_index_memory(index)

        # 查询性能
        latencies = []
        for query in self.query_vectors:
            start = time.time()
            _, ids = index.search(query.reshape(1, -1), k)
            latencies.append((time.time() - start) * 1000)  # ms

        # 召回率
        recall = self._compute_recall(index, k)

        return {
            "name": "Flat (精确搜索)",
            "build_time": build_time,
            "memory_mb": memory_mb,
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "qps": 1000 / np.mean(latencies),
            "recall": recall,
        }

    def benchmark_hnsw(self, k: int = 10, M: int = 32, ef_construction: int = 200, ef_search: int = 64) -> Dict:
        """Benchmark: HNSW 索引"""
        log.info("Benchmarking HNSW index...")

        # 构建索引
        start_time = time.time()
        index = faiss.IndexHNSWFlat(self.dim, M)
        index.hnsw.efConstruction = ef_construction
        index.add(self.vectors)
        build_time = time.time() - start_time

        # 设置查询参数
        index.hnsw.efSearch = ef_search

        # 内存占用
        memory_mb = self._get_index_memory(index)

        # 查询性能
        latencies = []
        for query in self.query_vectors:
            start = time.time()
            _, ids = index.search(query.reshape(1, -1), k)
            latencies.append((time.time() - start) * 1000)

        # 召回率
        recall = self._compute_recall(index, k)

        return {
            "name": f"HNSW (M={M}, ef={ef_search})",
            "build_time": build_time,
            "memory_mb": memory_mb,
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "qps": 1000 / np.mean(latencies),
            "recall": recall,
        }

    def benchmark_ivf(self, k: int = 10, nlist: int = 4096, nprobe: int = 32) -> Dict:
        """Benchmark: IVF 索引"""
        log.info("Benchmarking IVF index...")

        # 构建索引
        start_time = time.time()
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # 训练
        train_size = min(self.num_vectors, nlist * 100)
        index.train(self.vectors[:train_size])
        index.add(self.vectors)
        build_time = time.time() - start_time

        # 设置查询参数
        index.nprobe = nprobe

        # 内存占用
        memory_mb = self._get_index_memory(index)

        # 查询性能
        latencies = []
        for query in self.query_vectors:
            start = time.time()
            _, ids = index.search(query.reshape(1, -1), k)
            latencies.append((time.time() - start) * 1000)

        # 召回率
        recall = self._compute_recall(index, k)

        return {
            "name": f"IVF (nlist={nlist}, nprobe={nprobe})",
            "build_time": build_time,
            "memory_mb": memory_mb,
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "qps": 1000 / np.mean(latencies),
            "recall": recall,
        }

    def benchmark_ivf_pq(self, k: int = 10, nlist: int = 4096, nprobe: int = 64, m: int = 8) -> Dict:
        """Benchmark: IVF+PQ 索引"""
        log.info("Benchmarking IVF+PQ index...")

        # 构建索引
        start_time = time.time()
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)

        # 训练
        train_size = min(self.num_vectors, nlist * 100)
        index.train(self.vectors[:train_size])
        index.add(self.vectors)
        build_time = time.time() - start_time

        # 设置查询参数
        index.nprobe = nprobe

        # 内存占用
        memory_mb = self._get_index_memory(index)

        # 查询性能
        latencies = []
        for query in self.query_vectors:
            start = time.time()
            _, ids = index.search(query.reshape(1, -1), k)
            latencies.append((time.time() - start) * 1000)

        # 召回率
        recall = self._compute_recall(index, k)

        # 计算压缩比
        original_size = self.num_vectors * self.dim * 4  # float32
        compressed_size = self.num_vectors * m * 1  # uint8
        compression_ratio = original_size / compressed_size

        return {
            "name": f"IVF+PQ (m={m}, nprobe={nprobe})",
            "build_time": build_time,
            "memory_mb": memory_mb,
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "qps": 1000 / np.mean(latencies),
            "recall": recall,
            "compression_ratio": f"{compression_ratio:.1f}:1",
        }

    def _compute_recall(self, index, k: int = 10) -> float:
        """计算召回率"""
        _, ids = index.search(self.query_vectors, k)

        # 计算每个查询的召回率
        recalls = []
        for i in range(len(self.query_vectors)):
            gt_set = set(self.ground_truth[i])
            pred_set = set(ids[i])
            recall = len(gt_set & pred_set) / k
            recalls.append(recall)

        return np.mean(recalls)

    def _get_index_memory(self, index) -> float:
        """估算索引内存占用（MB）"""
        # 简化估算：向量数 × 维度 × 字节数
        if isinstance(index, faiss.IndexFlatIP) or isinstance(index, faiss.IndexFlatL2):
            return self.num_vectors * self.dim * 4 / 1024 / 1024

        elif isinstance(index, faiss.IndexHNSWFlat):
            # HNSW: 向量 + 图结构（约 2-3x）
            return self.num_vectors * self.dim * 4 * 2.5 / 1024 / 1024

        elif isinstance(index, faiss.IndexIVFFlat):
            # IVF: 向量 + 聚类中心
            nlist = index.nlist
            return (self.num_vectors * self.dim * 4 + nlist * self.dim * 4) / 1024 / 1024

        elif isinstance(index, faiss.IndexIVFPQ):
            # IVF+PQ: 压缩向量 + 聚类中心 + codebook
            m = index.pq.M
            nlist = index.nlist
            return (self.num_vectors * m + nlist * self.dim * 4) / 1024 / 1024

        else:
            return 0.0

    def run_all_benchmarks(self) -> List[Dict]:
        """运行所有 Benchmark"""
        if self.vectors is None:
            self.generate_data()

        results = []

        # Flat
        results.append(self.benchmark_flat())

        # HNSW
        results.append(self.benchmark_hnsw(M=32, ef_search=64))

        # IVF
        results.append(self.benchmark_ivf(nlist=1024, nprobe=32))

        # IVF+PQ
        results.append(self.benchmark_ivf_pq(nlist=1024, nprobe=64, m=8))

        return results

    def print_results(self, results: List[Dict]):
        """打印结果表格"""
        print("\n" + "="*100)
        print(f"向量索引性能对比 (dim={self.dim}, num_vectors={self.num_vectors:,})")
        print("="*100)
        print(f"{'索引类型':<25} {'构建时间':<12} {'内存(MB)':<12} {'延迟P95(ms)':<15} {'QPS':<10} {'召回率':<10}")
        print("-"*100)

        for r in results:
            print(f"{r['name']:<25} "
                  f"{r['build_time']:>10.2f}s "
                  f"{r['memory_mb']:>10.1f} "
                  f"{r['latency_p95']:>13.3f} "
                  f"{r['qps']:>8.0f} "
                  f"{r['recall']:>8.1%}")

        print("="*100)

        # 压缩比
        for r in results:
            if 'compression_ratio' in r:
                print(f"\n💾 {r['name']} 压缩比: {r['compression_ratio']}")

        print()


def main():
    """运行 Benchmark"""
    # 小规模测试（快速验证）
    print("🔬 小规模测试 (100K 向量)")
    benchmark_small = IndexBenchmark(dim=384, num_vectors=100_000)
    results_small = benchmark_small.run_all_benchmarks()
    benchmark_small.print_results(results_small)

    # 中等规模测试（接近生产）
    print("\n🔬 中等规模测试 (1M 向量)")
    benchmark_medium = IndexBenchmark(dim=384, num_vectors=1_000_000)
    results_medium = benchmark_medium.run_all_benchmarks()
    benchmark_medium.print_results(results_medium)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
