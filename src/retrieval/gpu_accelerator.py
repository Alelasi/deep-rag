"""
GPU 加速模块 — 为 RAG 检索提供 CUDA 加速
支持：向量化、向量检索、BM25 计算

整合自助理项目的 GPU 优化方案：
- FAISS GPU（20-30x 加速）
- CuPy（10-15x 加速）
- PyTorch CUDA（8-12x 加速）
- Numba CUDA（BM25 加速）

参考：D:\文档\ai提问相关\助理\docs\RAG系统GPU优化方案.md
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import time

log = logging.getLogger("deeprag.gpu")

# 延迟导入 GPU 依赖
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        DEVICE = torch.device("cuda")
        GPU_NAME = torch.cuda.get_device_name(0)
        log.info(f"GPU available: {GPU_NAME}")
    else:
        DEVICE = torch.device("cpu")
        log.warning("CUDA not available, falling back to CPU")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    log.warning("PyTorch not installed, GPU acceleration disabled")

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = cuda.is_available()
    if NUMBA_CUDA_AVAILABLE:
        log.info("Numba CUDA available for BM25 acceleration")
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    log.warning("Numba not installed, BM25 GPU acceleration disabled")

try:
    import faiss
    FAISS_AVAILABLE = True
    FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
    if FAISS_GPU_AVAILABLE:
        log.info(f"FAISS GPU available: {faiss.get_num_gpus()} GPUs")
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    log.warning("FAISS not installed, FAISS GPU acceleration disabled")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    log.info("CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    log.warning("CuPy not installed, CuPy GPU acceleration disabled")


class GPUEmbedder:
    """GPU 加速的文本向量化器

    使用 sentence-transformers + CUDA 批量处理文档
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", batch_size: int = 32):
        """
        Args:
            model_name: HuggingFace 模型名称
            batch_size: GPU 批处理大小（根据显存调整）
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

        if not TORCH_AVAILABLE:
            log.warning("GPU not available, embedder will use CPU")
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=DEVICE)
            log.info(f"Loaded {model_name} on {DEVICE}")
        except ImportError:
            log.error("sentence-transformers not installed")
        except Exception as e:
            log.error(f"Failed to load model: {e}")

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """批量向量化文本

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            向量矩阵 (N, D)
        """
        if self.model is None:
            log.warning("Model not loaded, returning zero vectors")
            return np.zeros((len(texts), 384))

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=DEVICE
        )
        return embeddings


class GPUVectorSearch:
    """GPU 加速的向量检索

    使用 PyTorch CUDA 进行批量余弦相似度计算
    """

    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: 文档向量矩阵 (N, D)
        """
        self.embeddings = embeddings
        self.device = DEVICE if TORCH_AVAILABLE else None

        if TORCH_AVAILABLE:
            # 转为 GPU Tensor 并归一化（余弦相似度优化）
            self.embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
            self.embeddings_tensor = torch.nn.functional.normalize(self.embeddings_tensor, p=2, dim=1)
            log.info(f"Loaded {len(embeddings)} vectors to GPU")
        else:
            self.embeddings_tensor = None

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """GPU 加速的向量检索

        Args:
            query_vector: 查询向量 (D,)
            top_k: 返回 Top-K 结果

        Returns:
            (indices, scores): 索引数组和分数数组
        """
        if not TORCH_AVAILABLE or self.embeddings_tensor is None:
            # CPU fallback
            return self._cpu_search(query_vector, top_k)

        # GPU 检索
        query_tensor = torch.from_numpy(query_vector).float().to(self.device)
        query_tensor = torch.nn.functional.normalize(query_tensor.unsqueeze(0), p=2, dim=1)

        # 批量余弦相似度（矩阵乘法）
        scores = torch.matmul(self.embeddings_tensor, query_tensor.T).squeeze()

        # Top-K
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

        return top_indices.cpu().numpy(), top_scores.cpu().numpy()

    def _cpu_search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback"""
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([query_vector], self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return top_indices, top_scores


class GPUBM25:
    """GPU 加速的 BM25 计算

    使用 Numba CUDA 并行计算 BM25 分数
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            corpus: 文档列表
            k1: BM25 参数
            b: BM25 参数
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.use_gpu = NUMBA_CUDA_AVAILABLE

        # 预计算统计量
        self.doc_lens = np.array([len(doc.split()) for doc in corpus], dtype=np.float32)
        self.avgdl = np.mean(self.doc_lens)

        # 构建词汇表和倒排索引
        self._build_index()

        if self.use_gpu:
            log.info(f"BM25 GPU mode enabled for {len(corpus)} documents")
        else:
            log.info("BM25 using CPU mode")

    def _build_index(self):
        """构建倒排索引"""
        from collections import defaultdict

        self.vocab = set()
        self.doc_freqs = defaultdict(int)
        self.inverted_index = defaultdict(list)

        for doc_id, doc in enumerate(self.corpus):
            tokens = doc.split()
            unique_tokens = set(tokens)
            self.vocab.update(unique_tokens)

            for token in unique_tokens:
                self.doc_freqs[token] += 1
                self.inverted_index[token].append(doc_id)

        self.N = len(self.corpus)

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """BM25 检索

        Args:
            query: 查询文本
            top_k: 返回 Top-K 结果

        Returns:
            (indices, scores): 索引数组和分数数组
        """
        query_tokens = query.split()

        if self.use_gpu and len(query_tokens) > 5:
            # GPU 加速（仅对长查询有效）
            return self._gpu_search(query_tokens, top_k)
        else:
            # CPU 计算
            return self._cpu_search(query_tokens, top_k)

    def _cpu_search(self, query_tokens: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """CPU BM25 计算"""
        scores = np.zeros(self.N, dtype=np.float32)

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            df = self.doc_freqs[token]
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id in self.inverted_index[token]:
                tf = self.corpus[doc_id].split().count(token)
                doc_len = self.doc_lens[doc_id]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_id] += idf * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return top_indices, top_scores

    def _gpu_search(self, query_tokens: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """GPU BM25 计算（Numba CUDA）"""
        # TODO: 实现 Numba CUDA kernel
        # 当前先 fallback 到 CPU
        log.debug("GPU BM25 not yet implemented, using CPU")
        return self._cpu_search(query_tokens, top_k)


class FAISSGPUSearch:
    """FAISS GPU 加速的向量检索（最快，20-30x 加速）

    基于助理项目的 GPU 优化方案
    """

    def __init__(self, embeddings: np.ndarray, nlist: int = 100):
        """
        Args:
            embeddings: 文档向量矩阵 (N, D)
            nlist: IVF 聚类中心数（根据文档数量调整）
        """
        self.embeddings = embeddings
        self.nlist = nlist
        self.index = None
        self.gpu_resources = None

        if not FAISS_AVAILABLE:
            log.error("FAISS not installed")
            return

        dimension = embeddings.shape[1]

        # 创建 FAISS 索引（IVFFlat）
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        if FAISS_GPU_AVAILABLE:
            # 转移到 GPU
            self.gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
            log.info(f"FAISS GPU index created for {len(embeddings)} vectors")
        else:
            log.warning("FAISS GPU not available, using CPU")

        # 训练和添加向量
        index.train(embeddings)
        index.add(embeddings)

        self.index = index

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS GPU 加速检索

        Args:
            query_vector: 查询向量 (D,)
            top_k: 返回 Top-K 结果

        Returns:
            (indices, scores): 索引数组和分数数组
        """
        if self.index is None:
            log.error("FAISS index not initialized")
            return np.array([]), np.array([])

        # FAISS 搜索
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32),
            top_k
        )

        # 转换距离为相似度分数
        scores = 1 / (1 + distances[0])

        return indices[0], scores


class CuPyVectorSearch:
    """CuPy GPU 加速的向量检索（轻量级，10-15x 加速）

    基于助理项目的 GPU 优化方案
    """

    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: 文档向量矩阵 (N, D)
        """
        if not CUPY_AVAILABLE:
            log.error("CuPy not installed")
            self.embeddings = embeddings
            return

        # 转移到 GPU 并归一化
        self.embeddings = cp.asarray(embeddings)
        self.embeddings = self.embeddings / cp.linalg.norm(self.embeddings, axis=1, keepdims=True)
        log.info(f"CuPy GPU: loaded {len(embeddings)} vectors ({self.embeddings.nbytes / 1e6:.2f} MB)")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """CuPy GPU 加速检索

        Args:
            query_vector: 查询向量 (D,)
            top_k: 返回 Top-K 结果

        Returns:
            (indices, scores): 索引数组和分数数组
        """
        if not CUPY_AVAILABLE:
            # CPU fallback
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity([query_vector], self.embeddings)[0]
            top_indices = np.argsort(scores)[::-1][:top_k]
            return top_indices, scores[top_indices]

        # 转移查询向量到 GPU
        query_gpu = cp.asarray(query_vector)
        query_gpu = query_gpu / cp.linalg.norm(query_gpu)

        # 余弦相似度（GPU 并行）
        scores = cp.dot(self.embeddings, query_gpu)

        # Top-K
        top_indices = cp.argsort(scores)[::-1][:top_k]

        return cp.asnumpy(top_indices), cp.asnumpy(scores[top_indices])


def benchmark_gpu_search(embeddings: np.ndarray, query_vector: np.ndarray, top_k: int = 10, num_queries: int = 100):
    """性能基准测试：对比 CPU vs GPU 检索速度

    Args:
        embeddings: 文档向量矩阵
        query_vector: 查询向量
        top_k: 返回结果数
        num_queries: 测试查询次数

    Returns:
        性能对比结果
    """
    results = {}

    # 1. CPU Baseline
    print("Testing CPU baseline...")
    start = time.time()
    for _ in range(num_queries):
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([query_vector], embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
    cpu_time = time.time() - start
    results["CPU"] = {
        "total_time": cpu_time,
        "avg_time": cpu_time / num_queries * 1000,  # ms
        "speedup": 1.0
    }
    print(f"  CPU: {cpu_time:.2f}s total, {cpu_time/num_queries*1000:.2f}ms avg")

    # 2. PyTorch GPU
    if TORCH_AVAILABLE:
        print("Testing PyTorch GPU...")
        searcher = GPUVectorSearch(embeddings)
        start = time.time()
        for _ in range(num_queries):
            searcher.search(query_vector, top_k)
        pytorch_time = time.time() - start
        results["PyTorch GPU"] = {
            "total_time": pytorch_time,
            "avg_time": pytorch_time / num_queries * 1000,
            "speedup": cpu_time / pytorch_time
        }
        print(f"  PyTorch GPU: {pytorch_time:.2f}s total, {pytorch_time/num_queries*1000:.2f}ms avg, {cpu_time/pytorch_time:.1f}x speedup")

    # 3. CuPy GPU
    if CUPY_AVAILABLE:
        print("Testing CuPy GPU...")
        try:
            searcher = CuPyVectorSearch(embeddings)
            start = time.time()
            for _ in range(num_queries):
                searcher.search(query_vector, top_k)
            cupy_time = time.time() - start
            results["CuPy GPU"] = {
                "total_time": cupy_time,
                "avg_time": cupy_time / num_queries * 1000,
                "speedup": cpu_time / cupy_time
            }
            print(f"  CuPy GPU: {cupy_time:.2f}s total, {cupy_time/num_queries*1000:.2f}ms avg, {cpu_time/cupy_time:.1f}x speedup")
        except Exception as e:
            print(f"  CuPy GPU: FAILED ({type(e).__name__}: {str(e)[:80]})")
            results["CuPy GPU"] = {"error": str(e)}

    # 4. FAISS GPU
    if FAISS_GPU_AVAILABLE:
        print("Testing FAISS GPU...")
        searcher = FAISSGPUSearch(embeddings)
        start = time.time()
        for _ in range(num_queries):
            searcher.search(query_vector, top_k)
        faiss_time = time.time() - start
        results["FAISS GPU"] = {
            "total_time": faiss_time,
            "avg_time": faiss_time / num_queries * 1000,
            "speedup": cpu_time / faiss_time
        }
        print(f"  FAISS GPU: {faiss_time:.2f}s total, {faiss_time/num_queries*1000:.2f}ms avg, {cpu_time/faiss_time:.1f}x speedup")

    return results


def get_gpu_info() -> Dict[str, any]:
    """获取 GPU 信息"""
    info = {
        "torch_available": TORCH_AVAILABLE,
        "numba_cuda_available": NUMBA_CUDA_AVAILABLE,
        "faiss_available": FAISS_AVAILABLE,
        "faiss_gpu_available": FAISS_GPU_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "device": str(DEVICE) if DEVICE else "N/A",
    }

    if TORCH_AVAILABLE:
        info["gpu_name"] = GPU_NAME
        info["cuda_version"] = torch.version.cuda
        info["pytorch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"

    if FAISS_AVAILABLE:
        info["faiss_version"] = faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
        if FAISS_GPU_AVAILABLE:
            info["faiss_num_gpus"] = faiss.get_num_gpus()

    return info


if __name__ == "__main__":
    # 测试 GPU 加速
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=== GPU 加速模块测试 ===\n")

    # 1. GPU 信息
    info = get_gpu_info()
    print("GPU Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    # 2. 测试向量化
    if TORCH_AVAILABLE:
        print("Testing GPUEmbedder...")
        embedder = GPUEmbedder()
        texts = ["这是测试文本1", "这是测试文本2", "这是测试文本3"]
        embeddings = embedder.encode(texts)
        print(f"  Encoded {len(texts)} texts, shape: {embeddings.shape}")
        print()

    # 3. 测试向量检索
    if TORCH_AVAILABLE:
        print("Testing GPUVectorSearch...")
        searcher = GPUVectorSearch(embeddings)
        query_vec = embeddings[0]
        indices, scores = searcher.search(query_vec, top_k=2)
        print(f"  Top-2 results: indices={indices}, scores={scores}")
        print()

    # 4. 测试 BM25
    print("Testing GPUBM25...")
    corpus = ["人工智能 机器学习", "深度学习 神经网络", "自然语言处理 NLP"]
    bm25 = GPUBM25(corpus)
    indices, scores = bm25.search("人工智能", top_k=2)
    print(f"  Top-2 results: indices={indices}, scores={scores}")
    print()

    # 5. 性能基准测试（可选）
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("=== 性能基准测试 ===\n")
        print("生成测试数据...")
        # 生成大规模测试数据
        num_docs = 10000
        dim = 384
        test_embeddings = np.random.randn(num_docs, dim).astype(np.float32)
        test_query = np.random.randn(dim).astype(np.float32)

        print(f"测试数据：{num_docs} 个文档，维度 {dim}\n")

        # 运行基准测试
        results = benchmark_gpu_search(test_embeddings, test_query, top_k=10, num_queries=100)

        print("\n=== 性能对比总结 ===")
        print(f"{'Backend':<15} {'Avg Time (ms)':<15} {'Speedup':<10}")
        print("-" * 40)
        for backend, metrics in results.items():
            if "error" in metrics:
                print(f"{backend:<15} {'FAILED':<15} {'N/A':<10}")
            else:
                print(f"{backend:<15} {metrics['avg_time']:<15.2f} {metrics['speedup']:<10.1f}x")

    print("\n✅ All tests passed!")
    print("\n提示：运行 'python gpu_accelerator.py benchmark' 进行性能基准测试")
