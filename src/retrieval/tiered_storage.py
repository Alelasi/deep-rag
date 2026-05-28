"""分层存储：热数据（FAISS 内存）+ 冷数据（LanceDB 磁盘）

架构：
  查询 → FAISS（内存，<0.1ms）→ 命中返回
                              ↓ 未命中
                         LanceDB（磁盘）→ 返回 + 提升到 FAISS

特性：
  - 自动热数据提升（访问频率 > 阈值）
  - LRU 淘汰（内存满时）
  - 持久化（LanceDB 磁盘）
"""
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
import time


class TieredVectorStore:
    """分层向量存储（热内存 + 冷磁盘）"""

    def __init__(
        self,
        hot_capacity: int = 10000,  # 热数据容量（内存）
        promotion_threshold: int = 3,  # 提升阈值（访问 3 次 → 热数据）
        use_gpu: bool = False,
    ):
        """
        Args:
            hot_capacity: 热数据最大容量（超过则 LRU 淘汰）
            promotion_threshold: 冷→热提升阈值（访问次数）
            use_gpu: 是否使用 GPU 加速 FAISS
        """
        self.hot_capacity = hot_capacity
        self.promotion_threshold = promotion_threshold
        self.use_gpu = use_gpu

        # 热数据层（FAISS 内存）
        self.hot_index = None
        self.hot_ids = []  # 热数据 ID 列表
        self.hot_id_to_idx = {}  # ID → FAISS 索引映射

        # 冷数据层（LanceDB 磁盘）
        self.cold_db = None
        self.cold_table = None

        # 访问统计（用于热数据提升）
        self.access_count = defaultdict(int)
        self.last_access_time = {}

    def init_hot_index(self, dim: int):
        """初始化 FAISS 热索引"""
        try:
            import faiss

            if self.use_gpu and faiss.get_num_gpus() > 0:
                # GPU 索引
                res = faiss.StandardGpuResources()
                self.hot_index = faiss.GpuIndexFlatL2(res, dim)
                print(f"✅ FAISS GPU index initialized (dim={dim})")
            else:
                # CPU 索引
                self.hot_index = faiss.IndexFlatL2(dim)
                print(f"✅ FAISS CPU index initialized (dim={dim})")
        except ImportError:
            print("⚠️ FAISS not installed, hot tier disabled")
            self.hot_index = None

    def init_cold_db(self, db_path: str = "data/lancedb"):
        """初始化 LanceDB 冷存储"""
        try:
            import lancedb

            self.cold_db = lancedb.connect(db_path)

            # 尝试打开已有表
            try:
                self.cold_table = self.cold_db.open_table("vectors")
                print(f"✅ LanceDB connected: {db_path} (table: vectors, {self.cold_table.count_rows()} rows)")
            except Exception:
                print(f"✅ LanceDB connected: {db_path} (no existing table)")

        except ImportError:
            print("⚠️ LanceDB not installed, cold tier disabled")
            self.cold_db = None

    def add(self, doc_id: str, vector: np.ndarray, metadata: dict = None):
        """添加向量（默认存冷数据）"""
        if self.cold_db is None:
            raise RuntimeError("Cold DB not initialized")

        # 存入 LanceDB（冷数据）
        data = {
            "id": doc_id,
            "vector": vector.tolist(),
            "metadata": metadata or {},
        }

        if self.cold_table is None:
            self.cold_table = self.cold_db.create_table("vectors", data=[data])
        else:
            self.cold_table.add([data])

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """分层检索（热 → 冷）"""
        results = []

        # 1. 先查热数据（FAISS 内存）
        if self.hot_index is not None and self.hot_index.ntotal > 0:
            hot_results = self._search_hot(query_vector, top_k)
            results.extend(hot_results)

            # 记录访问
            for doc_id, score in hot_results:
                self._record_access(doc_id)

        # 2. 热数据不足，查冷数据（LanceDB 磁盘）
        if len(results) < top_k and self.cold_table is not None:
            cold_results = self._search_cold(query_vector, top_k - len(results))
            results.extend(cold_results)

            # 检查是否需要提升到热数据
            for doc_id, score in cold_results:
                self._record_access(doc_id)
                if self._should_promote(doc_id):
                    self._promote_to_hot(doc_id)

        return results[:top_k]

    def _search_hot(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """搜索热数据（FAISS）"""
        if self.hot_index is None or self.hot_index.ntotal == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.hot_index.search(query, min(top_k, len(self.hot_ids)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.hot_ids):
                doc_id = self.hot_ids[idx]
                results.append((doc_id, float(dist)))

        return results

    def _search_cold(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """搜索冷数据（LanceDB）"""
        if self.cold_table is None:
            return []

        results = (
            self.cold_table.search(query_vector.tolist()).limit(top_k).to_list()
        )

        return [(r["id"], r["_distance"]) for r in results]

    def _record_access(self, doc_id: str):
        """记录访问（用于热数据提升）"""
        self.access_count[doc_id] += 1
        self.last_access_time[doc_id] = time.time()

    def _should_promote(self, doc_id: str) -> bool:
        """判断是否应该提升到热数据"""
        return self.access_count[doc_id] >= self.promotion_threshold

    def _promote_to_hot(self, doc_id: str):
        """提升到热数据（冷 → 热）"""
        if self.hot_index is None:
            return

        # 检查是否已在热数据
        if doc_id in self.hot_id_to_idx:
            return

        # 从 LanceDB 读取向量
        result = self.cold_table.search().where(f"id = '{doc_id}'").limit(1).to_list()
        if not result:
            return

        vector = np.array(result[0]["vector"], dtype=np.float32)

        # LRU 淘汰（内存满时）
        if len(self.hot_ids) >= self.hot_capacity:
            self._evict_lru()

        # 添加到 FAISS
        self.hot_index.add(vector.reshape(1, -1))
        idx = len(self.hot_ids)
        self.hot_ids.append(doc_id)
        self.hot_id_to_idx[doc_id] = idx

        print(f"🔥 Promoted to hot: {doc_id} (access_count={self.access_count[doc_id]})")

    def _evict_lru(self):
        """LRU 淘汰（热 → 冷）"""
        if not self.hot_ids:
            return

        # 找到最久未访问的
        lru_id = min(self.hot_ids, key=lambda x: self.last_access_time.get(x, 0))
        lru_idx = self.hot_id_to_idx[lru_id]

        # 从热数据移除（FAISS 不支持删除，需要重建索引）
        # 简化实现：只从映射中移除
        del self.hot_id_to_idx[lru_id]
        self.hot_ids[lru_idx] = None  # 标记为空

        print(f"❄️ Evicted from hot: {lru_id}")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "hot_count": len([x for x in self.hot_ids if x is not None]),
            "cold_count": self.cold_table.count_rows() if self.cold_table else 0,
            "hot_capacity": self.hot_capacity,
            "total_accesses": sum(self.access_count.values()),
        }


# 使用示例
if __name__ == "__main__":
    # 初始化分层存储
    store = TieredVectorStore(hot_capacity=1000, promotion_threshold=3, use_gpu=False)
    store.init_hot_index(dim=384)
    store.init_cold_db("data/lancedb")

    # 添加向量（存入冷数据）
    for i in range(100):
        vector = np.random.randn(384).astype(np.float32)
        store.add(doc_id=f"doc_{i}", vector=vector, metadata={"index": i})

    # 检索（自动热数据提升）
    query = np.random.randn(384).astype(np.float32)
    for _ in range(5):
        results = store.search(query, top_k=10)
        print(f"Results: {results[:3]}")

    # 统计
    print(store.get_stats())
