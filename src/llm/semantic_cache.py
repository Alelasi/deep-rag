"""语义缓存 — v2.9.1新增

基于向量相似度的LLM响应缓存。
问题向量化 → 相似度搜索 → 命中则跳过LLM调用。

设计要点：
- 使用项目已有的 bge-base-zh-v1.5 embedding模型（不额外加载）
- 阈值默认0.90（FAQ类问题缓存1小时）
- 仅缓存LLM回答，不缓存RAG检索结果
- 内存缓存，无需持久化（进程重启自动清空）

v3.0改进（缓存与限流统一抽象）：
- 复用 cache 模块的 TTLCache 抽象（统一 TTL 语义 + 可插拔后端）
- embedding 懒加载与单例初始化加锁（线程安全）
- 日志统一使用标准库 logging.getLogger(__name__)
- 补充类型注解
"""
import time
import logging
import threading
from typing import Optional
from dataclasses import dataclass, field

from src.retrieval.cache import (
    CacheBackend,
    MemoryBackend,
    RedisBackend,
    TTLCache,
)

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_THRESHOLD = 0.90
DEFAULT_TTL = 3600  # 1小时
DEFAULT_MAX_ENTRIES = 500  # 最大缓存条目数


@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    query_embedding: list
    answer: str
    timestamp: float
    task_type: str
    hit_count: int = 0


class SemanticCache:
    """语义缓存

    用法：
        cache = SemanticCache(threshold=0.90, ttl=3600)
        # 查询
        cached = cache.get("INTJ的主导功能是什么？")
        if cached:
            return cached  # 跳过LLM调用
        # 未命中，调用LLM后写入
        answer = llm.invoke(...)
        cache.set("INTJ的主导功能是什么？", answer, task_type="generation")

    存储后端可通过 backend 参数替换（默认 MemoryBackend），预留 RedisBackend 钩子。
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        ttl: int = DEFAULT_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        backend: Optional[CacheBackend] = None,
    ):
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        # 复用统一 TTLCache 抽象（统一 TTL 语义 + 可插拔后端）
        self._store: TTLCache = TTLCache(
            ttl=ttl, max_size=max_entries, backend=backend, lru=True
        )
        self._lock = threading.Lock()           # 保护本类计数器/自增 id
        self._embedder_lock = threading.Lock()  # 保护 embedding 模型懒加载
        self._embedder = None
        self._hit_count = 0
        self._miss_count = 0
        self._id_counter = 0

        logger.info(
            f"[SemanticCache] 初始化: threshold={threshold}, ttl={ttl}s, "
            f"max_entries={max_entries}, backend={type(self._store._backend).__name__}"
        )

    def _get_embedder(self):
        """获取embedding模型（懒加载，复用全局缓存；双检锁保证只加载一次）"""
        if self._embedder is not None:
            return self._embedder
        with self._embedder_lock:
            if self._embedder is not None:
                return self._embedder
            try:
                from src.ui.model_cache import get_embedding_model
                from src.config import EMBEDDING_MODEL, DEVICE
                self._embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)
            except Exception as e:
                logger.warning(f"[SemanticCache] 无法加载embedding模型: {e}")
                return None
        return self._embedder

    def _embed(self, query: str) -> Optional[list]:
        """将查询文本转向量"""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            emb = embedder.encode([query])
            return emb[0].tolist()
        except Exception as e:
            logger.warning(f"[SemanticCache] embedding失败: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query: str, task_type: str = "generation") -> Optional[str]:
        """查询语义缓存

        Args:
            query: 查询文本
            task_type: 任务类型（仅generation类型使用缓存）

        Returns:
            命中则返回缓存的答案，未命中返回None
        """
        if task_type != "generation":
            return None  # 仅缓存生成任务

        query_emb = self._embed(query)
        if query_emb is None:
            return None

        now = time.time()

        with self._lock:
            best_entry: Optional[CacheEntry] = None
            best_score = 0.0

            for _key, entry in self._store.items():
                # items() 已过滤过期条目，这里双重校验 TTL
                if now - entry.timestamp > self.ttl:
                    continue

                # 计算相似度
                score = self._cosine_similarity(query_emb, entry.query_embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry is not None and best_score >= self.threshold:
                best_entry.hit_count += 1
                self._hit_count += 1
                logger.debug(
                    f"[SemanticCache] 命中: score={best_score:.3f}, "
                    f"hit_count={best_entry.hit_count}"
                )
                return best_entry.answer

            self._miss_count += 1
            return None

    def set(self, query: str, answer: str, task_type: str = "generation") -> None:
        """写入缓存

        Args:
            query: 查询文本
            answer: LLM生成的答案
            task_type: 任务类型
        """
        if task_type != "generation":
            return  # 仅缓存生成任务

        query_emb = self._embed(query)
        if query_emb is None:
            return

        with self._lock:
            self._id_counter += 1
            self._store.set(
                str(self._id_counter),
                CacheEntry(
                    query=query,
                    query_embedding=query_emb,
                    answer=answer,
                    timestamp=time.time(),
                    task_type=task_type,
                ),
            )

    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total * 100) if total > 0 else 0.0
            return {
                "total_entries": len(self._store.items()),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(hit_rate, 1),
                "threshold": self.threshold,
                "ttl": self.ttl,
            }

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._store.clear()
            self._hit_count = 0
            self._miss_count = 0
            logger.info("[SemanticCache] 缓存已清空")


# 全局单例
_cache_instance: Optional[SemanticCache] = None
_init_lock = threading.Lock()  # 保护单例创建


def get_semantic_cache() -> Optional[SemanticCache]:
    """获取全局语义缓存实例

    根据 ENABLE_SEMANTIC_CACHE 配置决定是否启用。
    """
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    import os
    with _init_lock:
        if _cache_instance is not None:
            return _cache_instance

        enabled = os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() == "true"
        if not enabled:
            return None

        threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", str(DEFAULT_THRESHOLD)))
        ttl = int(os.getenv("SEMANTIC_CACHE_TTL", str(DEFAULT_TTL)))

        _cache_instance = SemanticCache(threshold=threshold, ttl=ttl)
        return _cache_instance
