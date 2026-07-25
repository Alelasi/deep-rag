"""语义缓存 — v2.9.1新增

基于向量相似度的LLM响应缓存。
问题向量化 → 相似度搜索 → 命中则跳过LLM调用。

设计要点：
- 使用项目已有的 bge-base-zh-v1.5 embedding模型（不额外加载）
- 阈值默认0.90（FAQ类问题缓存1小时）
- 仅缓存LLM回答，不缓存RAG检索结果
- 内存缓存，无需持久化（进程重启自动清空）
"""
import time
import logging
import threading
from typing import Optional
from dataclasses import dataclass, field

log = logging.getLogger("deeprag")

# 默认配置
DEFAULT_THRESHOLD = 0.90
DEFAULT_TTL = 3600  # 1小时
DEFAULT_MAX_ENTRIES = 500  # 最大缓存条目数


@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    query_embedding: list[float]
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
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        ttl: int = DEFAULT_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self._cache: list[CacheEntry] = []
        self._lock = threading.Lock()
        self._embedder = None
        self._hit_count = 0
        self._miss_count = 0

        log.info(
            f"[SemanticCache] 初始化: threshold={threshold}, ttl={ttl}s, max_entries={max_entries}"
        )

    def _get_embedder(self):
        """获取embedding模型（懒加载，复用全局缓存）"""
        if self._embedder is None:
            try:
                from src.ui.model_cache import get_embedding_model
                from src.config import EMBEDDING_MODEL, DEVICE
                self._embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)
            except Exception as e:
                log.warning(f"[SemanticCache] 无法加载embedding模型: {e}")
                return None
        return self._embedder

    def _embed(self, query: str) -> Optional[list[float]]:
        """将查询文本转向量"""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            emb = embedder.encode([query])
            return emb[0].tolist()
        except Exception as e:
            log.warning(f"[SemanticCache] embedding失败: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
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
            best_entry = None
            best_score = 0.0

            for entry in self._cache:
                # 检查TTL
                if now - entry.timestamp > self.ttl:
                    continue

                # 计算相似度
                score = self._cosine_similarity(query_emb, entry.query_embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry and best_score >= self.threshold:
                best_entry.hit_count += 1
                self._hit_count += 1
                log.debug(
                    f"[SemanticCache] 命中: score={best_score:.3f}, "
                    f"hit_count={best_entry.hit_count}"
                )
                return best_entry.answer

            self._miss_count += 1
            return None

    def set(self, query: str, answer: str, task_type: str = "generation"):
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
            # 检查容量，LRU淘汰
            if len(self._cache) >= self.max_entries:
                # 按hit_count和timestamp排序，淘汰最不活跃的
                self._cache.sort(key=lambda e: (e.hit_count, e.timestamp))
                self._cache.pop(0)
                log.debug("[SemanticCache] LRU淘汰一条旧缓存")

            self._cache.append(
                CacheEntry(
                    query=query,
                    query_embedding=query_emb,
                    answer=answer,
                    timestamp=time.time(),
                    task_type=task_type,
                )
            )

    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total * 100) if total > 0 else 0.0
            return {
                "total_entries": len(self._cache),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(hit_rate, 1),
                "threshold": self.threshold,
                "ttl": self.ttl,
            }

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hit_count = 0
            self._miss_count = 0
            log.info("[SemanticCache] 缓存已清空")


# 全局单例
_cache_instance: Optional[SemanticCache] = None


def get_semantic_cache() -> Optional[SemanticCache]:
    """获取全局语义缓存实例

    根据 ENABLE_SEMANTIC_CACHE 配置决定是否启用。
    """
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    import os
    enabled = os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() == "true"
    if not enabled:
        return None

    threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", str(DEFAULT_THRESHOLD)))
    ttl = int(os.getenv("SEMANTIC_CACHE_TTL", str(DEFAULT_TTL)))

    _cache_instance = SemanticCache(threshold=threshold, ttl=ttl)
    return _cache_instance
