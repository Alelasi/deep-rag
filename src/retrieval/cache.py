"""检索缓存层 — BM25索引缓存 + 文档列表缓存 + LLM响应缓存

v2.7优化：解决每次查询重建BM25索引导致~10秒浪费的问题
v2.8优化：BM25索引持久化到磁盘，避免重启后重建

v3.0改进（缓存与限流统一抽象）：
- 提炼统一的 TTL 语义与可插拔后端抽象（CacheBackend / MemoryBackend / RedisBackend 钩子 / TTLCache）
- 所有共享可变状态（OrderedDict、TTLCache 内部字典）均用 Lock 保护
- 三个缓存入口共用同一套 TTLCache 抽象，避免各写一套过期/淘汰逻辑
- 日志统一使用标准库 logging.getLogger(__name__)
- 补充类型注解
"""
import os
import time
import abc
import threading
import hashlib
import logging
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# === 磁盘缓存路径 ===
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 统一缓存抽象：TTL 语义 + 可插拔后端
# ============================================================

class CacheBackend(abc.ABC):
    """缓存后端抽象接口（可插拔）。默认实现见 MemoryBackend。

    预留 Redis 等后端钩子：继承本类并实现以下方法即可接入。
    """

    @abc.abstractmethod
    def get(self, key: str) -> Any:
        """按 key 取值，未命中返回 None"""

    @abc.abstractmethod
    def set(self, key: str, value: Any) -> None:
        """写入 key-value"""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """删除 key"""

    @abc.abstractmethod
    def contains(self, key: str) -> bool:
        """判断 key 是否存在"""

    @abc.abstractmethod
    def clear(self) -> None:
        """清空全部条目"""

    @abc.abstractmethod
    def items(self) -> List[Tuple[str, Any]]:
        """返回全部 (key, value) 快照（用于枚举）"""

    def __contains__(self, key: str) -> bool:
        """支持 `key in backend` 语法（委托给 contains）。"""
        return self.contains(key)


class MemoryBackend(CacheBackend):
    """内存后端（默认实现），基于 dict，自带锁保护，线程安全。"""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def items(self) -> List[Tuple[str, Any]]:
        with self._lock:
            return list(self._store.items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class RedisBackend(CacheBackend):
    """Redis 后端占位（钩子）。

    说明：当前未接入 redis-py，调用任何方法都会抛出 NotImplementedError。
    接入方式：pip install redis 后，在此实现 get/set/delete/contains/clear/items，
    并让 TTLCache(backend=RedisBackend(...)) 即可无缝替换内存后端。
    注意：items() 在 Redis 上需通过 SCAN 实现，枚举语义与 MemoryBackend 保持一致。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "RedisBackend 尚未接入：请实现 get/set/delete/contains/clear/items "
            "以对接 redis-py，然后传入 TTLCache(backend=...)"
        )

    def get(self, key: str) -> Any:
        raise NotImplementedError("RedisBackend 未实现")

    def set(self, key: str, value: Any) -> None:
        raise NotImplementedError("RedisBackend 未实现")

    def delete(self, key: str) -> None:
        raise NotImplementedError("RedisBackend 未实现")

    def contains(self, key: str) -> bool:
        raise NotImplementedError("RedisBackend 未实现")

    def clear(self) -> None:
        raise NotImplementedError("RedisBackend 未实现")

    def items(self) -> List[Tuple[str, Any]]:
        raise NotImplementedError("RedisBackend 未实现")


@dataclass
class _Stored:
    """TTLCache 内部存储单元"""
    value: Any
    timestamp: float


class TTLCache:
    """带 TTL 语义的缓存（线程安全）。

    统一职责：
    - TTL 过期：get/items 时自动剔除过期条目
    - 可选 LRU 淘汰：超过 max_size 时淘汰最久未访问的条目
    - 可插拔后端：默认 MemoryBackend，可传入 RedisBackend 等

    Args:
        ttl: 过期秒数（0 或负数表示永不过期）
        max_size: 最大条目数（<=0 表示不限制）
        backend: 缓存后端，默认 MemoryBackend
        lru: 是否启用 LRU 淘汰
    """

    def __init__(
        self,
        ttl: int = 3600,
        max_size: int = 128,
        backend: Optional[CacheBackend] = None,
        lru: bool = True,
    ) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._backend = backend if backend is not None else MemoryBackend()
        self._lru = lru
        self._lock = threading.Lock()
        self._access: "OrderedDict[str, None]" = OrderedDict()  # LRU 访问顺序

    def _is_expired(self, timestamp: float) -> bool:
        if self._ttl is None or self._ttl <= 0:
            return False
        return (time.time() - timestamp) > self._ttl

    def _touch(self, key: str) -> None:
        if not self._lru:
            return
        self._access.pop(key, None)
        self._access[key] = None

    def get(self, key: str) -> Optional[Any]:
        """取值，过期或缺失返回 None（自动剔除过期条目）"""
        with self._lock:
            item = self._backend.get(key)
            if item is None:
                return None
            if self._is_expired(item.timestamp):
                self._backend.delete(key)
                self._access.pop(key, None)
                return None
            self._touch(key)
            return item.value

    def set(self, key: str, value: Any) -> None:
        """写入值（带当前时间戳）；超容量时按 LRU 淘汰"""
        with self._lock:
            self._touch(key)
            if (
                self._max_size > 0
                and key not in self._backend
                and len(self._backend) >= self._max_size
            ):
                self._evict()
            self._backend.set(key, _Stored(value=value, timestamp=time.time()))

    def delete(self, key: str) -> None:
        with self._lock:
            self._backend.delete(key)
            self._access.pop(key, None)

    def contains(self, key: str) -> bool:
        with self._lock:
            item = self._backend.get(key)
            if item is None:
                return False
            if self._is_expired(item.timestamp):
                self._backend.delete(key)
                self._access.pop(key, None)
                return False
            return True

    def clear(self) -> None:
        with self._lock:
            self._backend.clear()
            self._access.clear()

    def items(self) -> List[Tuple[str, Any]]:
        """返回未过期条目的 (key, value) 快照"""
        with self._lock:
            out: List[Tuple[str, Any]] = []
            expired: List[str] = []
            for k, item in self._backend.items():
                if self._is_expired(item.timestamp):
                    expired.append(k)
                    continue
                out.append((k, item.value))
            for k in expired:
                self._backend.delete(k)
                self._access.pop(k, None)
            return out

    def __len__(self) -> int:
        return len(self.items())

    def _evict(self) -> None:
        """淘汰最久未访问的条目（LRU）"""
        if not self._lru or not self._access:
            # 退化策略：直接清空后端（极端情况保护）
            self._backend.clear()
            self._access.clear()
            return
        oldest, _ = self._access.popitem(last=False)
        self._backend.delete(oldest)
        logger.debug(f"TTLCache LRU evicted: {oldest}")


# === LRU缓存（最大容量限制）===
_MAX_CACHE_SIZE = 128
_doc_cache: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()  # collection_name -> (docs, timestamp)
_bm25_cache: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()  # collection_name -> (bm25_instance, timestamp)
_cache_lock = threading.Lock()  # 保护上述两个 OrderedDict 的共享可变状态

_CACHE_TTL = 3600  # 1小时过期（内存缓存）
_DISK_TTL_MULTIPLIER = 24  # 磁盘缓存 TTL 更长（24小时）


def _touch(cache: "OrderedDict[str, Tuple[Any, float]]", key: str) -> None:
    """LRU: 移到末尾（最近使用）"""
    if key in cache:
        cache.move_to_end(key)


def _evict(cache: "OrderedDict[str, Tuple[Any, float]]") -> None:
    """LRU: 超容量时淘汰最久未使用的"""
    while len(cache) > _MAX_CACHE_SIZE:
        k, _ = cache.popitem(last=False)
        logger.debug(f"Cache evicted: {k}")


# === 文档列表缓存（内存 + 磁盘持久化）===

def _doc_cache_path(collection_name: str) -> Path:
    """文档缓存文件路径"""
    return CACHE_DIR / f"docs_{collection_name}.pkl"


def get_cached_documents(collection_name: str, fetcher_fn: Callable[[], Any]) -> list:
    """获取collection的所有文档（带缓存 + 磁盘持久化）

    首次调用会通过fetcher_fn获取，后续直接返回缓存。
    重启后从磁盘加载，避免重新读取（节省~5秒）。

    Args:
        collection_name: 集合名称
        fetcher_fn: 从数据源获取文档列表的回调
    """
    now = time.time()

    # 1. 检查内存缓存
    with _cache_lock:
        if collection_name in _doc_cache:
            docs, ts = _doc_cache[collection_name]
            if now - ts < _CACHE_TTL:
                _touch(_doc_cache, collection_name)
                logger.debug(f"Doc cache HIT (memory): {collection_name} ({len(docs)} docs)")
                return docs
            else:
                del _doc_cache[collection_name]
                logger.debug(f"Doc cache EXPIRED (memory): {collection_name}")

    # 2. 检查磁盘缓存
    cache_path = _doc_cache_path(collection_name)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                doc_data = pickle.load(f)
            docs = doc_data["docs"]
            ts = doc_data["timestamp"]
            if now - ts < _CACHE_TTL * _DISK_TTL_MULTIPLIER:  # 磁盘缓存 TTL 更长（24小时）
                with _cache_lock:
                    _doc_cache[collection_name] = (docs, ts)
                logger.info(f"Doc cache HIT (disk): {collection_name} ({len(docs)} docs)")
                return docs
            else:
                logger.debug(f"Doc cache EXPIRED (disk): {collection_name}")
        except Exception as e:
            logger.warning(f"Doc disk cache load failed: {e}")

    # 3. 从数据源获取
    logger.info(f"Doc cache MISS: {collection_name}, fetching...")
    docs = fetcher_fn()
    with _cache_lock:
        _doc_cache[collection_name] = (docs, now)
        _evict(_doc_cache)

    # 4. 持久化到磁盘
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"docs": docs, "timestamp": now}, f)
        logger.info(f"Doc cache saved to disk: {cache_path}")
    except Exception as e:
        logger.warning(f"Doc disk cache save failed: {e}")

    logger.info(f"Doc cache LOADED: {collection_name} ({len(docs)} docs)")
    return docs


# === BM25索引缓存（内存 + 磁盘持久化）===

def _bm25_cache_path(collection_name: str) -> Path:
    """BM25 缓存文件路径"""
    return CACHE_DIR / f"bm25_{collection_name}.pkl"


def get_cached_bm25(collection_name: str, builder_fn: Callable[[], Any]) -> Any:
    """获取BM25检索器（带缓存 + 磁盘持久化）

    首次调用构建BM25索引，后续直接返回缓存实例。
    重启后从磁盘加载，避免重建（节省~10秒）。

    Args:
        collection_name: 集合名称
        builder_fn: 构建 BM25 检索器的回调
    """
    now = time.time()

    # 1. 检查内存缓存
    with _cache_lock:
        if collection_name in _bm25_cache:
            bm25, ts = _bm25_cache[collection_name]
            if now - ts < _CACHE_TTL:
                _touch(_bm25_cache, collection_name)
                logger.debug(f"BM25 cache HIT (memory): {collection_name}")
                return bm25
            else:
                del _bm25_cache[collection_name]
                logger.debug(f"BM25 cache EXPIRED (memory): {collection_name}")

    # 2. 检查磁盘缓存
    cache_path = _bm25_cache_path(collection_name)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                bm25_data = pickle.load(f)
            bm25 = bm25_data["bm25"]
            ts = bm25_data["timestamp"]
            if now - ts < _CACHE_TTL * _DISK_TTL_MULTIPLIER:  # 磁盘缓存 TTL 更长（24小时）
                with _cache_lock:
                    _bm25_cache[collection_name] = (bm25, ts)
                logger.info(f"BM25 cache HIT (disk): {collection_name}")
                return bm25
            else:
                logger.debug(f"BM25 cache EXPIRED (disk): {collection_name}")
        except Exception as e:
            logger.warning(f"BM25 disk cache load failed: {e}")

    # 3. 重建索引
    logger.info(f"BM25 cache MISS: {collection_name}, building index...")
    bm25 = builder_fn()
    with _cache_lock:
        _bm25_cache[collection_name] = (bm25, now)
        _evict(_bm25_cache)

    # 4. 持久化到磁盘
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"bm25": bm25, "timestamp": now}, f)
        logger.info(f"BM25 cache saved to disk: {cache_path}")
    except Exception as e:
        logger.warning(f"BM25 disk cache save failed: {e}")

    logger.info(f"BM25 cache BUILT: {collection_name}")
    return bm25


# === LLM响应缓存（统一使用 TTLCache 抽象）===

_llm_cache: "TTLCache" = TTLCache(ttl=_CACHE_TTL, max_size=_MAX_CACHE_SIZE, lru=True)


def _make_cache_key(question: str, context: str, model: str = "") -> str:
    """生成缓存键"""
    raw = f"{question}|||{context[:500]}|||{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached_llm_response(cache_key: str) -> Optional[str]:
    """获取缓存的LLM响应"""
    resp = _llm_cache.get(cache_key)
    if resp is not None:
        logger.debug(f"LLM cache HIT: {cache_key[:8]}")
    return resp


def set_cached_llm_response(cache_key: str, response: str) -> None:
    """缓存LLM响应"""
    _llm_cache.set(cache_key, response)


def make_llm_cache_key(question: str, context: str, model: str = "") -> str:
    """公开接口：生成LLM缓存键"""
    return _make_cache_key(question, context, model)


# === 缓存管理 ===

def invalidate_collection(collection_name: str) -> None:
    """失效某个collection的缓存（文档更新时调用）"""
    with _cache_lock:
        for cache in [_doc_cache, _bm25_cache]:
            if collection_name in cache:
                del cache[collection_name]
                logger.info(f"Cache invalidated: {collection_name}")


def get_cache_stats() -> dict:
    """获取缓存统计"""
    with _cache_lock:
        doc_size = len(_doc_cache)
        bm25_size = len(_bm25_cache)
    return {
        "doc_cache_size": doc_size,
        "bm25_cache_size": bm25_size,
        "llm_cache_size": len(_llm_cache.items()),
        "max_cache_size": _MAX_CACHE_SIZE,
        "ttl_seconds": _CACHE_TTL,
    }
