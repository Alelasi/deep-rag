"""检索缓存层 — BM25索引缓存 + 文档列表缓存 + LLM响应缓存

v2.7优化：解决每次查询重建BM25索引导致~10秒浪费的问题
v2.8优化：BM25索引持久化到磁盘，避免重启后重建
"""
import os
import time
import hashlib
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Optional

log = logging.getLogger("deeprag.cache")

# === 磁盘缓存路径 ===
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === LRU缓存（最大容量限制）===
_MAX_CACHE_SIZE = 128
_doc_cache: OrderedDict = OrderedDict()  # collection_name -> (docs, timestamp)
_bm25_cache: OrderedDict = OrderedDict()  # collection_name -> (bm25_instance, timestamp)
_llm_cache: OrderedDict = OrderedDict()   # cache_key -> (response, timestamp)

_CACHE_TTL = 3600  # 1小时过期


def _touch(cache: OrderedDict, key):
    """LRU: 移到末尾（最近使用）"""
    if key in cache:
        cache.move_to_end(key)


def _evict(cache: OrderedDict):
    """LRU: 超容量时淘汰最久未使用的"""
    while len(cache) > _MAX_CACHE_SIZE:
        k, _ = cache.popitem(last=False)
        log.debug(f"Cache evicted: {k}")


# === 文档列表缓存（内存 + 磁盘持久化）===

def _doc_cache_path(collection_name: str) -> Path:
    """文档缓存文件路径"""
    return CACHE_DIR / f"docs_{collection_name}.pkl"


def get_cached_documents(collection_name: str, fetcher_fn) -> list:
    """获取collection的所有文档（带缓存 + 磁盘持久化）

    首次调用会通过fetcher_fn获取，后续直接返回缓存。
    重启后从磁盘加载，避免重新读取（节省~5秒）。
    """
    now = time.time()

    # 1. 检查内存缓存
    if collection_name in _doc_cache:
        docs, ts = _doc_cache[collection_name]
        if now - ts < _CACHE_TTL:
            _touch(_doc_cache, collection_name)
            log.debug(f"Doc cache HIT (memory): {collection_name} ({len(docs)} docs)")
            return docs
        else:
            del _doc_cache[collection_name]
            log.debug(f"Doc cache EXPIRED (memory): {collection_name}")

    # 2. 检查磁盘缓存
    cache_path = _doc_cache_path(collection_name)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                doc_data = pickle.load(f)
            docs = doc_data["docs"]
            ts = doc_data["timestamp"]
            if now - ts < _CACHE_TTL * 24:  # 磁盘缓存 TTL 更长（24小时）
                _doc_cache[collection_name] = (docs, ts)
                log.info(f"Doc cache HIT (disk): {collection_name} ({len(docs)} docs)")
                return docs
            else:
                log.debug(f"Doc cache EXPIRED (disk): {collection_name}")
        except Exception as e:
            log.warning(f"Doc disk cache load failed: {e}")

    # 3. 从数据源获取
    log.info(f"Doc cache MISS: {collection_name}, fetching...")
    docs = fetcher_fn()
    _doc_cache[collection_name] = (docs, now)
    _evict(_doc_cache)

    # 4. 持久化到磁盘
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"docs": docs, "timestamp": now}, f)
        log.info(f"Doc cache saved to disk: {cache_path}")
    except Exception as e:
        log.warning(f"Doc disk cache save failed: {e}")

    log.info(f"Doc cache LOADED: {collection_name} ({len(docs)} docs)")
    return docs


# === BM25索引缓存（内存 + 磁盘持久化）===

def _bm25_cache_path(collection_name: str) -> Path:
    """BM25 缓存文件路径"""
    return CACHE_DIR / f"bm25_{collection_name}.pkl"


def get_cached_bm25(collection_name: str, builder_fn):
    """获取BM25检索器（带缓存 + 磁盘持久化）

    首次调用构建BM25索引，后续直接返回缓存实例。
    重启后从磁盘加载，避免重建（节省~10秒）。
    """
    now = time.time()

    # 1. 检查内存缓存
    if collection_name in _bm25_cache:
        bm25, ts = _bm25_cache[collection_name]
        if now - ts < _CACHE_TTL:
            _touch(_bm25_cache, collection_name)
            log.debug(f"BM25 cache HIT (memory): {collection_name}")
            return bm25
        else:
            del _bm25_cache[collection_name]
            log.debug(f"BM25 cache EXPIRED (memory): {collection_name}")

    # 2. 检查磁盘缓存
    cache_path = _bm25_cache_path(collection_name)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                bm25_data = pickle.load(f)
            bm25 = bm25_data["bm25"]
            ts = bm25_data["timestamp"]
            if now - ts < _CACHE_TTL * 24:  # 磁盘缓存 TTL 更长（24小时）
                _bm25_cache[collection_name] = (bm25, ts)
                log.info(f"BM25 cache HIT (disk): {collection_name}")
                return bm25
            else:
                log.debug(f"BM25 cache EXPIRED (disk): {collection_name}")
        except Exception as e:
            log.warning(f"BM25 disk cache load failed: {e}")

    # 3. 重建索引
    log.info(f"BM25 cache MISS: {collection_name}, building index...")
    bm25 = builder_fn()
    _bm25_cache[collection_name] = (bm25, now)
    _evict(_bm25_cache)

    # 4. 持久化到磁盘
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"bm25": bm25, "timestamp": now}, f)
        log.info(f"BM25 cache saved to disk: {cache_path}")
    except Exception as e:
        log.warning(f"BM25 disk cache save failed: {e}")

    log.info(f"BM25 cache BUILT: {collection_name}")
    return bm25


# === LLM响应缓存 ===

def _make_cache_key(question: str, context: str, model: str = "") -> str:
    """生成缓存键"""
    raw = f"{question}|||{context[:500]}|||{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached_llm_response(cache_key: str) -> Optional[str]:
    """获取缓存的LLM响应"""
    if cache_key in _llm_cache:
        resp, ts = _llm_cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            _touch(_llm_cache, cache_key)
            log.debug(f"LLM cache HIT: {cache_key[:8]}")
            return resp
        else:
            del _llm_cache[cache_key]
    return None


def set_cached_llm_response(cache_key: str, response: str):
    """缓存LLM响应"""
    _llm_cache[cache_key] = (response, time.time())
    _evict(_llm_cache)


def make_llm_cache_key(question: str, context: str, model: str = "") -> str:
    """公开接口：生成LLM缓存键"""
    return _make_cache_key(question, context, model)


# === 缓存管理 ===

def invalidate_collection(collection_name: str):
    """失效某个collection的缓存（文档更新时调用）"""
    for cache in [_doc_cache, _bm25_cache]:
        if collection_name in cache:
            del cache[collection_name]
            log.info(f"Cache invalidated: {collection_name}")


def get_cache_stats() -> dict:
    """获取缓存统计"""
    return {
        "doc_cache_size": len(_doc_cache),
        "bm25_cache_size": len(_bm25_cache),
        "llm_cache_size": len(_llm_cache),
        "max_cache_size": _MAX_CACHE_SIZE,
        "ttl_seconds": _CACHE_TTL,
    }
