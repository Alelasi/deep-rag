"""LLM限流器 + 自动重试 + 全局串行化 — 解决429速率限制问题

v2.8优化：
1. 全局串行锁：保证同一时刻只有一个LLM请求在执行（彻底消除429）
2. 指数退避重试：429时自动等待重试，不崩溃
3. LLM实例缓存：避免每次调用都创建新实例（线程安全）
4. 请求队列：多线程调用LLM时自动排队，一次只执行一个

设计原则：
- 用户要求"要么合并，要么排队" → 这里用排队（全局锁）
- doc_grader 用合并（5篇文档合为1次LLM调用）
- 其他LLM调用（generator等）用排队（全局锁保证串行）

v3.0改进（缓存与限流统一抽象）：
- 所有共享可变状态（串行锁持有者/计数、实例缓存字典）均用 threading.Lock 保护
- 日志统一使用标准库 logging.getLogger(__name__)
- 补充类型注解
"""
import time
import logging
import threading
from functools import wraps
from contextlib import contextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# === 全局串行锁（v2.8：一次只允许一个LLM请求）===

_llm_lock = threading.Lock()          # 全局串行锁（LLM请求排队）
_stats_lock = threading.Lock()        # 保护锁统计信息（_llm_lock_holder / _lock_acquire_count）
_llm_lock_holder: Optional[int] = None  # 当前持有锁的线程标识（调试用）
_lock_acquire_count: int = 0            # 统计排队次数


@contextmanager
def llm_serial():
    """全局LLM串行上下文管理器

    用法：
        with llm_serial():
            response = llm.invoke(messages)

    保证同一时刻只有一个线程在调用LLM，其他线程自动排队等待。
    彻底消除429速率限制问题。
    """
    global _llm_lock_holder, _lock_acquire_count
    thread_id = threading.current_thread().ident

    if _llm_lock_holder == thread_id:
        # 同一线程已持有锁（重入），直接执行
        yield
        return

    logger.debug(f"[LLM Lock] 线程 {thread_id} 等待锁...")
    _llm_lock.acquire()
    with _stats_lock:
        _lock_acquire_count += 1
        _llm_lock_holder = thread_id
    logger.debug(f"[LLM Lock] 线程 {thread_id} 获得锁 (排队序号: {_lock_acquire_count})")
    try:
        yield
    finally:
        with _stats_lock:
            _llm_lock_holder = None
        _llm_lock.release()
        logger.debug(f"[LLM Lock] 线程 {thread_id} 释放锁")


def get_lock_stats() -> dict:
    """获取锁统计信息（线程安全）"""
    with _stats_lock:
        holder = _llm_lock_holder
        count = _lock_acquire_count
    return {
        "current_holder": holder,
        "total_acquisitions": count,
        "is_locked": _llm_lock.locked(),
    }


# === 指数退避重试（v2.8：集成全局串行锁）===

def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0
) -> Callable[..., Any]:
    """装饰器：LLM调用自动排队 + 429退避重试

    v2.8改进：
    - 自动获取全局串行锁（排队）
    - 429时指数退避重试
    - 一次只有一个LLM请求执行

    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒），实际延迟 = base_delay * 2^attempt
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error: Optional[BaseException] = None
            for attempt in range(max_retries + 1):
                try:
                    # v2.8: 全局串行 — 一次只执行一个LLM请求
                    with llm_serial():
                        return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()

                    # 判断是否为速率限制错误
                    is_rate_limit = (
                        "429" in err_str or
                        "rate limit" in err_str or
                        "速率限制" in err_str or
                        "请求频率" in err_str
                    )

                    if not is_rate_limit or attempt == max_retries:
                        raise

                    delay = base_delay * (2 ** attempt) + 0.5
                    logger.warning(
                        f"LLM 429 rate limit, retry {attempt + 1}/{max_retries} "
                        f"after {delay:.1f}s delay... ({str(e)[:80]})"
                    )
                    time.sleep(delay)

            if last_error is not None:
                raise last_error
            return None  # 理论上不可达，仅用于类型完整性
        return wrapper
    return decorator


# === LLM实例缓存（线程安全）===

_llm_instances: dict = {}
_llm_instance_lock = threading.Lock()  # 注意：与全局串行锁 _llm_lock 分离，避免互相阻塞


def get_cached_llm(
    backend: str, model: str, temperature: float, factory_fn: Callable[[], Any]
) -> Any:
    """缓存LLM实例，避免每次调用都创建新实例（线程安全，双检锁避免重复创建）

    Args:
        backend: LLM后端名称
        model: 模型名称
        temperature: 温度参数
        factory_fn: 创建LLM实例的函数
    """
    cache_key = f"{backend}:{model}:{temperature}"

    # 第一次检查（只读，不加锁创建）
    with _llm_instance_lock:
        if cache_key in _llm_instances:
            logger.debug(f"LLM instance cache HIT: {cache_key}")
            return _llm_instances[cache_key]

    # 未命中：在锁外创建实例，避免持锁执行耗时构建
    llm = factory_fn()

    # 第二次检查后写入（双检锁，避免并发重复创建）
    with _llm_instance_lock:
        if cache_key not in _llm_instances:
            _llm_instances[cache_key] = llm
            logger.info(f"LLM instance cache MISS: {cache_key}, created new instance")
        else:
            # 已有其他线程先行创建，复用之
            llm = _llm_instances[cache_key]
    return llm


def reset_llm_cache():
    """清除LLM实例缓存（配置变更时调用）"""
    with _llm_instance_lock:
        _llm_instances.clear()
        logger.info("LLM instance cache cleared")
