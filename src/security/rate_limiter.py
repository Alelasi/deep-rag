"""进程内滑动窗口限流

适用：单实例 API（uvicorn 单 worker 或多线程共享本模块全局）。
多副本 / 多机部署请替换为 Redis 限流，接口保持 ``allow(key) -> (ok, remaining)``。
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class RateLimiter:
    """按 client key（通常为 IP）统计时间窗内请求次数。"""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        # 窗口内最大请求数；超出返回拒绝
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = float(window_seconds)
        # key -> 命中时间戳队列（仅保留窗内）
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> Tuple[bool, int]:
        """尝试放行一次请求。

        Returns:
            (是否允许, 剩余配额)；拒绝时剩余为 0
        """
        now = time.time()
        with self._lock:
            q = self._hits[key]
            cutoff = now - self.window_seconds
            # 弹出窗口外的旧时间戳
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                return False, 0
            q.append(now)
            remaining = self.max_requests - len(q)
            return True, max(0, remaining)


# 模块级单例：全 API 进程共享同一计数器
_limiter: RateLimiter | None = None
# 保护懒加载单例的创建，避免多线程并发时重复实例化
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """懒加载全局限流器；参数来自 RATE_LIMIT_PER_MINUTE。

    使用双检锁保证线程安全：只有首次创建会进入锁区。
    """
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                max_req = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
                _limiter = RateLimiter(max_requests=max_req, window_seconds=60.0)
    return _limiter
