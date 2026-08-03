"""
错误重试模块（Retry with Backoff）
实现指数退避的智能重试机制
"""

import time
from typing import Callable, Any, Optional, List, Type
from functools import wraps


try:
    from src.logging_config import get_logger
except Exception:
    import logging

    def get_logger(n):  # type: ignore
        return logging.getLogger(n)

logger = get_logger(__name__)

class RetryConfig:
    """重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        初始化重试配置

        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟（秒）
            max_delay: 最大延迟（秒）
            exponential_base: 指数基数（2表示每次翻倍）
            jitter: 是否添加随机抖动（避免雷鸣群效应）
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class RetryManager:
    """
    重试管理器（指数退避算法）

    核心能力：
    1. 指数退避：1s, 2s, 4s, 8s, 16s...
    2. 随机抖动：避免多个请求同时重试
    3. 错误分类：可重试 vs 不可重试
    4. 重试统计：记录重试次数和原因

    示例：
    第1次失败 → 等待1s → 重试
    第2次失败 → 等待2s → 重试
    第3次失败 → 等待4s → 重试
    """

    # 可重试的异常类型
    RETRYABLE_ERRORS = [
        "timeout",
        "connection_error",
        "rate_limit",
        "service_unavailable"
    ]

    # 不可重试的异常类型
    NON_RETRYABLE_ERRORS = [
        "invalid_sql",
        "permission_denied",
        "authentication_failed"
    ]

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        初始化重试管理器

        Args:
            config: 重试配置，默认使用标准配置
        """
        self.config = config or RetryConfig()
        self.retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0
        }

    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        error_classifier: Optional[Callable[[Exception], str]] = None,
        **kwargs
    ) -> Any:
        """
        执行函数并在失败时重试

        Args:
            func: 要执行的函数
            *args: 函数参数
            error_classifier: 错误分类器（可选）
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果

        Raises:
            最后一次失败的异常
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # 尝试执行
                result = func(*args, **kwargs)

                # 成功
                if attempt > 0:
                    self.retry_stats["successful_retries"] += 1
                return result

            except Exception as e:
                last_exception = e

                # 第一次尝试失败
                if attempt == 0:
                    self.retry_stats["total_retries"] += 1

                # 判断是否可重试
                error_type = self._classify_error(e, error_classifier)

                if error_type in self.NON_RETRYABLE_ERRORS:
                    # 不可重试，直接抛出
                    raise

                # 达到最大重试次数
                if attempt >= self.config.max_retries:
                    self.retry_stats["failed_retries"] += 1
                    raise

                # 计算延迟时间
                delay = self._calculate_delay(attempt)

                logger.warning(f"⚠️ 重试 {attempt + 1}/{self.config.max_retries}: {error_type} - 等待 {delay:.1f}s")

                # 等待后重试
                time.sleep(delay)

        # 理论上不会到这里
        if last_exception:
            raise last_exception

    def _classify_error(
        self,
        error: Exception,
        classifier: Optional[Callable[[Exception], str]] = None
    ) -> str:
        """
        分类错误类型

        Args:
            error: 异常对象
            classifier: 自定义分类器

        Returns:
            错误类型字符串
        """
        if classifier:
            return classifier(error)

        # 默认分类逻辑
        error_msg = str(error).lower()

        if "timeout" in error_msg:
            return "timeout"
        elif "connection" in error_msg or "network" in error_msg:
            return "connection_error"
        elif "rate limit" in error_msg or "429" in error_msg:
            return "rate_limit"
        elif "503" in error_msg or "unavailable" in error_msg:
            return "service_unavailable"
        elif "sql" in error_msg or "syntax" in error_msg:
            return "invalid_sql"
        elif "permission" in error_msg or "denied" in error_msg:
            return "permission_denied"
        else:
            return "unknown_error"

    def _calculate_delay(self, attempt: int) -> float:
        """
        计算延迟时间（指数退避 + 随机抖动）

        Args:
            attempt: 当前重试次数（从0开始）

        Returns:
            延迟时间（秒）
        """
        # 指数退避
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)

        # 限制最大延迟
        delay = min(delay, self.config.max_delay)

        # 添加随机抖动（±25%）
        if self.config.jitter:
            import random
            jitter_factor = random.uniform(0.75, 1.25)
            delay *= jitter_factor

        return delay

    def get_stats(self) -> dict:
        """获取重试统计"""
        return self.retry_stats.copy()

    def reset_stats(self):
        """重置统计"""
        self.retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0
        }


# ============================================================================
# 装饰器版本（简化使用）
# ============================================================================

def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential_base: float = 2.0
):
    """
    重试装饰器

    使用示例：
    @retry(max_retries=3, base_delay=1.0)
    def call_api():
        # 可能失败的操作
        pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                exponential_base=exponential_base
            )
            manager = RetryManager(config)
            return manager.retry_with_backoff(func, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# 使用示例
# ============================================================================

def demo_retry_manager():
    """演示重试管理器"""
    manager = RetryManager(RetryConfig(max_retries=3, base_delay=1.0))

    # 测试1: 模拟临时失败（最终成功）
    logger.info("=" * 60)
    logger.error("测试1: 临时失败（第3次成功）")
    logger.info("=" * 60)

    attempt_count = [0]  # 使用列表避免闭包问题

    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise Exception("Timeout: connection timeout")
        return "成功!"

    try:
        result = manager.retry_with_backoff(flaky_function)
        logger.info(f"✅ 最终结果: {result}")
    except Exception as e:
        logger.error(f"❌ 失败: {str(e)}")

    # 测试2: 不可重试错误
    logger.info("\n" + "=" * 60)
    logger.error("测试2: 不可重试错误（立即失败）")
    logger.info("=" * 60)

    def non_retryable_function():
        raise Exception("Invalid SQL syntax")

    try:
        result = manager.retry_with_backoff(non_retryable_function)
    except Exception as e:
        logger.error(f"❌ 立即失败（不重试）: {str(e)}")

    # 统计
    logger.info("\n" + "=" * 60)
    logger.info("重试统计")
    logger.info("=" * 60)
    stats = manager.get_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


def demo_retry_decorator():
    """演示重试装饰器"""
    logger.info("\n" + "=" * 60)
    logger.info("测试装饰器版本")
    logger.info("=" * 60)

    @retry(max_retries=3, base_delay=0.5)
    def api_call():
        import random
        if random.random() < 0.7:  # 70%失败率
            raise Exception("Connection timeout")
        return "API调用成功"

    try:
        result = api_call()
        logger.info(f"✅ {result}")
    except Exception as e:
        logger.error(f"❌ 最终失败: {str(e)}")


if __name__ == "__main__":
    demo_retry_manager()
    demo_retry_decorator()
