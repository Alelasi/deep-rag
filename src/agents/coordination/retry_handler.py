#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重试处理器（Retry Handler）- 失败重试与熔断

功能：
1. 指数退避重试（Exponential Backoff）
2. 熔断器模式（Circuit Breaker）
3. 失败统计与追踪
"""

import time
from typing import Callable, Dict, Any
from enum import Enum
from datetime import datetime, timedelta

try:
    from src.logging_config import get_logger
except Exception:
    import logging
    def get_logger(n):  # type: ignore
        return logging.getLogger(n)
logger = get_logger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 关闭（正常）
    OPEN = "open"          # 打开（熔断）
    HALF_OPEN = "half_open" # 半开（尝试恢复）


class RetryHandler:
    """重试处理器"""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        """
        Args:
            max_retries: 最大重试次数
            backoff_factor: 退避因子（每次重试等待时间 = backoff_factor ^ retry_count）
            circuit_breaker_threshold: 熔断阈值（连续失败次数）
            circuit_breaker_timeout: 熔断超时时间（秒），超时后尝试恢复
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout

        # 熔断器状态：{agent_name: CircuitState}
        self._circuit_states: Dict[str, CircuitState] = {}

        # 失败计数：{agent_name: count}
        self._failure_counts: Dict[str, int] = {}

        # 熔断时间：{agent_name: timestamp}
        self._circuit_open_times: Dict[str, datetime] = {}

        # 失败历史：[{agent, error, timestamp}, ...]
        self._failure_history = []

    def execute_with_retry(
        self,
        func: Callable,
        agent_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        带重试的执行函数

        Args:
            func: 要执行的函数
            agent_name: Agent名称（用于熔断器）
            *args, **kwargs: 传递给函数的参数

        Returns:
            函数执行结果

        Raises:
            Exception: 重试失败后抛出最后一次的异常
        """
        # 检查熔断器
        if self.is_circuit_open(agent_name):
            raise Exception(f"Circuit breaker is OPEN for {agent_name}")

        last_exception = None

        for retry_count in range(self.max_retries + 1):
            try:
                # 执行函数
                result = func(*args, **kwargs)

                # 成功 - 重置失败计数
                if retry_count > 0:
                    logger.info(f"✅ {agent_name} succeeded after {retry_count} retries")

                self._reset_failure_count(agent_name)
                return result

            except Exception as e:
                last_exception = e

                # 记录失败
                self._record_failure(agent_name, e)

                # 最后一次重试失败
                if retry_count == self.max_retries:
                    logger.error(f"❌ {agent_name} failed after {self.max_retries} retries: {e}")
                    break

                # 计算等待时间（指数退避）
                wait_time = self.backoff_factor ** retry_count
                logger.warning(
                    f"⚠️ {agent_name} failed (attempt {retry_count + 1}/{self.max_retries + 1}), "
                    f"retrying in {wait_time:.1f}s: {e}"
                )

                time.sleep(wait_time)

        # 所有重试失败 - 抛出异常
        raise last_exception

    def is_circuit_open(self, agent_name: str) -> bool:
        """
        检查熔断器是否打开

        Args:
            agent_name: Agent名称

        Returns:
            True 表示熔断器打开（拒绝请求）
        """
        state = self._circuit_states.get(agent_name, CircuitState.CLOSED)

        if state == CircuitState.CLOSED:
            return False

        if state == CircuitState.OPEN:
            # 检查是否超时（可以尝试恢复）
            open_time = self._circuit_open_times.get(agent_name)
            if open_time and datetime.now() - open_time > timedelta(seconds=self.circuit_breaker_timeout):
                # 进入半开状态
                self._circuit_states[agent_name] = CircuitState.HALF_OPEN
                logger.info(f"🔄 Circuit breaker for {agent_name} is now HALF_OPEN (attempting recovery)")
                return False

            return True

        # HALF_OPEN 状态 - 允许尝试
        return False

    def _record_failure(self, agent_name: str, error: Exception):
        """记录失败"""
        # 增加失败计数
        self._failure_counts[agent_name] = self._failure_counts.get(agent_name, 0) + 1

        # 记录失败历史
        self._failure_history.append({
            "agent": agent_name,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })

        # 检查是否需要打开熔断器
        if self._failure_counts[agent_name] >= self.circuit_breaker_threshold:
            self._open_circuit(agent_name)

    def _open_circuit(self, agent_name: str):
        """打开熔断器"""
        self._circuit_states[agent_name] = CircuitState.OPEN
        self._circuit_open_times[agent_name] = datetime.now()
        logger.error(f"🚨 Circuit breaker OPEN for {agent_name} (failed {self._failure_counts[agent_name]} times)")

    def _reset_failure_count(self, agent_name: str):
        """重置失败计数"""
        if agent_name in self._failure_counts:
            self._failure_counts[agent_name] = 0

        # 关闭熔断器
        if agent_name in self._circuit_states:
            self._circuit_states[agent_name] = CircuitState.CLOSED

    def get_metrics(self) -> Dict:
        """
        获取重试指标

        Returns:
            {
                "total_failures": 总失败次数,
                "failures_by_agent": {agent: count},
                "circuit_breaker_status": {agent: state}
            }
        """
        failures_by_agent = {}
        for failure in self._failure_history:
            agent = failure["agent"]
            failures_by_agent[agent] = failures_by_agent.get(agent, 0) + 1

        circuit_status = {
            agent: state.value
            for agent, state in self._circuit_states.items()
        }

        return {
            "total_failures": len(self._failure_history),
            "failures_by_agent": failures_by_agent,
            "circuit_breaker_status": circuit_status
        }

    def get_failure_history(self, agent_name: str = None) -> list:
        """
        获取失败历史

        Args:
            agent_name: 可选，只返回指定Agent的历史

        Returns:
            失败历史列表
        """
        if agent_name:
            return [f for f in self._failure_history if f["agent"] == agent_name]
        return self._failure_history.copy()

    def clear(self):
        """清空所有数据（用于测试）"""
        self._circuit_states.clear()
        self._failure_counts.clear()
        self._circuit_open_times.clear()
        self._failure_history.clear()
