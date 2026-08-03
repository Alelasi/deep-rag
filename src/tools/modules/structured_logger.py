"""
结构化日志模块（Structured Logging）
记录Agent执行过程的详细日志
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from src.logging_config import get_logger
except Exception:
    import logging
    def get_logger(n):  # type: ignore
        return logging.getLogger(n)
logger = get_logger(__name__)


class StructuredLogger:
    """
    结构化日志记录器

    核心能力：
    1. JSON格式日志（易于解析和分析）
    2. 记录每步耗时（性能分析）
    3. 分级日志（INFO/WARNING/ERROR）
    4. 支持日志导出和分析

    日志格式：
    {
        "timestamp": "2026-06-05T15:30:00.123",
        "level": "INFO",
        "event": "agent_step",
        "data": {
            "iteration": 1,
            "action": "search_database",
            "latency_ms": 45.2
        }
    }
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        console_output: bool = True,
        level: str = "INFO"
    ):
        """
        初始化日志记录器

        Args:
            log_file: 日志文件路径（可选）
            console_output: 是否输出到控制台
            level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        """
        self.logger = logging.getLogger("agent_executor")
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()  # 清除已有的handlers

        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(console_handler)

        # 文件输出
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)

        # 性能追踪
        self.timers: Dict[str, float] = {}

    def start_timer(self, name: str):
        """开始计时"""
        self.timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """
        停止计时并返回耗时

        Returns:
            耗时（毫秒）
        """
        if name not in self.timers:
            return 0.0

        elapsed = (time.time() - self.timers[name]) * 1000  # 转为毫秒
        del self.timers[name]
        return round(elapsed, 2)

    def log_agent_step(
        self,
        iteration: int,
        thought: str,
        action: Optional[str] = None,
        observation: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        记录Agent单步执行

        Args:
            iteration: 迭代次数
            thought: Agent思考内容
            action: 执行的动作
            observation: 观察结果
            latency_ms: 耗时（毫秒）
            metadata: 额外元数据
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "event": "agent_step",
            "data": {
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation[:200] if observation else None,  # 截断过长内容
                "latency_ms": latency_ms,
                "metadata": metadata or {}
            }
        }

        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

    def log_agent_complete(
        self,
        success: bool,
        iterations: int,
        total_latency_ms: float,
        answer: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        记录Agent完成

        Args:
            success: 是否成功
            iterations: 总迭代次数
            total_latency_ms: 总耗时（毫秒）
            answer: 最终答案
            error: 错误信息（如果失败）
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO" if success else "ERROR",
            "event": "agent_complete",
            "data": {
                "success": success,
                "iterations": iterations,
                "total_latency_ms": total_latency_ms,
                "answer": answer,
                "error": error
            }
        }

        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

    def log_error(self, error_type: str, message: str, context: Optional[Dict] = None):
        """
        记录错误

        Args:
            error_type: 错误类型（llm_error/sql_error/timeout等）
            message: 错误信息
            context: 错误上下文
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "event": "error",
            "data": {
                "error_type": error_type,
                "message": message,
                "context": context or {}
            }
        }

        self.logger.error(json.dumps(log_entry, ensure_ascii=False))

    def log_warning(self, warning_type: str, message: str, context: Optional[Dict] = None):
        """
        记录警告

        Args:
            warning_type: 警告类型（retry/fallback等）
            message: 警告信息
            context: 警告上下文
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "event": "warning",
            "data": {
                "warning_type": warning_type,
                "message": message,
                "context": context or {}
            }
        }

        self.logger.warning(json.dumps(log_entry, ensure_ascii=False))

    def log_metric(self, metric_name: str, value: float, unit: str = "", metadata: Optional[Dict] = None):
        """
        记录性能指标

        Args:
            metric_name: 指标名称（如llm_latency/sql_latency）
            value: 指标值
            unit: 单位（如ms/bytes）
            metadata: 额外元数据
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "event": "metric",
            "data": {
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "metadata": metadata or {}
            }
        }

        self.logger.info(json.dumps(log_entry, ensure_ascii=False))


class LogAnalyzer:
    """
    日志分析器

    读取结构化日志并生成分析报告
    """

    @staticmethod
    def analyze_log_file(log_file: str) -> Dict[str, Any]:
        """
        分析日志文件

        Args:
            log_file: 日志文件路径

        Returns:
            分析报告
        """
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = [json.loads(line) for line in f if line.strip()]

        # 统计
        stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_iterations": 0,
            "avg_iterations": 0.0,
            "avg_latency_ms": 0.0,
            "errors": [],
            "warnings": []
        }

        latencies = []
        iterations_list = []

        for log in logs:
            event = log.get("event")

            if event == "agent_complete":
                stats["total_executions"] += 1

                data = log["data"]
                if data["success"]:
                    stats["successful_executions"] += 1
                else:
                    stats["failed_executions"] += 1

                iterations_list.append(data["iterations"])
                latencies.append(data["total_latency_ms"])

            elif event == "error":
                stats["errors"].append(log["data"])

            elif event == "warning":
                stats["warnings"].append(log["data"])

        # 计算平均值
        if iterations_list:
            stats["avg_iterations"] = round(sum(iterations_list) / len(iterations_list), 2)
        if latencies:
            stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)

        return stats


# ============================================================================
# 使用示例
# ============================================================================

def demo_structured_logger():
    """演示结构化日志"""
    # 创建日志记录器（输出到文件和控制台）
    slog = StructuredLogger(
        log_file="logs/agent_execution.log",
        console_output=True
    )

    logger.info("=" * 60)
    logger.info("模拟Agent执行并记录日志")
    logger.info("=" * 60)

    # 模拟Agent执行
    slog.start_timer("total")

    for i in range(1, 4):
        slog.start_timer(f"step_{i}")

        # 模拟执行
        time.sleep(0.1)

        step_latency = slog.stop_timer(f"step_{i}")

        slog.log_agent_step(
            iteration=i,
            thought=f"这是第{i}步的思考",
            action="search_database",
            observation=f"查询返回{i}条结果",
            latency_ms=step_latency
        )

    total_latency = slog.stop_timer("total")

    slog.log_agent_complete(
        success=True,
        iterations=3,
        total_latency_ms=total_latency,
        answer="最终答案"
    )

    # 记录指标
    slog.log_metric("llm_latency", 45.2, "ms")
    slog.log_metric("sql_latency", 12.8, "ms")

    logger.info("\n" + "=" * 60)
    logger.info("日志已保存到: logs/agent_execution.log")
    logger.info("=" * 60)


if __name__ == "__main__":
    demo_structured_logger()
