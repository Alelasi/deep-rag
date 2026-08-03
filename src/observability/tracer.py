"""
可观测性模块 - LangFuse集成

功能：
1. 分布式追踪：追踪每个节点的执行时间
2. LLM调用监控：tokens、cost、latency
3. 性能分析：识别瓶颈
4. 错误追踪：堆栈信息

设计要点（生产化）：
- 默认结构化 console 输出（JSON 行），便于日志采集（Loki/ES 等）。
- langfuse 懒导入守卫：所有 `import langfuse` 均放在函数/方法内部，
  且仅当 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 存在时才导入与初始化；
  缺失或不可用时静默降级为本地 console，绝不因缺少 langfuse 依赖而崩溃。

使用方式：
```python
from src.observability.tracer import trace_node

@trace_node(name="retrieve")
def node_retrieve(state):
    # 你的代码
    return result
```
"""
import logging
import time
import functools
import os
import json
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

log = logging.getLogger(__name__)


# ========== 结构化日志辅助 ==========

def _emit(level: int, event: str, **fields: Any) -> None:
    """以 JSON 行输出结构化日志，便于集中采集。

    所有字段（含 event）序列化为单行 JSON；日志级别仍由标准 logging 控制。
    """
    payload: Dict[str, Any] = {"event": event}
    payload.update(fields)
    log.log(level, json.dumps(payload, ensure_ascii=False, default=str))


# ========== langfuse 懒导入守卫 ==========

# 模块级缓存（非导入）；None 表示未配置/不可用。
_LANGFUSE_MODULES: Optional[Dict[str, Any]] = None


def _load_langfuse() -> Optional[Dict[str, Any]]:
    """懒导入 langfuse，且仅在配置了密钥时执行。

    返回 {"Langfuse":..., "observe":..., "langfuse_context":...} 或 None。
    缺少依赖或缺少配置时返回 None（静默降级），绝不抛出异常。
    """
    global _LANGFUSE_MODULES
    if _load_langfuse._resolved:  # type: ignore[attr-defined]
        # 已解析过（成功或失败都缓存），直接返回缓存值
        return _LANGFUSE_MODULES

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        log.debug("LANGFUSE_PUBLIC_KEY/SECRET 未配置，跳过 langfuse（降级为 console）")
        _LANGFUSE_MODULES = None
        _load_langfuse._resolved = True  # type: ignore[attr-defined]
        return None

    try:
        from langfuse import Langfuse
        from langfuse.decorators import observe, langfuse_context
        _LANGFUSE_MODULES = {
            "Langfuse": Langfuse,
            "observe": observe,
            "langfuse_context": langfuse_context,
        }
    except ImportError:
        log.debug("langfuse 未安装，跳过（降级为 console）")
        _LANGFUSE_MODULES = None

    _load_langfuse._resolved = True  # type: ignore[attr-defined]
    return _LANGFUSE_MODULES


# 标注解析状态（避免每次都重跑判断）
_load_langfuse._resolved = False  # type: ignore[attr-defined]


# ========== 1. Tracer基类 ==========

class Tracer:
    """追踪器基类（支持多种后端）"""

    def __init__(self, backend: str = "console"):
        """
        Args:
            backend: console / langfuse
        """
        self.backend: str = backend
        self.enabled: bool = True
        self._langfuse_client: Optional[Any] = None

        if backend == "langfuse":
            if self._init_langfuse() is None:
                log.info("LangFuse 未配置/不可用，降级为 console")
                self.backend = "console"

    def _init_langfuse(self) -> Optional[Any]:
        """初始化 LangFuse（懒导入，密钥缺失或不可用时返回 None）。"""
        lf = _load_langfuse()
        if lf is None:
            return None

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            log.info("LangFuse 密钥未设置，降级为 console")
            return None

        try:
            self._langfuse_client = lf["Langfuse"](
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            log.info("LangFuse 初始化成功")
            return self._langfuse_client
        except Exception as e:
            log.error(f"LangFuse 初始化失败，降级为 console: {e}")
            self._langfuse_client = None
            return None

    def trace_node(self, name: str) -> Callable:
        """装饰器：追踪节点执行"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.enabled:
                    return func(*args, **kwargs)

                start_time = time.time()
                error: Optional[Exception] = None
                result: Any = None

                try:
                    lf = _load_langfuse() if self.backend == "langfuse" else None
                    if self.backend == "langfuse" and lf is not None and self._langfuse_client is not None:
                        observed_func = lf["observe"](name=name)(func)
                        result = observed_func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                except Exception as e:
                    error = e
                    raise
                finally:
                    elapsed = time.time() - start_time
                    self._log_trace(name, elapsed, error)

                return result

            return wrapper
        return decorator

    def _log_trace(self, name: str, elapsed: float, error: Optional[Exception]) -> None:
        """记录追踪日志（结构化 JSON 行）"""
        status = "ERROR" if error else "OK"
        _emit(
            logging.INFO,
            "trace",
            node=name,
            latency_ms=round(elapsed * 1000, 3),
            status=status,
        )

        if error:
            _emit(
                logging.ERROR,
                "trace_error",
                node=name,
                error_type=type(error).__name__,
                error=str(error),
            )

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """上下文管理器：追踪代码块"""
        start_time = time.time()
        error: Optional[Exception] = None

        lf = _load_langfuse() if self.backend == "langfuse" else None
        try:
            if self.backend == "langfuse" and lf is not None and self._langfuse_client is not None:
                with lf["langfuse_context"].observe(name=name) as span:
                    if metadata:
                        span.update(metadata=metadata)
                    yield span
            else:
                yield None
        except Exception as e:
            error = e
            raise
        finally:
            elapsed = time.time() - start_time
            self._log_trace(name, elapsed, error)


# ========== 2. 全局Tracer实例 ==========

# 从环境变量读取配置
OBSERVABILITY_BACKEND = os.getenv("OBSERVABILITY_BACKEND", "console")  # console / langfuse
OBSERVABILITY_ENABLED = os.getenv("OBSERVABILITY_ENABLED", "true").lower() == "true"

tracer = Tracer(backend=OBSERVABILITY_BACKEND)
tracer.enabled = OBSERVABILITY_ENABLED


# ========== 3. 便捷装饰器 ==========

def trace_node(name: Optional[str] = None) -> Callable:
    """
    便捷装饰器：追踪节点执行

    使用方式：
    ```python
    @trace_node("retrieve")
    def node_retrieve(state):
        return result
    ```
    """
    def decorator(func: Callable) -> Callable:
        node_name = name or func.__name__
        return tracer.trace_node(node_name)(func)

    return decorator


def trace_llm_call(model: str = "unknown") -> Callable:
    """
    装饰器：追踪LLM调用

    使用方式：
    ```python
    @trace_llm_call(model="gpt-4")
    def call_llm(prompt):
        return llm.invoke(prompt)
    ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.span(f"llm_call_{model}") as span:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time

                    # 尝试提取tokens信息
                    tokens: Optional[Dict[str, Any]] = None
                    if hasattr(result, 'response_metadata'):
                        tokens = result.response_metadata.get('token_usage', {})

                    # 记录元数据
                    if span and tokens:
                        span.update(metadata={
                            'model': model,
                            'latency_ms': elapsed * 1000,
                            'input_tokens': tokens.get('prompt_tokens', 0),
                            'output_tokens': tokens.get('completion_tokens', 0),
                            'total_tokens': tokens.get('total_tokens', 0)
                        })

                    _emit(
                        logging.INFO,
                        "llm_call",
                        model=model,
                        latency_ms=round(elapsed * 1000, 3),
                        tokens=tokens,
                    )

                    return result

                except Exception as e:
                    _emit(
                        logging.ERROR,
                        "llm_call_error",
                        model=model,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    raise

        return wrapper
    return decorator


# ========== 4. 性能监控 ==========

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def record(self, name: str, value: float, unit: str = "ms") -> None:
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = {
                'values': [],
                'unit': unit,
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': 0.0
            }

        metric = self.metrics[name]
        metric['values'].append(value)
        metric['count'] += 1
        metric['sum'] += value
        metric['min'] = min(metric['min'], value)
        metric['max'] = max(metric['max'], value)

    def get_stats(self, name: str) -> Dict[str, Any]:
        """获取统计信息"""
        if name not in self.metrics:
            return {}

        metric = self.metrics[name]
        values = metric['values']

        # 计算百分位数
        sorted_values = sorted(values)
        n = len(sorted_values)
        p50 = sorted_values[int(n * 0.5)] if n > 0 else 0
        p90 = sorted_values[int(n * 0.9)] if n > 0 else 0
        p99 = sorted_values[int(n * 0.99)] if n > 0 else 0

        return {
            'count': metric['count'],
            'mean': metric['sum'] / metric['count'] if metric['count'] > 0 else 0,
            'min': metric['min'],
            'max': metric['max'],
            'p50': p50,
            'p90': p90,
            'p99': p99,
            'unit': metric['unit']
        }

    def report(self) -> str:
        """生成报告"""
        lines = ["=== Performance Report ==="]

        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            lines.append(
                f"{name}: "
                f"mean={stats['mean']:.1f}{stats['unit']} "
                f"p50={stats['p50']:.1f}{stats['unit']} "
                f"p90={stats['p90']:.1f}{stats['unit']} "
                f"(n={stats['count']})"
            )

        return "\n".join(lines)


# 全局性能监控器
performance_monitor = PerformanceMonitor()


# ========== 5. 使用示例 ==========

if __name__ == "__main__":
    # 示例1：追踪节点
    @trace_node("test_node")
    def test_function():
        time.sleep(0.1)
        return "success"

    result = test_function()
    print(f"Result: {result}")

    # 示例2：追踪LLM调用
    @trace_llm_call(model="gpt-4")
    def test_llm():
        time.sleep(0.2)
        return {"response": "test"}

    llm_result = test_llm()

    # 示例3：性能监控
    for i in range(10):
        performance_monitor.record("retrieval", 50 + i * 5, "ms")

    print(performance_monitor.report())
