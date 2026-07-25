"""
可观测性模块 - LangFuse集成

功能：
1. 分布式追踪：追踪每个节点的执行时间
2. LLM调用监控：tokens、cost、latency
3. 性能分析：识别瓶颈
4. 错误追踪：堆栈信息

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
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

# 可选依赖：LangFuse
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    langfuse_context = None

log = logging.getLogger(__name__)


# ========== 1. Tracer基类 ==========

class Tracer:
    """追踪器基类（支持多种后端）"""

    def __init__(self, backend: str = "console"):
        """
        Args:
            backend: console / langfuse / langsmith
        """
        self.backend = backend
        self.enabled = True

        if backend == "langfuse" and LANGFUSE_AVAILABLE:
            self._init_langfuse()
        elif backend == "langfuse":
            log.warning("LangFuse not available, falling back to console")
            self.backend = "console"

    def _init_langfuse(self):
        """初始化LangFuse"""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            log.warning("LangFuse keys not set, falling back to console")
            self.backend = "console"
            return

        try:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            log.info("LangFuse initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize LangFuse: {e}")
            self.backend = "console"

    def trace_node(self, name: str):
        """装饰器：追踪节点执行"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                start_time = time.time()
                error = None
                result = None

                try:
                    if self.backend == "langfuse" and LANGFUSE_AVAILABLE:
                        # 使用LangFuse的observe装饰器
                        observed_func = observe(name=name)(func)
                        result = observed_func(*args, **kwargs)
                    else:
                        # Console模式
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

    def _log_trace(self, name: str, elapsed: float, error: Optional[Exception]):
        """记录追踪日志"""
        status = "ERROR" if error else "OK"
        log.info(f"[TRACE] {name}: {elapsed*1000:.1f}ms ({status})")

        if error:
            log.error(f"[TRACE] {name} failed: {error}")

    @contextmanager
    def span(self, name: str, metadata: Dict = None):
        """上下文管理器：追踪代码块"""
        start_time = time.time()
        error = None

        try:
            if self.backend == "langfuse" and LANGFUSE_AVAILABLE:
                with langfuse_context.observe(name=name) as span:
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

def trace_node(name: str = None):
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


def trace_llm_call(model: str = "unknown"):
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
        def wrapper(*args, **kwargs):
            with tracer.span(f"llm_call_{model}") as span:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time

                    # 尝试提取tokens信息
                    tokens = None
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

                    log.info(f"[LLM] {model}: {elapsed*1000:.1f}ms, tokens={tokens}")

                    return result

                except Exception as e:
                    log.error(f"[LLM] {model} failed: {e}")
                    raise

        return wrapper
    return decorator


# ========== 4. 性能监控 ==========

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {}

    def record(self, name: str, value: float, unit: str = "ms"):
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

    def get_stats(self, name: str) -> Dict:
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
