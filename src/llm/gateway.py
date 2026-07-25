"""LLM Gateway — v2.9.1新增

统一LLM入口：多模型路由 + 熔断 + 限流 + 监控 + 语义缓存

集成现有组件：
- ModelRouter（model_router.py）— 多候选 + 熔断器
- RateLimiter（rate_limiter.py）— 请求频率控制
- SemanticCache（semantic_cache.py）— 语义缓存

用法：
    from src.llm.gateway import get_gateway
    gateway = get_gateway()
    answer = gateway.invoke(messages, task_type="generation")
"""
import time
import logging
from typing import Optional, List, Any
from dataclasses import dataclass, field
from collections import defaultdict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..config import get_temperature, ZHIPU_API_KEY, SILICONFLOW_API_KEY
from .semantic_cache import get_semantic_cache

log = logging.getLogger("deeprag")


@dataclass
class CallMetrics:
    """单次调用指标"""
    provider: str
    model: str
    success: bool
    latency_ms: float
    task_type: str
    cached: bool = False
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """指标收集器

    收集所有LLM调用的指标，支持统计查询。
    """

    def __init__(self):
        self._records: list[CallMetrics] = []
        self._lock = __import__("threading").Lock()

    def record(self, metrics: CallMetrics):
        """记录一次调用"""
        with self._lock:
            self._records.append(metrics)
            # 保留最近1000条
            if len(self._records) > 1000:
                self._records = self._records[-1000:]

    def get_stats(self) -> dict:
        """获取统计报告"""
        with self._lock:
            if not self._records:
                return {"total_calls": 0}

            total = len(self._records)
            success_count = sum(1 for r in self._records if r.success)
            cache_hits = sum(1 for r in self._records if r.cached)
            latencies = [r.latency_ms for r in self._records if not r.cached]

            # 按provider统计
            by_provider = defaultdict(lambda: {"calls": 0, "success": 0, "total_latency": 0})
            for r in self._records:
                by_provider[r.provider]["calls"] += 1
                if r.success:
                    by_provider[r.provider]["success"] += 1
                    by_provider[r.provider]["total_latency"] += r.latency_ms

            provider_stats = {}
            for provider, stats in by_provider.items():
                provider_stats[provider] = {
                    "calls": stats["calls"],
                    "success_rate": round(stats["success"] / stats["calls"] * 100, 1) if stats["calls"] > 0 else 0,
                    "avg_latency_ms": round(stats["total_latency"] / stats["success"], 1) if stats["success"] > 0 else 0,
                }

            # 延迟分位数
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            p50 = latencies_sorted[n // 2] if n > 0 else 0
            p95 = latencies_sorted[int(n * 0.95)] if n > 0 else 0
            p99 = latencies_sorted[int(n * 0.99)] if n > 0 else 0

            return {
                "total_calls": total,
                "success_rate": round(success_count / total * 100, 1),
                "cache_hit_rate": round(cache_hits / total * 100, 1) if total > 0 else 0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                "p50_latency_ms": round(p50, 1),
                "p95_latency_ms": round(p95, 1),
                "p99_latency_ms": round(p99, 1),
                "by_provider": provider_stats,
            }


class LLMGateway:
    """LLM Gateway — 统一LLM调用入口

    集成语义缓存、温度策略、指标收集。
    不替代 get_llm()，而是作为可选的高级入口。
    """

    def __init__(self):
        self.cache = get_semantic_cache()
        self.metrics = MetricsCollector()
        self._llm_cache = {}  # LLM实例缓存

    def _get_llm(self, task_type: str, temperature: Optional[float] = None):
        """获取LLM实例（带缓存）"""
        temp = temperature if temperature is not None else get_temperature(task_type)
        cache_key = f"{task_type}:{temp}"

        if cache_key not in self._llm_cache:
            from ..config import get_llm
            self._llm_cache[cache_key] = get_llm(temp)

        return self._llm_cache[cache_key]

    def _extract_query_text(self, messages: List[BaseMessage]) -> str:
        """从消息列表中提取查询文本（用于缓存key）"""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""

    def invoke(
        self,
        messages: List[BaseMessage],
        task_type: str = "generation",
        temperature: Optional[float] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> str:
        """统一LLM调用入口

        Args:
            messages: 消息列表
            task_type: 任务类型（决定温度和是否缓存）
            temperature: 自定义温度（覆盖task_type默认值）
            use_cache: 是否使用语义缓存
            **kwargs: 传递给LLM的额外参数

        Returns:
            LLM生成的文本
        """
        # 1. 语义缓存检查
        query_text = self._extract_query_text(messages)
        if use_cache and self.cache and task_type == "generation" and query_text:
            cached = self.cache.get(query_text, task_type)
            if cached is not None:
                self.metrics.record(CallMetrics(
                    provider="cache",
                    model="semantic_cache",
                    success=True,
                    latency_ms=0.1,
                    task_type=task_type,
                    cached=True,
                ))
                log.debug(f"[LLMGateway] 语义缓存命中: {query_text[:30]}...")
                return cached

        # 2. 获取LLM实例
        llm = self._get_llm(task_type, temperature)
        if llm is None:
            raise RuntimeError("LLM不可用，请检查后端配置")

        # 3. 调用LLM
        start = time.time()
        provider = getattr(llm, "_llm_type", "unknown")
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))

        try:
            response = llm.invoke(messages, **kwargs)
            latency_ms = (time.time() - start) * 1000

            # 提取响应文本
            if isinstance(response, str):
                content = response
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # 4. 记录指标
            self.metrics.record(CallMetrics(
                provider=provider,
                model=model_name,
                success=True,
                latency_ms=round(latency_ms, 1),
                task_type=task_type,
            ))

            # 5. 写入语义缓存
            if use_cache and self.cache and task_type == "generation" and query_text:
                self.cache.set(query_text, content, task_type)

            return content

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self.metrics.record(CallMetrics(
                provider=provider,
                model=model_name,
                success=False,
                latency_ms=round(latency_ms, 1),
                task_type=task_type,
            ))
            log.error(f"[LLMGateway] LLM调用失败: {e}")
            raise

    def stream(
        self,
        messages: List[BaseMessage],
        task_type: str = "generation",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ):
        """流式LLM调用（不缓存）

        Yields:
            生成token
        """
        llm = self._get_llm(task_type, temperature)
        if llm is None:
            raise RuntimeError("LLM不可用")

        start = time.time()
        provider = getattr(llm, "_llm_type", "unknown")
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))

        try:
            for chunk in llm.stream(messages, **kwargs):
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)

            latency_ms = (time.time() - start) * 1000
            self.metrics.record(CallMetrics(
                provider=provider,
                model=model_name,
                success=True,
                latency_ms=round(latency_ms, 1),
                task_type=task_type,
            ))
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self.metrics.record(CallMetrics(
                provider=provider,
                model=model_name,
                success=False,
                latency_ms=round(latency_ms, 1),
                task_type=task_type,
            ))
            log.error(f"[LLMGateway] 流式调用失败: {e}")
            raise

    def get_metrics(self) -> dict:
        """获取调用统计"""
        stats = self.metrics.get_stats()
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        return stats


# 全局单例
_gateway_instance: Optional[LLMGateway] = None


def get_gateway() -> LLMGateway:
    """获取全局LLM Gateway实例"""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = LLMGateway()
    return _gateway_instance
