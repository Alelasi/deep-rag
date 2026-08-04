"""运行时故障转移 LLM 门面（v2.9.x）

背景：Cerebras 通过 Cloudflare 按国家/地区封锁（HTTP 1009 country_banned / 403），
中国大陆出口直连必然在「调用时」失败；而 ChatOpenAI 等客户端在构建阶段不发
起任何网络请求，因此构建期降级链（如旧版 get_llm_with_fallback）实际无效。

本模块提供 RuntimeFailoverLLM：首次 invoke/stream 抛错时，按给定链自动切换
到下一后端并重放本次调用，之后保持用新后端，避免反复支付超时/封锁开销。

设计要点：
- 仅同步接口（本项目无 ainvoke/astream/agenerate 调用，见全仓 grep）
- 兼容 invoke / stream / bind / bind_tools / with_structured_output
- __getattr__ 透传当前后端的属性（_llm_type / model_name 等）
- 线程安全：索引切换用 Lock 保护
- 错误判定集中 in _is_failover_error()，命中才降级，其余异常照常上抛
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 触发故障转移的错误特征（小写匹配）
_FAILOVER_MARKERS = (
    "1009",
    "country_banned",
    "country banned",
    "forbidden",
    "403",
    "unauthorized",
    "401",
    "invalid_api_key",
    "proxy",
    "tunnel",
    "connect error",
    "connection refused",
    "connection reset",
    "connection error",
    "timed out",
    "timeout",
    "getaddrinfo",
    "dns",
    "ssl",
    "tls",
    "502",
    "503",
    "504",
    "522",
    "524",
    "525",
    "526",
)

# 明确不触发故障转移的异常类型（构造失败 / 本地规则等）
_NON_FAILOVER_TYPES: tuple = ()


def _is_failover_error(exc: BaseException) -> bool:
    """判断异常是否属于可用网络/地区/鉴权类，值得切换到下一后端。"""
    if _NON_FAILOVER_TYPES and isinstance(exc, _NON_FAILOVER_TYPES):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _FAILOVER_MARKERS)


class _BoundFailover:
    """bind/bind_tools/with_structured_output 结果：绑定参数后再走运行时降级。"""

    def __init__(
        self,
        owner: "RuntimeFailoverLLM",
        tools: Optional[List[Any]] = None,
        structured: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._owner = owner
        self._tools = tools
        self._structured = structured
        self._kwargs = kwargs or {}

    def _bind_current(self, model: Any) -> Any:
        """对当前后端实例应用绑定参数（bind / bind_tools / with_structured_output）。"""
        if self._structured is not None:
            return model.with_structured_output(self._structured, **self._kwargs)
        if self._tools is not None:
            return model.bind_tools(self._tools, **self._kwargs)
        return model.bind(**self._kwargs)

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        last_exc: Optional[BaseException] = None
        for _ in range(self._owner._remaining()):
            model = self._owner._current()
            try:
                return self._bind_current(model).invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._owner._failover(exc):
                    raise
        raise last_exc  # type: ignore[misc]

    def stream(self, messages: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None
        for _ in range(self._owner._remaining()):
            model = self._owner._current()
            yielded = False
            try:
                for chunk in self._bind_current(model).stream(messages, **kwargs):
                    yielded = True
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if yielded or not self._owner._failover(exc):
                    raise
        raise last_exc  # type: ignore[misc]


class RuntimeFailoverLLM:
    """运行时故障转移 LLM 门面：主后端优先，失败自动降级。"""

    def __init__(self, chain: List[Tuple[str, Callable[[], Any]]]) -> None:
        # 链不允许为空
        if not chain:
            raise ValueError("RuntimeFailoverLLM chain 不能为空")
        self._chain: List[Tuple[str, Callable[[], Any]]] = chain
        self._idx: int = 0
        self._instances: List[Any] = [None for _ in chain]  # type: ignore[list-item]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 内部

    def _remaining(self) -> int:
        """当前索引之后的候选数。"""
        with self._lock:
            return len(self._chain) - self._idx

    def _current(self) -> Any:
        """获取当前后端模型实例（惰性构建并缓存）。"""
        with self._lock:
            idx = self._idx
        if self._instances[idx] is None:
            self._instances[idx] = self._chain[idx][1]()
        return self._instances[idx]

    def _failover(self, exc: BaseException) -> bool:
        """错误可降级且还有候选时切换到下一后端；否则返回 False。"""
        if not _is_failover_error(exc):
            return False
        with self._lock:
            if self._idx + 1 >= len(self._chain):
                return False
            self._idx += 1
            name = self._chain[self._idx][0]
        logger.warning(
            f"[Failover] {self._chain[self._idx - 1][0]} 失败: {str(exc)[:120]}，"
            f"已降级到 → {name}"
        )
        return True

    @property
    def current_provider(self) -> str:
        """当前生效后端名（日志/调试用）。"""
        with self._lock:
            return self._chain[self._idx][0]

    # ------------------------------------------------------------------ 接口

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        last_exc: Optional[BaseException] = None
        for _ in range(self._remaining()):
            model = self._current()
            try:
                return model.invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._failover(exc):
                    raise
        raise last_exc  # type: ignore[misc]

    def stream(self, messages: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None
        for _ in range(self._remaining()):
            model = self._current()
            yielded = False
            try:
                for chunk in model.stream(messages, **kwargs):
                    yielded = True
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if yielded or not self._failover(exc):
                    raise
        raise last_exc  # type: ignore[misc]

    def bind(self, **kwargs: Any) -> _BoundFailover:
        """绑定模型参数（langchain 惯例）。"""
        return _BoundFailover(self, kwargs=kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> _BoundFailover:
        """绑定工具列表（Function Calling 场景）。"""
        return _BoundFailover(self, tools=tools, kwargs=kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _BoundFailover:
        """结构化输出（Pydantic schema）。"""
        return _BoundFailover(self, structured=schema, kwargs=kwargs)

    def __getattr__(self, name: str) -> Any:
        """属性透传：把 _llm_type / model_name / model 等委托给当前后端。"""
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._current(), name)


def build_failover_llm(
    chain: List[Tuple[str, Callable[[], Any]]]
) -> RuntimeFailoverLLM:
    """构造运行时故障转移 LLM（保留独立性，方便单测）。"""
    return RuntimeFailoverLLM(chain)