"""生产向轻量降级 / 熔断

面试可讲：
- 主链路 LLM 429 / 超时 → 降级到更小模型或拒答模板
- 熔断：连续失败 N 次，冷却 T 秒不再打主后端
- 与 get_llm_with_fallback 互补：本模块管「何时放弃主路径」
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DegradePolicy:
    """降级策略参数。"""

    max_failures: int = 3
    open_seconds: float = 30.0
    # 主路径失败时是否允许返回固定拒答（无 LLM）
    allow_static_refuse: bool = True


@dataclass
class CircuitBreaker:
    """简单熔断器：closed → open → half_open（单进程内存）。"""

    policy: DegradePolicy = field(default_factory=DegradePolicy)
    failures: int = 0
    opened_at: float = 0.0
    state: str = "closed"  # closed | open | half_open

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.opened_at >= self.policy.open_seconds:
                self.state = "half_open"
                return True
            return False
        return True  # half_open 试探

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"
        self.opened_at = 0.0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.policy.max_failures:
            self.state = "open"
            self.opened_at = time.time()


def degrade_answer(reason: str = "upstream_unavailable") -> dict:
    """主链路不可用时的标准降级响应（可被 API/UI 直接返回）。"""
    return {
        "answer": (
            "【直接回答】上游模型暂时不可用，已进入降级模式，无法完成本次生成。\n\n"
            f"【详细解释】原因：{reason}。请稍后重试，或切换备用 LLM 后端"
            "（Groq / Silicon / Zhipu）。\n\n"
            "【引用来源】（无）"
        ),
        "no_knowledge": False,
        "degraded": True,
        "degrade_reason": reason,
        "citations": [],
        "hallucination_score": 0.0,
        "fact_check_passed": True,
    }


# 进程级默认熔断器（API 可注入自定义实例）
_default_breaker = CircuitBreaker()


def get_default_breaker() -> CircuitBreaker:
    return _default_breaker
