"""可靠性工具：降级与熔断（轻量、可单测，不伪装成完整 SRE 平台）。"""

from .degrade import DegradePolicy, CircuitBreaker, degrade_answer

__all__ = ["DegradePolicy", "CircuitBreaker", "degrade_answer"]
