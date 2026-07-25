"""
模型路由器 + 简单熔断器
支持多候选模型 + 故障转移 + 连续失败熔断

设计思路（参考 Ragent）：
1. 配置多个候选模型（主力 + 备用）
2. 按优先级尝试，失败自动切换
3. 连续失败 N 次后熔断，一段时间后重试
"""

import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常，允许请求
    OPEN = "open"          # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 半开，试探性允许请求


@dataclass
class ModelCandidate:
    """模型候选配置"""
    id: str                    # 模型 ID（如 "qwen3-max"）
    provider: str              # 供应商（如 "bailian"）
    api_key: str              # API Key
    endpoint: Optional[str] = None  # 自定义 endpoint
    priority: int = 1          # 优先级（数字越小越优先）
    enabled: bool = True       # 是否启用

    def __repr__(self):
        return f"ModelCandidate(id={self.id}, provider={self.provider}, priority={self.priority})"


class CircuitBreaker:
    """简单熔断器（三态）"""

    def __init__(
        self,
        failure_threshold: int = 2,     # 连续失败多少次触发熔断
        open_duration_sec: int = 30,    # 熔断持续时间（秒）
    ):
        self.failure_threshold = failure_threshold
        self.open_duration_sec = open_duration_sec

        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.open_until = 0  # 熔断截止时间戳
        self.last_failure_time = 0

    def allow_call(self) -> bool:
        """是否允许调用"""
        now = time.time()

        if self.state == CircuitState.CLOSED:
            return True

        elif self.state == CircuitState.OPEN:
            # 检查冷却时间是否到了
            if now >= self.open_until:
                # 转为半开状态，试探性允许请求
                self.state = CircuitState.HALF_OPEN
                return True
            return False  # 还在熔断中

        elif self.state == CircuitState.HALF_OPEN:
            # 半开状态允许请求（试探）
            return True

        return False

    def record_success(self):
        """记录成功调用"""
        if self.state == CircuitState.HALF_OPEN:
            # 半开状态下成功 → 恢复到关闭状态
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            print(f"  ✅ 熔断器恢复：HALF_OPEN → CLOSED")

        # 关闭状态下成功 → 重置计数器
        self.consecutive_failures = 0

    def record_failure(self):
        """记录失败调用"""
        now = time.time()
        self.last_failure_time = now
        self.consecutive_failures += 1

        if self.state == CircuitState.HALF_OPEN:
            # 半开状态下失败 → 重新熔断
            self.state = CircuitState.OPEN
            self.open_until = now + self.open_duration_sec
            print(f"  ⚠️ 熔断器重新打开：HALF_OPEN → OPEN（冷却 {self.open_duration_sec}s）")

        elif self.state == CircuitState.CLOSED:
            # 关闭状态下连续失败达到阈值 → 熔断
            if self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.open_until = now + self.open_duration_sec
                print(f"  🔥 熔断器打开：连续失败 {self.consecutive_failures} 次 → OPEN（冷却 {self.open_duration_sec}s）")

    def get_status(self) -> Dict:
        """获取熔断器状态"""
        now = time.time()
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "time_until_retry": max(0, int(self.open_until - now)) if self.state == CircuitState.OPEN else 0
        }


class ModelRouter:
    """模型路由器（支持多候选 + 熔断器）"""

    def __init__(self, candidates: List[ModelCandidate]):
        # 按优先级排序
        self.candidates = sorted(candidates, key=lambda c: (not c.enabled, c.priority, c.id))

        # 为每个候选创建熔断器
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            c.id: CircuitBreaker() for c in self.candidates
        }

        print(f"✅ 模型路由器初始化完成：{len(self.candidates)} 个候选")
        for i, c in enumerate(self.candidates, 1):
            print(f"  {i}. {c.id} ({c.provider}) - 优先级 {c.priority}")

    def call_with_fallback(
        self,
        caller: Callable[[ModelCandidate], str],  # 调用函数（传入候选，返回结果）
        max_retries: int = 3  # 最多尝试多少个候选
    ) -> str:
        """
        带故障转移的模型调用

        Args:
            caller: 调用函数，接收 ModelCandidate，返回结果字符串
            max_retries: 最多尝试多少个候选

        Returns:
            str: 模型返回的结果

        Raises:
            RuntimeError: 所有候选都失败
        """
        attempted = []

        for i, candidate in enumerate(self.candidates[:max_retries]):
            if not candidate.enabled:
                continue

            breaker = self.circuit_breakers[candidate.id]

            # 检查熔断器是否允许调用
            if not breaker.allow_call():
                status = breaker.get_status()
                print(f"  ⏭️ 跳过 {candidate.id}：熔断中（{status['time_until_retry']}s 后重试）")
                attempted.append((candidate.id, "circuit_open"))
                continue

            try:
                print(f"\n🔄 尝试候选 {i+1}/{min(max_retries, len(self.candidates))}: {candidate.id}")

                # 调用模型
                result = caller(candidate)

                # 成功：记录到熔断器
                breaker.record_success()
                print(f"  ✅ 成功：{candidate.id}")

                return result

            except Exception as e:
                # 失败：记录到熔断器
                breaker.record_failure()
                print(f"  ❌ 失败：{candidate.id} - {str(e)[:50]}")
                attempted.append((candidate.id, str(e)[:50]))

        # 所有候选都失败
        raise RuntimeError(
            f"所有候选模型都失败了（尝试了 {len(attempted)} 个）：\n" +
            "\n".join(f"  - {cid}: {err}" for cid, err in attempted)
        )

    def get_status(self) -> Dict:
        """获取所有候选的状态"""
        return {
            c.id: self.circuit_breakers[c.id].get_status()
            for c in self.candidates
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置候选模型
    candidates = [
        ModelCandidate(
            id="qwen3-max",
            provider="bailian",
            api_key="sk-xxx",
            priority=1,
            enabled=True
        ),
        ModelCandidate(
            id="glm-4.7",
            provider="siliconflow",
            api_key="sk-yyy",
            priority=2,
            enabled=True
        ),
        ModelCandidate(
            id="qwen-plus",
            provider="bailian",
            api_key="sk-zzz",
            priority=3,
            enabled=True
        ),
    ]

    router = ModelRouter(candidates)

    # 定义调用函数（模拟）
    call_count = {"qwen3-max": 0, "glm-4.7": 0, "qwen-plus": 0}

    def mock_call_llm(candidate: ModelCandidate) -> str:
        """模拟 LLM 调用（前 2 次 qwen3-max 失败，后面成功）"""
        call_count[candidate.id] += 1

        if candidate.id == "qwen3-max" and call_count[candidate.id] <= 2:
            raise RuntimeError("Connection timeout")  # 模拟失败

        return f"Response from {candidate.id}"

    # 测试故障转移
    print("\n" + "="*60)
    print("测试 1：第一次调用（qwen3-max 失败 → 切换到 glm-4.7）")
    print("="*60)
    try:
        result = router.call_with_fallback(mock_call_llm)
        print(f"\n最终结果：{result}")
    except Exception as e:
        print(f"\n调用失败：{e}")

    print("\n" + "="*60)
    print("测试 2：第二次调用（qwen3-max 失败第 2 次 → 熔断 → 直接走 glm-4.7）")
    print("="*60)
    try:
        result = router.call_with_fallback(mock_call_llm)
        print(f"\n最终结果：{result}")
    except Exception as e:
        print(f"\n调用失败：{e}")

    print("\n" + "="*60)
    print("测试 3：第三次调用（qwen3-max 已熔断 → 跳过 → 直接走 glm-4.7）")
    print("="*60)
    try:
        result = router.call_with_fallback(mock_call_llm)
        print(f"\n最终结果：{result}")
    except Exception as e:
        print(f"\n调用失败：{e}")

    # 查看熔断器状态
    print("\n" + "="*60)
    print("熔断器状态")
    print("="*60)
    for model_id, status in router.get_status().items():
        print(f"{model_id}: {status}")
