"""
成本控制模块

功能：
1. Token计数（精确统计）
2. 成本追踪（实时计算）
3. 预算管理（超预算告警）
4. 成本优化建议

设计要点（生产化）：
- 默认结构化 console 输出（JSON 行），便于日志采集。
- 线程安全：token / cost 累计使用 threading.Lock 保护。
- tiktoken 为可选依赖，缺失时静默降级为估算（不崩溃）。

使用方式：
```python
from src.observability.cost_tracker import cost_tracker, track_cost

@track_cost(model="gpt-4")
def call_llm(prompt):
    return llm.invoke(prompt)

# 查看统计
print(cost_tracker.get_summary())
```
"""
import logging
import time
import json
import threading
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
import functools

# 可选依赖：tiktoken（精确token计数）
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

log = logging.getLogger(__name__)


# ========== 结构化日志辅助 ==========

def _emit(level: int, event: str, **fields: Any) -> None:
    """以 JSON 行输出结构化日志，便于集中采集。"""
    payload: Dict[str, Any] = {"event": event}
    payload.update(fields)
    log.log(level, json.dumps(payload, ensure_ascii=False, default=str))


# ========== 1. 定价表 ==========

PRICING = {
    # OpenAI
    "gpt-4": {"input": 0.03/1000, "output": 0.06/1000},
    "gpt-4-turbo": {"input": 0.01/1000, "output": 0.03/1000},
    "gpt-4o": {"input": 0.005/1000, "output": 0.015/1000},
    "gpt-4o-mini": {"input": 0.00015/1000, "output": 0.0006/1000},
    "gpt-3.5-turbo": {"input": 0.0015/1000, "output": 0.002/1000},

    # Anthropic
    "claude-opus-4": {"input": 0.015/1000, "output": 0.075/1000},
    "claude-sonnet-4": {"input": 0.003/1000, "output": 0.015/1000},
    "claude-haiku-4": {"input": 0.0008/1000, "output": 0.004/1000},

    # 本地模型（免费）
    "ollama": {"input": 0.0, "output": 0.0},
    "lmstudio": {"input": 0.0, "output": 0.0},
    "local": {"input": 0.0, "output": 0.0},
}


@dataclass
class TokenUsage:
    """Token使用记录"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    cost_usd: float
    timestamp: float


class CostTracker:
    """成本追踪器（线程安全）"""

    def __init__(self):
        self.usage_records: list[TokenUsage] = []
        self.total_cost: float = 0.0
        self.budget_limit: Optional[float] = None
        self.budget_warning_threshold: float = 0.8  # 80%预算时告警

        # 线程安全锁：保护累计状态
        self._lock: threading.Lock = threading.Lock()
        # tiktoken 编码缓存（构建时也加锁保护）
        self.encodings: Dict[str, Any] = {}
        # tiktoken 运行时不可用（如下载依赖缺失）时整体降级为估算
        self._tiktoken_disabled: bool = False

        if not TIKTOKEN_AVAILABLE:
            log.warning("tiktoken 未安装，使用估算（约 1.3 token/字符，对中文近似）")

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        精确计数tokens

        Args:
            text: 文本
            model: 模型名（用于选择正确的tokenizer）

        Returns:
            token数量
        """
        if TIKTOKEN_AVAILABLE and not self._tiktoken_disabled:
            # 编码对象构建需加锁（延迟初始化，非线程安全）
            with self._lock:
                if model not in self.encodings:
                    try:
                        self.encodings[model] = tiktoken.encoding_for_model(model)
                    except Exception:
                        # 下载/依赖不可用（如缺 requests），标记该模型编码不可用
                        self.encodings[model] = None
                        self._tiktoken_disabled = True

                encoding = self.encodings[model]

            if encoding is not None:
                return len(encoding.encode(text))
            else:
                # 本模型无法使用 tiktoken，降级为估算（不崩溃）
                log.warning(f"tiktoken 编码加载失败，模型 {model} 使用估算")
                return int(len(text) * 1.3)
        else:
            # 估算（中文约1.3 token/char，英文约0.25 token/word）
            # 简化：统一按1.3计算
            return int(len(text) * 1.3)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """
        计算成本

        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数
            model: 模型名

        Returns:
            成本（美元）
        """
        # 查找定价（支持模糊匹配）
        pricing: Optional[Dict[str, float]] = None
        for key, value in PRICING.items():
            if key in model.lower() or model.lower() in key:
                pricing = value
                break

        if not pricing:
            log.warning(f"未知模型定价: {model}，按免费处理")
            pricing = {"input": 0.0, "output": 0.0}

        cost = (
            input_tokens * pricing["input"] +
            output_tokens * pricing["output"]
        )

        return cost

    def track(
        self,
        input_text: str,
        output_text: str,
        model: str
    ) -> TokenUsage:
        """
        追踪一次LLM调用

        Args:
            input_text: 输入文本
            output_text: 输出文本
            model: 模型名

        Returns:
            TokenUsage记录
        """
        # 计数tokens
        input_tokens = self.count_tokens(input_text, model)
        output_tokens = self.count_tokens(output_text, model)
        total_tokens = input_tokens + output_tokens

        # 计算成本
        cost = self.calculate_cost(input_tokens, output_tokens, model)

        # 记录
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
            cost_usd=cost,
            timestamp=time.time()
        )

        # 加锁保护累计状态与预算检查
        with self._lock:
            self.usage_records.append(usage)
            self.total_cost += cost

            _emit(
                logging.DEBUG,
                "cost_track",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                total_cost_usd=self.total_cost,
            )

            # 检查预算（锁内读取，避免重复加锁）
            self._check_budget_locked()

        return usage

    def set_budget(self, budget_usd: float, warning_threshold: float = 0.8) -> None:
        """
        设置预算

        Args:
            budget_usd: 预算（美元）
            warning_threshold: 告警阈值（0-1）
        """
        with self._lock:
            self.budget_limit = budget_usd
            self.budget_warning_threshold = warning_threshold
        log.info(f"预算已设置: ${budget_usd:.2f} (告警阈值 {warning_threshold*100:.0f}%)")

    def _check_budget_locked(self) -> None:
        """检查预算（调用方须已持有 self._lock）"""
        if self.budget_limit is None:
            return

        usage_rate = self.total_cost / self.budget_limit

        if usage_rate >= 1.0:
            _emit(
                logging.ERROR,
                "budget_exceeded",
                total_cost_usd=self.total_cost,
                budget_usd=self.budget_limit,
                usage_rate=usage_rate,
            )
        elif usage_rate >= self.budget_warning_threshold:
            _emit(
                logging.WARNING,
                "budget_warning",
                total_cost_usd=self.total_cost,
                budget_usd=self.budget_limit,
                usage_rate=usage_rate,
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要

        Returns:
            {
                'total_cost': 总成本,
                'total_tokens': 总token数,
                'by_model': 按模型统计,
                'budget': 预算信息
            }
        """
        # 加锁复制累计状态，避免读取过程中被并发修改
        with self._lock:
            records = list(self.usage_records)
            total_cost = self.total_cost
            budget_limit = self.budget_limit

        if not records:
            return {
                'total_cost': 0.0,
                'total_tokens': 0,
                'total_calls': 0,
                'by_model': {},
                'budget': None
            }

        # 按模型统计
        by_model: Dict[str, Dict[str, Any]] = {}
        for record in records:
            model = record.model
            if model not in by_model:
                by_model[model] = {
                    'calls': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'cost_usd': 0.0
                }

            by_model[model]['calls'] += 1
            by_model[model]['input_tokens'] += record.input_tokens
            by_model[model]['output_tokens'] += record.output_tokens
            by_model[model]['total_tokens'] += record.total_tokens
            by_model[model]['cost_usd'] += record.cost_usd

        # 预算信息
        budget_info: Optional[Dict[str, Any]] = None
        if budget_limit is not None:
            budget_info = {
                'limit_usd': budget_limit,
                'used_usd': total_cost,
                'remaining_usd': budget_limit - total_cost,
                'usage_rate': total_cost / budget_limit
            }

        return {
            'total_cost': total_cost,
            'total_tokens': sum(r.total_tokens for r in records),
            'total_calls': len(records),
            'by_model': by_model,
            'budget': budget_info
        }

    def print_summary(self) -> None:
        """打印统计摘要"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("成本统计摘要")
        print("="*60)

        print(f"\n总调用次数: {summary['total_calls']}")
        print(f"总Tokens: {summary['total_tokens']:,}")
        print(f"总成本: ${summary['total_cost']:.6f}")

        if summary['budget']:
            budget = summary['budget']
            print(f"\n预算:")
            print(f"  限额: ${budget['limit_usd']:.2f}")
            print(f"  已用: ${budget['used_usd']:.6f} ({budget['usage_rate']*100:.1f}%)")
            print(f"  剩余: ${budget['remaining_usd']:.6f}")

        if summary['by_model']:
            print(f"\n按模型统计:")
            for model, stats in summary['by_model'].items():
                print(f"  {model}:")
                print(f"    调用: {stats['calls']}次")
                print(f"    Tokens: {stats['total_tokens']:,} "
                      f"(输入:{stats['input_tokens']:,}, 输出:{stats['output_tokens']:,})")
                print(f"    成本: ${stats['cost_usd']:.6f}")

        print("="*60)


# ========== 2. 全局实例 ==========

cost_tracker = CostTracker()


# ========== 3. 装饰器 ==========

def track_cost(model: str = "gpt-4") -> Callable:
    """
    装饰器：自动追踪成本

    使用方式：
    ```python
    @track_cost(model="gpt-4")
    def call_llm(prompt):
        response = llm.invoke(prompt)
        return response.content
    ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 执行函数
            result = func(*args, **kwargs)

            # 尝试提取输入输出
            # （需要函数返回包含input/output的对象）
            if hasattr(result, 'content'):
                # LangChain AIMessage
                input_text = args[0] if args else ""
                output_text = result.content
                cost_tracker.track(input_text, output_text, model)

            return result

        return wrapper
    return decorator


# ========== 4. 使用示例 ==========

if __name__ == "__main__":
    # 示例1：手动追踪
    tracker = CostTracker()
    tracker.set_budget(1.0)  # $1预算

    # 模拟10次调用
    for i in range(10):
        tracker.track(
            input_text="如何配置LangChain的API Key？",
            output_text="配置 LangChain 的 API Key 步骤：1. 创建.env文件..." * 20,
            model="gpt-4"
        )

    tracker.print_summary()

    # 示例2：对比不同模型成本
    print("\n对比不同模型:")
    models = ["gpt-4", "gpt-4o-mini", "claude-sonnet-4", "lmstudio"]
    input_text = "什么是RAG？" * 10
    output_text = "RAG是检索增强生成..." * 50

    for model in models:
        cost = tracker.calculate_cost(
            tracker.count_tokens(input_text, model),
            tracker.count_tokens(output_text, model),
            model
        )
        print(f"  {model:20s}: ${cost:.6f}")
