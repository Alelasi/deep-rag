"""
成本控制模块

功能：
1. Token计数（精确统计）
2. 成本追踪（实时计算）
3. 预算管理（超预算告警）
4. 成本优化建议

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
from typing import Dict, Optional
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
    """成本追踪器"""

    def __init__(self):
        self.usage_records = []
        self.total_cost = 0.0
        self.budget_limit = None
        self.budget_warning_threshold = 0.8  # 80%预算时告警

        # 初始化tiktoken
        if TIKTOKEN_AVAILABLE:
            self.encodings = {}
        else:
            log.warning("tiktoken not installed, using estimation (1.3 char/token for Chinese)")

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        精确计数tokens

        Args:
            text: 文本
            model: 模型名（用于选择正确的tokenizer）

        Returns:
            token数量
        """
        if TIKTOKEN_AVAILABLE:
            # 使用tiktoken精确计数
            if model not in self.encodings:
                try:
                    self.encodings[model] = tiktoken.encoding_for_model(model)
                except:
                    # 回退到默认编码
                    self.encodings[model] = tiktoken.get_encoding("cl100k_base")

            return len(self.encodings[model].encode(text))
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
        pricing = None
        for key, value in PRICING.items():
            if key in model.lower() or model.lower() in key:
                pricing = value
                break

        if not pricing:
            log.warning(f"Unknown model pricing: {model}, assuming free")
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

        self.usage_records.append(usage)
        self.total_cost += cost

        # 检查预算
        self._check_budget()

        log.debug(
            f"[Cost] {model}: {input_tokens}+{output_tokens}={total_tokens} tokens, "
            f"${cost:.6f} (total: ${self.total_cost:.6f})"
        )

        return usage

    def set_budget(self, budget_usd: float, warning_threshold: float = 0.8):
        """
        设置预算

        Args:
            budget_usd: 预算（美元）
            warning_threshold: 告警阈值（0-1）
        """
        self.budget_limit = budget_usd
        self.budget_warning_threshold = warning_threshold
        log.info(f"Budget set: ${budget_usd:.2f} (warning at {warning_threshold*100:.0f}%)")

    def _check_budget(self):
        """检查预算（内部方法）"""
        if self.budget_limit is None:
            return

        usage_rate = self.total_cost / self.budget_limit

        if usage_rate >= 1.0:
            log.error(
                f"⛔ Budget exceeded! ${self.total_cost:.2f} / ${self.budget_limit:.2f} "
                f"({usage_rate*100:.1f}%)"
            )
        elif usage_rate >= self.budget_warning_threshold:
            log.warning(
                f"⚠️ Budget warning! ${self.total_cost:.2f} / ${self.budget_limit:.2f} "
                f"({usage_rate*100:.1f}%)"
            )

    def get_summary(self) -> Dict:
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
        if not self.usage_records:
            return {
                'total_cost': 0.0,
                'total_tokens': 0,
                'by_model': {},
                'budget': None
            }

        # 按模型统计
        by_model = {}
        for record in self.usage_records:
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
        budget_info = None
        if self.budget_limit:
            budget_info = {
                'limit_usd': self.budget_limit,
                'used_usd': self.total_cost,
                'remaining_usd': self.budget_limit - self.total_cost,
                'usage_rate': self.total_cost / self.budget_limit
            }

        return {
            'total_cost': self.total_cost,
            'total_tokens': sum(r.total_tokens for r in self.usage_records),
            'total_calls': len(self.usage_records),
            'by_model': by_model,
            'budget': budget_info
        }

    def print_summary(self):
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

def track_cost(model: str = "gpt-4"):
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
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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
