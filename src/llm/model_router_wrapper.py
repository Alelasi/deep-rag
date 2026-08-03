"""
模型路由包装器（LangChain 集成）

把 ModelRouter 包装成 LangChain 兼容的 LLM 对象
支持故障转移 + 熔断器
"""

from typing import Any, List, Optional
from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .model_router import ModelRouter, ModelCandidate
from ..config import (
    MODEL_CANDIDATES,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    ZHIPU_API_KEY,
    SILICONFLOW_API_KEY,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_OPEN_DURATION_SEC,
)


try:
    from src.logging_config import get_logger
except Exception:
    import logging

    def get_logger(n):  # type: ignore
        return logging.getLogger(n)

logger = get_logger(__name__)

class RoutedLLM(BaseChatModel):
    """
    路由包装的 LLM（LangChain 兼容）

    用法：
        llm = RoutedLLM(router=router, temperature=0.3)
        response = llm.invoke("Hello")
    """

    router: ModelRouter = Field(...)  # 必填字段，Pydantic v2 要求先声明
    temperature: float = Field(default=0.3)

    def __init__(self, router: ModelRouter, temperature: float = 0.3, **kwargs):
        # 传入字段值给 Pydantic（关键：必须传 router=router）
        super().__init__(router=router, temperature=temperature, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成响应（带故障转移）"""

        def call_with_candidate(candidate: ModelCandidate) -> str:
            """用指定候选调用 LLM"""
            # 根据 provider 创建 LLM 实例
            if candidate.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model=candidate.id,
                    temperature=self.temperature,
                    api_key=candidate.api_key
                )
            elif candidate.provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=candidate.id,
                    temperature=self.temperature,
                    api_key=candidate.api_key
                )
            elif candidate.provider == "ollama":
                from langchain_ollama import ChatOllama
                llm = ChatOllama(
                    model=candidate.id,
                    temperature=self.temperature,
                    base_url=candidate.endpoint or "http://localhost:11434"
                )
            elif candidate.provider == "zhipu":  # v2.9.1: 智谱AI
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=candidate.id,
                    temperature=self.temperature,
                    api_key=candidate.api_key or ZHIPU_API_KEY,
                    base_url="https://open.bigmodel.cn/api/paas/v4",
                )
            elif candidate.provider == "siliconcloud":  # v2.9.1: 硅基流动
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=candidate.id,
                    temperature=self.temperature,
                    api_key=candidate.api_key or SILICONFLOW_API_KEY,
                    base_url="https://api.siliconflow.cn/v1",
                )
            else:
                raise ValueError(f"Unknown provider: {candidate.provider}")

            # 调用 LLM
            result = llm.invoke(messages, stop=stop, **kwargs)
            return result.content

        # 通过路由器调用（带故障转移）
        content = self.router.call_with_fallback(call_with_candidate)

        # 返回 LangChain 格式
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """流式生成响应（v2.9.1新增）— 使用第一个可用候选"""
        from langchain_core.outputs import ChatGenerationChunk

        for candidate in self.router.candidates:
            if not candidate.enabled:
                continue
            breaker = self.router.circuit_breakers[candidate.id]
            if not breaker.allow_call():
                continue

            try:
                # 创建LLM实例（复用call_with_candidate逻辑）
                if candidate.provider == "zhipu":
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=candidate.id, temperature=self.temperature,
                        api_key=candidate.api_key or ZHIPU_API_KEY,
                        base_url="https://open.bigmodel.cn/api/paas/v4",
                    )
                elif candidate.provider == "siliconcloud":
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=candidate.id, temperature=self.temperature,
                        api_key=candidate.api_key or SILICONFLOW_API_KEY,
                        base_url="https://api.siliconflow.cn/v1",
                    )
                elif candidate.provider == "ollama":
                    from langchain_ollama import ChatOllama
                    llm = ChatOllama(
                        model=candidate.id, temperature=self.temperature,
                        base_url=candidate.endpoint or "http://localhost:11434",
                    )
                else:
                    # anthropic/openai 等走原逻辑
                    if candidate.provider == "anthropic":
                        from langchain_anthropic import ChatAnthropic
                        llm = ChatAnthropic(model=candidate.id, temperature=self.temperature, api_key=candidate.api_key)
                    else:
                        from langchain_openai import ChatOpenAI
                        llm = ChatOpenAI(model=candidate.id, temperature=self.temperature, api_key=candidate.api_key)

                # 流式输出
                for chunk in llm.stream(messages, stop=stop, **kwargs):
                    if hasattr(chunk, "content") and chunk.content:
                        yield ChatGenerationChunk(message=AIMessage(content=chunk.content))

                breaker.record_success()
                return  # 成功则返回

            except Exception as e:
                breaker.record_failure()
                continue

        # 所有候选都失败
        raise RuntimeError("所有候选模型流式调用都失败")

    @property
    def _llm_type(self) -> str:
        return "routed_llm"


def parse_candidates(candidates_str: str) -> List[ModelCandidate]:
    """
    解析候选模型配置字符串

    格式：provider:model_id,provider:model_id,...
    示例：anthropic:claude-sonnet-4,openai:gpt-4o-mini

    优先级按顺序递增（第一个优先级最高）
    """
    candidates = []

    for i, item in enumerate(candidates_str.split(",")):
        item = item.strip()
        if not item:
            continue

        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid candidate format: {item}. Expected: provider:model_id")

        provider, model_id = parts
        provider = provider.strip().lower()
        model_id = model_id.strip()

        # 根据 provider 获取 API Key
        if provider == "anthropic":
            api_key = ANTHROPIC_API_KEY
        elif provider == "openai":
            api_key = OPENAI_API_KEY
        elif provider == "ollama":
            api_key = ""  # Ollama 不需要 API Key
        elif provider == "zhipu":  # v2.9.1: 智谱AI
            api_key = ZHIPU_API_KEY
        elif provider == "siliconcloud":  # v2.9.1: 硅基流动
            api_key = SILICONFLOW_API_KEY
        else:
            raise ValueError(f"Unknown provider: {provider}. Use: anthropic/openai/ollama/zhipu/siliconcloud")

        candidates.append(ModelCandidate(
            id=model_id,
            provider=provider,
            api_key=api_key,
            priority=i + 1,  # 优先级按顺序递增
            enabled=True
        ))

    return candidates


def get_routed_llm(temperature: float = 0.3) -> RoutedLLM:
    """
    创建路由 LLM

    从环境变量读取配置：
    - MODEL_CANDIDATES: 候选模型列表
    - CIRCUIT_BREAKER_FAILURE_THRESHOLD: 熔断阈值
    - CIRCUIT_BREAKER_OPEN_DURATION_SEC: 熔断时长
    """
    # 解析候选配置
    candidates = parse_candidates(MODEL_CANDIDATES)

    if not candidates:
        raise ValueError("MODEL_CANDIDATES is empty. Set it to: provider:model_id,provider:model_id,...")

    # 创建路由器（会自动创建熔断器）
    router = ModelRouter(candidates)

    # 覆盖熔断器配置
    for breaker in router.circuit_breakers.values():
        breaker.failure_threshold = CIRCUIT_BREAKER_FAILURE_THRESHOLD
        breaker.open_duration_sec = CIRCUIT_BREAKER_OPEN_DURATION_SEC

    # 返回包装的 LLM
    return RoutedLLM(router=router, temperature=temperature)


# ==================== 测试 ====================

if __name__ == "__main__":
    import os

    # 设置测试环境变量
    os.environ["MODEL_CANDIDATES"] = "anthropic:claude-sonnet-4,openai:gpt-4o-mini"
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-anthropic"
    os.environ["OPENAI_API_KEY"] = "sk-test-openai"
    os.environ["CIRCUIT_BREAKER_FAILURE_THRESHOLD"] = "2"
    os.environ["CIRCUIT_BREAKER_OPEN_DURATION_SEC"] = "30"

    # 创建路由 LLM
    llm = get_routed_llm(temperature=0.3)

    logger.info("✅ 路由 LLM 创建成功")
    logger.info(f"候选模型：{[c.id for c in llm.router.candidates]}")
    logger.info(f"熔断阈值：{CIRCUIT_BREAKER_FAILURE_THRESHOLD} 次")
    logger.info(f"熔断时长：{CIRCUIT_BREAKER_OPEN_DURATION_SEC} 秒")
