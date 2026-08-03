"""Prompt Caching管理 — v2.9.2新增

面试要点（04-14 KV Cache/Prompt Caching）：
- KV Cache是「单次推理内」的优化（同一次生成的不同token之间复用）
- Prompt Caching是「跨请求」的优化（不同请求之间复用相同前缀）
- 核心原则：固定内容在前、动态内容在后
- 缓存命中率监控是工程必备

本模块提供：
1. Prompt结构化管理（固定/动态分离）
2. 缓存友好的消息构建
3. 缓存命中率统计

v3.0改进（缓存与限流统一抽象）：
- 统计计数器加锁（线程安全）
- 接入统一 CacheBackend 抽象（backend 参数，默认 MemoryBackend，预留 RedisBackend 钩子）
- 日志统一使用标准库 logging.getLogger(__name__)
- 补充类型注解

用法：
    from src.llm.prompt_cache import PromptCacheManager

    manager = PromptCacheManager()

    # 构建缓存友好的消息
    messages = manager.build_messages(
        system_prompt="你是一个...",
        context="检索到的文档...",
        user_query="用户问题",
    )

    # 获取缓存统计
    stats = manager.get_stats()
"""
import time
import logging
import threading
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from src.retrieval.cache import CacheBackend, MemoryBackend, RedisBackend

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """缓存统计"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_saved: int = 0
    last_hit_time: Optional[float] = None

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate * 100, 1),
            "total_tokens_saved": self.total_tokens_saved,
        }


class PromptCacheManager:
    """Prompt缓存管理器

    核心设计：
    1. 固定内容（System Prompt、Few-shot）放在前面
    2. 动态内容（用户查询、时间戳）放在后面
    3. 统计缓存命中率

    对于Ollama后端：
    - KV Cache通过OLLAMA_KV_CACHE_TYPE配置量化
    - 相同前缀的请求会自动复用KV Cache

    对于API后端（Claude/OpenAI）：
    - Claude: 使用cache_control断点标记
    - OpenAI: 自动缓存（>1024 tokens）

    Args:
        backend: 缓存后端（默认 MemoryBackend），预留 RedisBackend 钩子
    """

    def __init__(self, backend: Optional[CacheBackend] = None):
        self._backend: CacheBackend = backend if backend is not None else MemoryBackend()
        self._stats = CacheStats()
        self._last_system_prompt_hash: Optional[str] = None
        self._system_prompt_cache_count: int = 0
        self._lock = threading.Lock()  # 保护统计与命中状态

    def build_messages(
        self,
        system_prompt: str,
        user_query: str,
        context: Optional[str] = None,
        few_shots: Optional[List[Dict[str, str]]] = None,
        dynamic_prefix: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """构建缓存友好的消息列表

        顺序：System Prompt → Few-shot → Context → 动态前缀 → 用户查询

        Args:
            system_prompt: 系统提示词（固定内容，放最前面）
            user_query: 用户查询（动态内容，放最后面）
            context: 检索到的文档上下文（半固定）
            few_shots: Few-shot示例列表（固定内容）
            dynamic_prefix: 动态前缀（如日期、用户名）

        Returns:
            消息列表
        """
        messages: List[Dict[str, str]] = []

        # 1. System Prompt（固定，放最前面，最有可能被缓存）
        messages.append({"role": "system", "content": system_prompt})

        # 2. Few-shot示例（固定，紧随System Prompt）
        if few_shots:
            for shot in few_shots:
                if "user" in shot:
                    messages.append({"role": "user", "content": shot["user"]})
                if "assistant" in shot:
                    messages.append({"role": "assistant", "content": shot["assistant"]})

        # 3. 上下文文档（半固定，放在动态内容之前）
        if context:
            messages.append({"role": "user", "content": f"参考文档：\n{context}"})
            messages.append({"role": "assistant", "content": "已收到文档，请提问。"})

        # 4. 动态前缀（如有）
        if dynamic_prefix:
            user_content = f"{dynamic_prefix}\n\n{user_query}"
        else:
            user_content = user_query

        # 5. 用户查询（动态，放最后）
        messages.append({"role": "user", "content": user_content})

        # 统计（线程安全）
        with self._lock:
            self._stats.total_requests += 1
        self._check_cache_hit(system_prompt)

        return messages

    def build_rag_messages(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        collection_name: str = "default",
    ) -> List[Dict[str, str]]:
        """构建RAG问答的消息列表（缓存优化版）

        固定部分：System Prompt + RAG指令
        动态部分：用户问题

        Args:
            question: 用户问题
            context: 检索到的文档
            system_prompt: 自定义系统提示词
            collection_name: 知识库集合名
        """
        default_system = """你是知识库问答专家。根据提供的文档回答问题。

## 回答规范
1. 只基于文档内容回答，不要编造
2. 每个关键事实标注来源：[来源: 文档名, 第X块]
3. 如果文档中没有相关信息，明确说明

## 引用格式
- 事实性陈述必须标注来源
- 使用方括号格式：[来源: 文件名]"""

        return self.build_messages(
            system_prompt=system_prompt or default_system,
            user_query=f"问题：{question}",
            context=context,
        )

    def _check_cache_hit(self, system_prompt: str) -> None:
        """检查是否命中缓存（基于System Prompt哈希）；线程安全"""
        current_hash = hashlib.md5(system_prompt.encode()).hexdigest()

        with self._lock:
            if self._last_system_prompt_hash == current_hash:
                self._stats.cache_hits += 1
                self._system_prompt_cache_count += 1
                self._stats.last_hit_time = time.time()
                logger.debug(
                    f"[PromptCache] System Prompt缓存命中 "
                    f"(连续命中: {self._system_prompt_cache_count})"
                )
            else:
                self._stats.cache_misses += 1
                self._system_prompt_cache_count = 0
                self._last_system_prompt_hash = current_hash

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计（线程安全）"""
        with self._lock:
            return self._stats.to_dict()

    def estimate_tokens_saved(self) -> int:
        """估算节省的token数

        假设System Prompt平均500 tokens，每次命中节省重新计算的开销。
        """
        with self._lock:
            return self._stats.cache_hits * 500

    def get_cache_config(self) -> Dict[str, Any]:
        """获取当前缓存配置建议"""
        import os
        return {
            "backend": type(self._backend).__name__,
            "ollama_kv_cache_type": os.getenv("OLLAMA_KV_CACHE_TYPE", "未配置"),
            "ollama_flash_attention": os.getenv("OLLAMA_FLASH_ATTENTION", "未配置"),
            "ollama_keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "未配置"),
            "recommendations": [
                "设置 OLLAMA_KV_CACHE_TYPE=q8_0 启用KV Cache量化",
                "设置 OLLAMA_FLASH_ATTENTION=1 启用Flash Attention",
                "设置 OLLAMA_KEEP_ALIVE=5m 保持模型加载",
                "System Prompt保持固定，动态内容放在最后",
            ],
        }


# ============================================================
# 缓存友好的Prompt模板
# ============================================================

class CacheFriendlyPrompts:
    """缓存友好的Prompt模板

    设计原则：
    1. 固定指令放前面（可缓存部分）
    2. 变量占位符放后面
    3. Few-shot示例作为固定前缀
    """

    # RAG问答模板（固定部分）
    RAG_SYSTEM = """你是知识库问答专家。根据文档回答问题。

## 规则
1. 只基于文档回答
2. 标注引用来源
3. 无相关信息时说明

## 输出格式
结论 → 证据 → 引用"""

    # 事实核查模板（固定部分）
    FACT_CHECK_SYSTEM = """你是事实核查专家。验证声明的准确性。

## 核查维度
1. 事实正确性
2. 数据准确性
3. 逻辑一致性

## 输出格式
JSON: {"verified": true/false, "confidence": 0-100, "evidence": [...]}"""

    # 代码审查模板（固定部分）
    CODE_REVIEW_SYSTEM = """你是资深代码审查专家。

## 审查维度
1. 功能正确性
2. 安全漏洞
3. 性能问题
4. 代码规范

## 输出格式
问题列表 + 严重程度 + 修改建议"""

    @staticmethod
    def get_rag_few_shots() -> List[Dict[str, str]]:
        """RAG问答的Few-shot示例（固定，可缓存）"""
        return [
            {
                "user": "什么是向量检索？",
                "assistant": "向量检索是将文本转换为向量表示，通过计算向量相似度来搜索相关文档的技术。[来源: 检索技术文档]"
            },
            {
                "user": "RAG系统的核心组件有哪些？",
                "assistant": "RAG系统的核心组件包括：1) 文档索引模块 2) 向量检索模块 3) 上下文组装模块 4) LLM生成模块。[来源: RAG架构文档]"
            },
        ]


# ============================================================
# 全局实例
# ============================================================

_manager_instance: Optional[PromptCacheManager] = None
_manager_lock = threading.Lock()  # 保护单例创建


def get_prompt_cache_manager() -> PromptCacheManager:
    """获取全局Prompt缓存管理器（线程安全单例）"""
    global _manager_instance
    if _manager_instance is not None:
        return _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = PromptCacheManager()
        return _manager_instance
