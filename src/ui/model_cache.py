"""模型缓存 — 全局缓存 SentenceTransformer 和 LLM 实例，避免重复加载"""
import functools


@functools.lru_cache(maxsize=4)
def get_embedding_model(model_name: str, device: str):
    """全局缓存 SentenceTransformer 实例

    Args:
        model_name: 模型名称（如 BAAI/bge-base-zh-v1.5）
        device: 设备（cuda / cpu）

    Returns:
        SentenceTransformer 实例
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


@functools.lru_cache(maxsize=4)
def get_llm_cached(backend: str, model: str, temperature: float):
    """全局缓存 LLM 实例

    Args:
        backend: LLM 后端（zhipu / anthropic / openai / ollama）
        model: 模型名称
        temperature: 温度参数

    Returns:
        LLM 实例
    """
    from src.config import get_llm
    return get_llm(temperature)


def clear_all_caches():
    """清除所有缓存（切换模型时调用）"""
    get_embedding_model.cache_clear()
    get_llm_cached.cache_clear()
