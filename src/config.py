"""配置"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "sample_docs"

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM后端切换：anthropic / ollama / openai / none（规则模式）
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = 0.3

# 向量数据库配置（2026 最新推荐：lancedb > chromadb）
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb")  # chromadb / lancedb / qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "deep_rag_docs")

# 检索配置
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Agentic RAG配置（快速路由器）
ENABLE_AGENTIC_RAG = os.getenv("ENABLE_AGENTIC_RAG", "false").lower() == "true"
AGENTIC_ROUTER = os.getenv("AGENTIC_ROUTER", "rule")  # rule（零延迟）/ llm（智能）
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "true").lower() == "true"
ENABLE_GRAPH_SEARCH = os.getenv("ENABLE_GRAPH_SEARCH", "false").lower() == "true"

# GPU 加速配置
ENABLE_GPU_SEARCH = os.getenv("ENABLE_GPU_SEARCH", "false").lower() == "true"


def get_llm(temperature: float = None):
    """
    统一LLM工厂 — 支持 CC（Claude Code）和本地部署模型切换

    优先级：API Key > 本地 Ollama > 规则模式（零成本）

    切换示例：
      # CC / 云端（默认）
      export LLM_BACKEND=auto

      # 本地 Ollama（离线可用）
      export LLM_BACKEND=ollama
      export LLM_MODEL=qwen2.5:7b

      # 纯规则（零延迟、零成本）
      export LLM_BACKEND=none
    """
    backend = LLM_BACKEND.lower()
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    # auto: API Key → Ollama → none
    if backend == "auto":
        if ANTHROPIC_API_KEY:
            backend = "anthropic"
        elif OPENAI_API_KEY:
            backend = "openai"
        else:
            # 尝试本地 Ollama
            try:
                import urllib.request
                req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        backend = "ollama"
                    else:
                        backend = "none"
            except Exception:
                backend = "none"

    if backend == "none":
        return None

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = LLM_MODEL or "claude-sonnet-4-20250514"
        return ChatAnthropic(model=model, temperature=temp)

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        model = LLM_MODEL or "qwen2.5:7b"
        return ChatOllama(model=model, temperature=temp, base_url="http://localhost:11434")

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        model = LLM_MODEL or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temp)

    raise ValueError(f"Unknown LLM_BACKEND: {backend}. Use: auto/anthropic/ollama/openai/none")
