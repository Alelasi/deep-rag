"""配置"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 修复 OpenMP 冲突
os.environ["no_proxy"] = "localhost,127.0.0.1"  # 绕过代理，防止 ChromaDB 连接被拦截
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# GPU 自动检测
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ChromaDB 数据目录（仅 chroma run --path 使用；代码禁止 PersistentClient）
# 默认中心：哲思灵智/向量数据库
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    r"D:\文档\ai提问相关\哲思灵智\向量数据库",
)
DOCS_DIR = DATA_DIR / "sample_docs"

# ChromaDB 服务器模式（host 用 localhost，勿与 127.0.0.1 混用）
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost")
CHROMA_SERVER_PORT = int(os.getenv("CHROMA_SERVER_PORT", "8000"))

_chroma_client = None

def get_chroma_client():
    """获取 ChromaDB 客户端 — 使用 HttpClient 连接服务器模式

    安全说明：
    - 禁止使用 PersistentClient 直接访问已有数据库（会导致 HNSW 索引损坏，已发生5次）
    - 必须先启动 ChromaDB 服务器：chroma run --path <DB_PATH> --port 8000
    - 所有代码通过本函数获取 HttpClient，由服务器统一管理 compactor
    """
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.HttpClient(
            host=CHROMA_SERVER_HOST,
            port=CHROMA_SERVER_PORT,
        )
    return _chroma_client

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")           # 智谱AI（永久免费GLM-4-Flash）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")  # 硅基流动（免费降级）

# 免费低延迟 API（v3.0新增，2026-07-13实测）
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")              # Groq：LPU芯片，延迟300-800ms
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")       # Cerebras：超快推理，延迟500-600ms
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")   # OpenRouter：免费模型聚合

# LLM后端切换：anthropic / zhipu / openai / ollama / none（规则模式）
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = 0.3  # 向后兼容默认值，新代码应使用 get_temperature()

# v2.9: 动态温度策略 — 根据任务类型调节 temperature
# 事实校验/文档评分需要确定性(temp=0)，答案生成适度创造性(temp=0.3)，
# 查询改写需要多样性(temp=0.5)，创意场景高创造性(temp=0.7)
TEMPERATURE_STRATEGY = {
    "fact_check": 0.0,      # 事实校验: 确定性
    "doc_grading": 0.0,     # 文档评分: 确定性
    "generation": 0.3,      # 答案生成: 适度创造性
    "query_rewrite": 0.5,   # 查询改写: 多样性
    "creative": 0.7,        # 创意场景: 高创造性
    "comparison": 0.2,      # 答案对比: 低随机性
    "arbitration": 0.2,     # 仲裁: 低随机性
}


def get_temperature(task_type: str) -> float:
    """根据任务类型获取对应的 temperature 值

    Args:
        task_type: 任务类型，见 TEMPERATURE_STRATEGY 的 key

    Returns:
        对应的 temperature 值，未知类型返回默认值 0.3
    """
    return TEMPERATURE_STRATEGY.get(task_type, 0.3)

# 向量数据库：默认 qdrant（中心在哲思灵智；避免 Chroma 损坏）
VECTOR_DB = os.getenv("VECTOR_DB", "qdrant")  # qdrant / chromadb / lancedb / pgvector
# server=Docker/独立服务（可多客户端同时访问，推荐）
# local=单进程独占磁盘（建库时不能开前端，不推荐日常）
QDRANT_MODE = os.getenv("QDRANT_MODE", "server")
QDRANT_PATH = os.getenv("QDRANT_PATH", r"D:\文档\ai提问相关\哲思灵智\qdrant_data")
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")  # 勿用 localhost(IPv6 慢)
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "proj_work")

# pgvector配置
PGVECTOR_HOST = os.getenv("PGVECTOR_HOST", "localhost")
PGVECTOR_PORT = int(os.getenv("PGVECTOR_PORT", "5432"))
PGVECTOR_DB = os.getenv("PGVECTOR_DB", "deep_rag")
PGVECTOR_USER = os.getenv("PGVECTOR_USER", "postgres")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "postgres")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "documents")

# Embedding 模型配置（双模式切换）
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "precise")  # "fast" | "precise"
_EMBEDDING_MODELS = {
    "fast": "BAAI/bge-small-zh-v1.5",    # 512维, 33M参数, 速度快
    "precise": "BAAI/bge-base-zh-v1.5",  # 768维, 110M参数, 精度高
}
# 如果 .env 显式设置了 EMBEDDING_MODEL，优先使用；否则按模式切换
_env_model = os.getenv("EMBEDDING_MODEL")
if _env_model:
    EMBEDDING_MODEL = _env_model
else:
    EMBEDDING_MODEL = _EMBEDDING_MODELS.get(EMBEDDING_MODE, _EMBEDDING_MODELS["precise"])
# Embedding 维度映射表（添加新模型时在此扩展）
EMBEDDING_DIM_MAP = {
    "BAAI/bge-small-zh-v1.5": 512,
    "BAAI/bge-base-zh-v1.5": 768,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "all-MiniLM-L6-v2": 384,
}


def get_embedding_dim(model_name: str = None) -> int:
    """获取 embedding 模型的维度。支持自动检测或查表。"""
    name = model_name or EMBEDDING_MODEL
    if name in EMBEDDING_DIM_MAP:
        return EMBEDDING_DIM_MAP[name]
    # 自动检测：加载模型获取维度
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(name)
        return m.get_sentence_embedding_dimension()
    except Exception:
        raise ValueError(
            f"未知 embedding 模型 '{name}'，维度无法自动检测。"
            f"请在 config.py 的 EMBEDDING_DIM_MAP 中手动添加。"
        )


# 检索配置
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# 语义缓存配置（v2.9.1新增）
ENABLE_SEMANTIC_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
SEMANTIC_CACHE_TTL = int(os.getenv("SEMANTIC_CACHE_TTL", "3600"))

# Prompt Caching配置（v2.9.2新增）
# 面试要点：固定内容在前、动态内容在后
# Ollama: 通过KV Cache量化减少显存占用
# API: 通过cache_control断点标记实现跨请求缓存
ENABLE_PROMPT_CACHE = os.getenv("ENABLE_PROMPT_CACHE", "true").lower() == "true"
PROMPT_CACHE_SYSTEM_PROMPT_FIXED = True  # System Prompt保持固定（缓存友好）

# KV Cache量化配置（v2.9.2新增，面试要点04-14）
# KV Cache是Transformer推理的核心优化，缓存前面token的K/V矩阵
# 量化可以减少显存占用，但可能影响长链路推理质量
# 配置项（需在系统环境变量中设置）：
#   OLLAMA_KV_CACHE_TYPE=q8_0    # KV Cache量化（q8_0/q4_0/f16）
#   OLLAMA_FLASH_ATTENTION=1     # 启用Flash Attention加速
#   OLLAMA_KEEP_ALIVE=5m         # 模型加载后保持5分钟
KV_CACHE_TYPE = os.getenv("OLLAMA_KV_CACHE_TYPE", "未配置")
FLASH_ATTENTION = os.getenv("OLLAMA_FLASH_ATTENTION", "未配置")
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "未配置")

# Agentic RAG配置（快速路由器）
ENABLE_AGENTIC_RAG = os.getenv("ENABLE_AGENTIC_RAG", "false").lower() == "true"
AGENTIC_ROUTER = os.getenv("AGENTIC_ROUTER", "llm")  # rule（零延迟）/ llm（智能，需LLM）
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "true").lower() == "true"
ENABLE_GRAPH_SEARCH = os.getenv("ENABLE_GRAPH_SEARCH", "false").lower() == "true"

# Self-RAG 闭环（默认关闭：省 5–7s 延迟；开启后 fact_check 失败可 regenerate）
ENABLE_SELF_RAG_LOOP = os.getenv("ENABLE_SELF_RAG_LOOP", "false").lower() == "true"
SELF_RAG_MAX_REGENERATE = int(os.getenv("SELF_RAG_MAX_REGENERATE", "1"))
# true=使用 LLM 版 analyze/grade/fact_check（失败仍降级 offline）；false=规则 offline（零 Key 可跑）
USE_LLM_PIPELINE_NODES = os.getenv("USE_LLM_PIPELINE_NODES", "false").lower() == "true"

# 能力版本（产品叙事）vs 包版本（pyproject）
CAPABILITY_VERSION = "2.9.x"
PACKAGE_VERSION = "0.2.9"

# 检索模式配置（v2.2新增，v2.4更新，v2.9新增function_calling）
# enhanced: 增强检索（问题拒识+多路推理+重排序，推荐）
# agentic: Agent动态路由（v2.1，RuleBasedRouter/LLMRouter选择工具）
# hybrid: 混合检索（BM25+向量，v1.0基线）
# agentic_react: ReAct Agent循环（v2.4，LLM自主决策多轮检索）
# function_calling: 原生Function Calling（v2.9，LLM通过tool_calls原生决策）
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "enhanced")  # enhanced / agentic / hybrid / agentic_react / function_calling

# GPU 加速配置
ENABLE_GPU_SEARCH = os.getenv("ENABLE_GPU_SEARCH", "false").lower() == "true"

# 模型路由配置（多候选 + 熔断器）
ENABLE_MODEL_ROUTING = os.getenv("ENABLE_MODEL_ROUTING", "false").lower() == "true"
MODEL_CANDIDATES = os.getenv("MODEL_CANDIDATES", "").strip()  # 逗号分隔，如 "anthropic:claude-sonnet-4,openai:gpt-4"
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "2"))
CIRCUIT_BREAKER_OPEN_DURATION_SEC = int(os.getenv("CIRCUIT_BREAKER_OPEN_DURATION_SEC", "30"))


def get_llm(temperature: float = None):
    """
    统一LLM工厂 — 支持多后端切换（v2.7：实例缓存 + 限流重试）

    优先级（auto模式）：anthropic → zhipu(免费) → openai → ollama → none(规则)

    v2.7优化：
    - LLM实例缓存：相同参数不重复创建
    - 限流器：控制请求频率避免429
    - 重试装饰器：429时自动退避重试
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    # 如果启用模型路由，返回路由包装器
    if ENABLE_MODEL_ROUTING and MODEL_CANDIDATES:
        from .llm.model_router_wrapper import get_routed_llm
        return get_routed_llm(temp)

    backend = LLM_BACKEND.lower()

    # auto: 2026-07-18 免费模型实测后优先「快+稳」
    # Cerebras GPT-OSS-120B 综合分最高(~805ms) → Groq 8B → Silicon → Zhipu
    if backend == "auto":
        if ANTHROPIC_API_KEY:
            backend = "anthropic"
        elif CEREBRAS_API_KEY:
            backend = "cerebras"  # 吞吐+质量最佳免费档
        elif GROQ_API_KEY:
            backend = "groq"  # 高 RPM + 低延迟
        elif SILICONFLOW_API_KEY:
            backend = "siliconcloud"
        elif ZHIPU_API_KEY:
            backend = "zhipu"
        elif OPENAI_API_KEY:
            backend = "openai"
        elif OPENROUTER_API_KEY:
            backend = "openrouter"
        elif LLM_MODEL and (LLM_MODEL.startswith("google/") or LLM_MODEL.startswith("qwen/") or LLM_MODEL.startswith("deepseek/")):
            backend = "lmstudio"
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

    # v2.7: 使用缓存的LLM实例
    from .llm.rate_limiter import get_cached_llm

    if backend == "anthropic":
        model = LLM_MODEL or "claude-sonnet-4-20250514"
        def _factory():
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temp)
        return get_cached_llm(backend, model, temp, _factory)

    if backend == "zhipu":
        model = LLM_MODEL or "glm-4-flash"  # 免费档实测优于 4.5-flash
        def _factory():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp,
                api_key=ZHIPU_API_KEY,
                base_url="https://open.bigmodel.cn/api/paas/v4",
            )
        return get_cached_llm(backend, model, temp, _factory)

    if backend == "ollama":
        model = LLM_MODEL or "qwen2.5:1.5b"
        def _factory():
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model, temperature=temp,
                base_url="http://localhost:11434",
                num_predict=300,  # v2.8.1: 禁用思考后300 token足够
            )
        return get_cached_llm(backend, model, temp, _factory)

    if backend == "lmstudio":
        model = LLM_MODEL or "google/gemma-3-4b"
        def _factory():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp,
                api_key="lm-studio",  # LM Studio 不需要真实 key
                base_url="http://localhost:11434/v1",
            )
        return get_cached_llm(backend, model, temp, _factory)

    if backend == "siliconcloud":
        model = LLM_MODEL or "THUDM/GLM-Z1-9B-0414"
        def _factory():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp,
                api_key=SILICONFLOW_API_KEY,
                base_url="https://api.siliconflow.cn/v1",
            )
        return get_cached_llm(backend, model, temp, _factory)

    # v3.0: 免费低延迟 API（2026-07-13实测）
    if backend == "groq":
        # Groq：LPU，~30RPM；默认 8B 更快更稳（2026-07-18 实测）
        # 备选：qwen/qwen3.6-27b, openai/gpt-oss-20b, openai/gpt-oss-120b
        model = LLM_MODEL or "llama-3.1-8b-instant"
        def _factory():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp,
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
        return get_cached_llm(backend, model, temp, _factory)

    if backend == "cerebras":
        # Cerebras：超高吞吐；默认 gpt-oss-120b（v3 评测第1）
        # 备选：gemma-4-31b, zai-glm-4.7
        model = LLM_MODEL or "gpt-oss-120b"
        def _factory():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp,
                api_key=CEREBRAS_API_KEY,
                base_url="https://api.cerebras.ai/v1",
            )
        return get_cached_llm(backend, model, temp, _factory)

    if backend == "openrouter":
        # OpenRouter：免费模型聚合，20RPM/50RPD
        # 推荐模型：meta-llama/llama-3.3-70b-instruct:free, openai/gpt-oss-20b:free
        model = LLM_MODEL or "meta-llama/llama-3.3-70b-instruct:free"
        def _factory():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp,
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )
        return get_cached_llm(backend, model, temp, _factory)

    if backend == "openai":
        model = LLM_MODEL or "gpt-4o-mini"
        def _factory():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temp)
        return get_cached_llm(backend, model, temp, _factory)

    raise ValueError(f"Unknown LLM_BACKEND: {backend}. Use: auto/anthropic/zhipu/ollama/openai/none")


def get_llm_with_fallback(temperature: float = None):
    """多LLM降级（2026-07-18 免费模型实测）

    链：主后端 → Cerebras gpt-oss-120b → Groq llama-3.1-8b
        → Silicon GLM-Z1-9B → Zhipu glm-4-flash → Ollama → 规则

    用于 Agent / 评测联网回退，避免单家 429 全挂。
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    primary = (LLM_BACKEND or "auto").lower()

    try:
        llm = get_llm(temp)
        if llm is not None:
            return llm
    except Exception as e:
        print(f"[LLM] Primary failed: {e}, trying fallback chain...")

    from langchain_openai import ChatOpenAI

    # 有序候选：(name, enabled, factory)
    chain = []
    if CEREBRAS_API_KEY and primary != "cerebras":
        chain.append(
            (
                "cerebras",
                lambda: ChatOpenAI(
                    model="gpt-oss-120b",
                    temperature=temp,
                    api_key=CEREBRAS_API_KEY,
                    base_url="https://api.cerebras.ai/v1",
                ),
            )
        )
    if GROQ_API_KEY and primary != "groq":
        chain.append(
            (
                "groq",
                lambda: ChatOpenAI(
                    model="llama-3.1-8b-instant",
                    temperature=temp,
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                ),
            )
        )
    if SILICONFLOW_API_KEY and primary not in ("siliconcloud", "silicon"):
        chain.append(
            (
                "siliconcloud",
                lambda: ChatOpenAI(
                    model="THUDM/GLM-Z1-9B-0414",
                    temperature=temp,
                    api_key=SILICONFLOW_API_KEY,
                    base_url="https://api.siliconflow.cn/v1",
                ),
            )
        )
    if ZHIPU_API_KEY and primary != "zhipu":
        chain.append(
            (
                "zhipu",
                lambda: ChatOpenAI(
                    model="glm-4-flash",
                    temperature=temp,
                    api_key=ZHIPU_API_KEY,
                    base_url="https://open.bigmodel.cn/api/paas/v4",
                ),
            )
        )

    for name, factory in chain:
        try:
            llm = factory()
            print(f"[LLM] fallback → {name}")
            return llm
        except Exception as e:
            print(f"[LLM] {name} fallback failed: {e}")

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen2.5:7b", temperature=temp)
    except Exception:
        pass

    print("[LLM] All LLM backends unavailable, using rule mode")
    return None
