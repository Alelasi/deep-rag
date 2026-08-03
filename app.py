"""DeepRAG v2.9 — Streamlit 生产演示界面（6-Tab）"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
# 向量数据库配置：优先读环境变量，本地 Windows 路径仅作 fallback
os.environ.setdefault('QDRANT_MODE', os.getenv('QDRANT_MODE', 'server'))
os.environ.setdefault('QDRANT_HOST', os.getenv('QDRANT_HOST', '127.0.0.1'))
os.environ.setdefault('QDRANT_PORT', os.getenv('QDRANT_PORT', '6333'))
if os.getenv('QDRANT_PATH') is None and os.name == 'nt' and os.getenv('QDRANT_MODE', 'server') != 'cloud':
    os.environ.setdefault('QDRANT_PATH', r'D:\文档\ai提问相关\哲思灵智\qdrant_data')
os.environ.setdefault('VECTOR_DB', os.getenv('VECTOR_DB', 'qdrant'))
os.environ.setdefault('OMP_NUM_THREADS', '2')

import sys
import time
import json
import re

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

try:
    from src.logging_config import get_logger
except Exception:
    import logging
    def get_logger(n):  # type: ignore
        return logging.getLogger(n)
logger = get_logger(__name__)

# === Qdrant 向量数据库（替代 ChromaDB，解决 HNSW 重启损坏问题）===
def _check_vector_store():
    """检查向量数据库状态：优先有数据的 collection"""
    try:
        from src.retrieval.qdrant_retriever import get_qdrant_retriever, list_collection_stats
        stats = list_collection_stats()
        nonempty = [s for s in stats if s.get("points", 0) > 0]
        if nonempty:
            best = max(nonempty, key=lambda x: x["points"])
            retriever = get_qdrant_retriever(best["name"])
            logger.info(f"[Qdrant] Connected collections={len(stats)} "
                        f"best={best['name']} docs={best['points']}")
            return True
        retriever = get_qdrant_retriever("proj_psychology")
        logger.info(f"[Qdrant] Connected, {retriever.count()} documents (psychology)")
        return True
    except Exception as e:
        logger.error(f"[Qdrant] Connection failed: {e}")
        return False

_check_vector_store()

import streamlit as st
import pandas as pd

st.set_page_config(page_title="DeepRAG v2.9", page_icon="🔍", layout="wide")


# === 缓存索引器（模块级定义，供多处调用）===
@st.cache_resource
def get_indexer_cached(_collection_name):
    """缓存索引器实例（带集合名参数）"""
    from src.graph import get_indexer
    return get_indexer(_collection_name)


# === 辅助函数 ===

def render_answer(answer_text: str):
    """渲染答案 — 尝试解析三段式结构（直接回答/详细解释/引用来源），否则直接展示"""
    if not answer_text:
        st.info("（空回答）")
        return

    has_direct = "【直接回答】" in answer_text
    has_detail = "【详细解释】" in answer_text
    has_citation = "【引用来源】" in answer_text

    if has_direct or has_detail or has_citation:
        direct_match = re.search(r"【直接回答】(.*?)(?=【详细解释】|【引用来源】|$)", answer_text, re.DOTALL)
        detail_match = re.search(r"【详细解释】(.*?)(?=【引用来源】|$)", answer_text, re.DOTALL)
        citation_match = re.search(r"【引用来源】(.*?)$", answer_text, re.DOTALL)

        if direct_match:
            st.info(f"**📌 直接回答**\n\n{direct_match.group(1).strip()}")
        if detail_match:
            st.markdown(f"**📝 详细解释**\n\n{detail_match.group(1).strip()}")
        if citation_match and citation_match.group(1).strip():
            with st.expander("📎 引用来源", expanded=False):
                st.markdown(citation_match.group(1).strip())
    else:
        st.markdown(answer_text)


# 执行步骤名称映射
STEP_NAME_MAP = {
    "query_rewritten": "查询预处理",
    "Retrieved": "多路召回（BM25+向量+RRF）",
    "Reranked": "精排重排序",
    "Graded": "文档评分（CRAG）",
    "Generated": "答案生成",
    "Fact check": "事实校验（Self-RAG）",
    "web_search": "Web搜索兜底",
    "Web": "Web搜索兜底",
}


def map_step_name(step_text: str) -> str:
    """将 history 中的英文步骤名映射为中文"""
    for key, val in STEP_NAME_MAP.items():
        if key in step_text:
            return step_text.replace(key, val)
    return step_text


# === 侧边栏 ===
with st.sidebar:
    st.title("🔍 DeepRAG v2.9")
    st.markdown("**自纠错多源知识 Agent（可上线工程版）**")
    st.markdown("CRAG + 可选 Self-RAG + 混合检索 + API 鉴权/限流")
    st.caption("API: `uvicorn api:app --port 8000` · 文档见 `/docs`")
    st.markdown("---")

    # 生产开关（写入 session，并提示 .env）
    st.subheader("🛡️ 生产/质量开关")
    try:
        from src.config import ENABLE_SELF_RAG_LOOP, USE_LLM_PIPELINE_NODES, PACKAGE_VERSION
        st.caption(f"包版本 {PACKAGE_VERSION} | Self-RAG闭环={ENABLE_SELF_RAG_LOOP} | LLM节点={USE_LLM_PIPELINE_NODES}")
    except Exception:
        pass
    st.checkbox(
        "Self-RAG 重新生成闭环（需 .env ENABLE_SELF_RAG_LOOP=true 重启生效）",
        value=False,
        disabled=True,
        help="运行时改 env 需重启进程；默认关闭以省延迟",
    )
    st.markdown("---")

    # === GPU 状态 ===
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.caption(f"GPU: 🟢 CUDA可用 ({gpu_name})")
        else:
            st.caption("GPU: 🟡 CPU模式")
    except ImportError:
        st.caption("GPU: ❌ PyTorch未安装")

    # === 加载已保存的配置 ===
    from src.ui.config_persistence import load_env_dict, save_to_env
    saved_env = load_env_dict()

    # === LLM 配置 ===
    st.subheader("⚙️ LLM 配置")

    backend_options = [
        "cerebras", "groq", "siliconcloud", "zhipu",
        "anthropic", "openai", "ollama", "lmstudio", "none",
    ]
    saved_backend = saved_env.get("LLM_BACKEND", "cerebras")
    backend_idx = backend_options.index(saved_backend) if saved_backend in backend_options else 0

    llm_backend = st.selectbox(
        "LLM 后端",
        backend_options,
        index=backend_idx,
        format_func=lambda x: {
            "cerebras": "⚡ Cerebras（免费超快 · gpt-oss-120b）",
            "groq": "⚡ Groq（免费低延迟 · Llama3.1-8B）",
            "siliconcloud": "🆓 硅基流动 SiliconCloud（GLM-Z1-9B）",
            "zhipu": "🆓 智谱 AI（GLM-4-Flash）",
            "anthropic": "🟠 Anthropic Claude",
            "openai": "🟢 OpenAI GPT",
            "ollama": "🖥️ 本地 Ollama",
            "lmstudio": "🏠 本地 LM Studio",
            "none": "📋 纯规则模式（无LLM）",
        }.get(x, x),
        help="推荐免费快链：Cerebras → Groq → Silicon → Zhipu（见 2026-07-18 评测）",
    )

    zhipu_key = ""
    siliconflow_key = ""
    cerebras_key = ""
    groq_key = ""
    anthropic_key = ""
    openai_key = ""
    ollama_url = "http://localhost:11434"
    saved_model = saved_env.get("LLM_MODEL", "gpt-oss-120b")

    if llm_backend == "cerebras":
        cerebras_key = st.text_input(
            "Cerebras API Key",
            value=saved_env.get("CEREBRAS_API_KEY", ""),
            type="password",
            placeholder="https://cloud.cerebras.ai 注册",
        )
        _cb_models = ["gpt-oss-120b", "gemma-4-31b", "zai-glm-4.7"]
        default_idx = _cb_models.index(saved_model) if saved_model in _cb_models else 0
        llm_model = st.selectbox("选择模型", _cb_models, index=default_idx)
        st.caption("⚡ 实测 ~0.8s · 综合分最高 · 日额度约 1M tokens")

    elif llm_backend == "groq":
        groq_key = st.text_input(
            "Groq API Key",
            value=saved_env.get("GROQ_API_KEY", ""),
            type="password",
            placeholder="https://console.groq.com 注册",
        )
        _gq_models = [
            "llama-3.1-8b-instant",
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
        ]
        default_idx = _gq_models.index(saved_model) if saved_model in _gq_models else 0
        llm_model = st.selectbox("选择模型", _gq_models, index=default_idx)
        st.caption("⚡ ~30 RPM · 默认 8B 最稳；大模型易限流")

    elif llm_backend == "siliconcloud":
        siliconflow_key = st.text_input(
            "SiliconFlow API Key", value=saved_env.get("SILICONFLOW_API_KEY", ""),
            type="password", placeholder="在 https://cloud.siliconflow.cn 注册获取"
        )
        # SiliconCloud 免费模型列表
        _sf_free_models = [
            "THUDM/GLM-Z1-9B-0414",       # 免费, 9B推理模型
            "THUDM/GLM-4-9B-0414",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-8B",
        ]
        _sf_paid_models = [
            "deepseek-ai/DeepSeek-V4-Flash",  # ¥1/M, 极速
            "deepseek-ai/DeepSeek-V3.2",      # ¥4/M, 高质量
            "Qwen/Qwen3.5-35B-A3B",           # ¥0.4/M, MoE
        ]
        _sf_all_models = _sf_free_models + _sf_paid_models
        default_idx = 0
        if saved_model in _sf_all_models:
            default_idx = _sf_all_models.index(saved_model)
        llm_model = st.selectbox(
            "选择模型", _sf_all_models, index=default_idx,
            format_func=lambda x: f"{x} {'🆓免费' if x in _sf_free_models else '💰付费'}"
        )
        st.caption("🆓 有额度模型 | 质量尚可但延迟常 3–15s（不配当高速主模型）")

    elif llm_backend == "zhipu":
        zhipu_key = st.text_input(
            "智谱 API Key", value=saved_env.get("ZHIPU_API_KEY", ""),
            type="password", placeholder="在 https://open.bigmodel.cn 注册获取"
        )
        _zp_models = ["glm-4-flash", "glm-4.5-flash", "glm-4.5-air"]
        default_idx = _zp_models.index(saved_model) if saved_model in _zp_models else 0
        llm_model = st.selectbox("选择模型", _zp_models, index=default_idx)
        st.caption("🆓 Flash 档 · 中文稳 · 偏慢")
    elif llm_backend == "anthropic":
        anthropic_key = st.text_input("Anthropic API Key", value=saved_env.get("ANTHROPIC_API_KEY", ""), type="password")
        llm_model = st.text_input("模型名称", value=saved_model or "claude-sonnet-4-20250514")
    elif llm_backend == "openai":
        openai_key = st.text_input("OpenAI API Key", value=saved_env.get("OPENAI_API_KEY", ""), type="password")
        llm_model = st.text_input("模型名称", value=saved_model or "gpt-4o-mini")
    elif llm_backend == "ollama":
        ollama_url = st.text_input("Ollama URL", value=saved_env.get("OLLAMA_URL", "http://localhost:11434"))
        # v2.8.2: 动态列出已安装的 Ollama 模型
        try:
            import urllib.request as _urllib
            _r = _urllib.urlopen("http://localhost:11434/api/tags", timeout=3)
            _data = json.loads(_r.read())
            _model_names = [m["name"] for m in _data.get("models", [])]
            if _model_names:
                # 按大小排序，小的在前
                _model_names.sort(key=lambda n: next(
                    (m.get("size", 0) for m in _data.get("models", []) if m.get("name") == n), 0
                ))
                default_idx = 0
                if saved_model in _model_names:
                    default_idx = _model_names.index(saved_model)
                elif f"{saved_model}:latest" in _model_names:
                    default_idx = _model_names.index(f"{saved_model}:latest")
                llm_model = st.selectbox("选择模型", _model_names, index=default_idx)
            else:
                llm_model = st.text_input("模型名称", value=saved_model or "qwen2.5:7b")
                st.warning("Ollama 中没有已安装的模型，请先 `ollama pull <model>`")
        except Exception:
            llm_model = st.text_input("模型名称", value=saved_model or "qwen2.5:7b")
            st.warning("无法连接 Ollama (http://localhost:11434)，请确认 Ollama 正在运行。")

        # v2.8.2: 思考模式开关（仅对 qwen3 系列有效）
        if "qwen3" in (llm_model or "").lower():
            _think_enabled = st.checkbox(
                "🧠 思考模式（复杂问题开启，简单问题关闭以加速）",
                value=saved_env.get("OLLAMA_THINK", "false").lower() == "true",
                help="开启后 Qwen3 会先推理再回答，质量更高但速度慢5-10倍。关闭后直接回答，速度快。"
            )
            # 同步到全局
            from src.llm.ollama_helper import set_think_mode
            set_think_mode(_think_enabled)
            # 保存到环境变量
            os.environ["OLLAMA_THINK"] = "true" if _think_enabled else "false"
    elif llm_backend == "lmstudio":
        st.info("🏠 LM Studio API: http://localhost:11434/v1（需要在 LM Studio 中加载模型）")
        # 列出可用模型
        try:
            import urllib.request as _urllib
            _r = _urllib.urlopen("http://localhost:11434/v1/models", timeout=3)
            _data = json.loads(_r.read())
            _model_ids = [m["id"] for m in _data.get("data", [])]
            if _model_ids:
                default_idx = 0
                if saved_model in _model_ids:
                    default_idx = _model_ids.index(saved_model)
                llm_model = st.selectbox("选择已加载模型", _model_ids, index=default_idx)
            else:
                llm_model = st.text_input("模型名称", value=saved_model or "google/gemma-3-4b")
                st.warning("LM Studio 中没有已加载的模型，请先在 LM Studio 中加载一个模型。")
        except Exception:
            llm_model = st.text_input("模型名称", value=saved_model or "google/gemma-3-4b")
            st.warning("无法连接 LM Studio (http://localhost:11434)，请确认 LM Studio 正在运行。")
    else:
        llm_model = ""
        st.info("📋 纯规则模式：不调用LLM，使用正则匹配路由")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    # 保存配置按钮
    if st.button("💾 保存配置到.env"):
        env_updates = {"LLM_BACKEND": llm_backend, "LLM_MODEL": llm_model, "TEMPERATURE": str(temperature)}
        if llm_backend == "ollama":
            env_updates["OLLAMA_THINK"] = os.environ.get("OLLAMA_THINK", "false")
        env_updates["ENABLE_RERANKER"] = os.environ.get("ENABLE_RERANKER", "false")
        if zhipu_key:
            env_updates["ZHIPU_API_KEY"] = zhipu_key
        if siliconflow_key or os.environ.get("SILICONFLOW_API_KEY"):
            env_updates["SILICONFLOW_API_KEY"] = os.environ.get("SILICONFLOW_API_KEY", siliconflow_key)
        if cerebras_key or os.environ.get("CEREBRAS_API_KEY"):
            env_updates["CEREBRAS_API_KEY"] = cerebras_key or os.environ.get("CEREBRAS_API_KEY", "")
        if groq_key or os.environ.get("GROQ_API_KEY"):
            env_updates["GROQ_API_KEY"] = groq_key or os.environ.get("GROQ_API_KEY", "")
        if anthropic_key:
            env_updates["ANTHROPIC_API_KEY"] = anthropic_key
        if openai_key:
            env_updates["OPENAI_API_KEY"] = openai_key
        save_to_env(env_updates)
        st.success("配置已保存，重启后自动加载")

    # 应用配置到环境变量
    os.environ["LLM_BACKEND"] = llm_backend
    os.environ["LLM_MODEL"] = llm_model
    if siliconflow_key:
        os.environ["SILICONFLOW_API_KEY"] = siliconflow_key
    if zhipu_key:
        os.environ["ZHIPU_API_KEY"] = zhipu_key
    if cerebras_key:
        os.environ["CEREBRAS_API_KEY"] = cerebras_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    # LLM 连接测试
    if st.button("🔌 测试 LLM 连接"):
        if llm_backend == "none":
            st.warning("当前为纯规则模式，无LLM连接")
        else:
            with st.spinner("测试中..."):
                try:
                    import importlib
                    import src.config
                    importlib.reload(src.config)
                    from src.config import get_llm
                    llm = get_llm(temperature)
                    if llm is not None:
                        from langchain_core.messages import HumanMessage
                        resp = llm.invoke([HumanMessage(content="说'连接成功'四个字")])
                        text = resp.content if hasattr(resp, "content") else str(resp)
                        st.success(f"✅ 连接成功！模型回复: {text[:50]}")
                    else:
                        st.error("❌ LLM返回None，请检查API Key")
                except Exception as e:
                    st.error(f"❌ 连接失败: {str(e)[:200]}")

    st.markdown("---")

    # === Embedding 切换 ===
    st.subheader("🔤 Embedding 模型")
    saved_emb_mode = saved_env.get("EMBEDDING_MODE", "precise")
    emb_options = ["precise", "fast"]
    emb_idx = emb_options.index(saved_emb_mode) if saved_emb_mode in emb_options else 0
    embedding_mode = st.radio(
        "模式",
        emb_options,
        index=emb_idx,
        format_func=lambda x: {
            "precise": "🎯 精确模式 (bge-base-zh, 768维)",
            "fast": "⚡ 快速模式 (bge-small-zh, 512维)",
        }.get(x, x),
    )
    if st.button("切换Embedding"):
        save_to_env({"EMBEDDING_MODE": embedding_mode})
        from src.ui.model_cache import clear_all_caches
        clear_all_caches()
        st.success(f"已切换到{embedding_mode}模式，缓存已清除")
        st.rerun()

    st.markdown("---")

    # === 检索配置 ===
    st.subheader("🔧 检索配置")
    mode = st.radio(
        "检索模式",
        ["enhanced", "agentic_react", "precision"],
        index=0,  # 默认 Enhanced：通常 5–15s；ReAct 常 40s+ 不适合默认
        format_func=lambda x: {
            "enhanced": "🔄 Enhanced RAG（推荐·固定Pipeline·较快）",
            "agentic_react": "🤖 Agentic ReAct（慢·多轮LLM·40s+）",
            "precision": "🎯 精准模式（双Agent·更慢）",
        }.get(x, x),
    )
    max_retries = st.slider("最大检索轮次", 1, 5, 1)  # 默认 1 轮，控延迟
    top_k = st.slider("Top-K 检索数量", 1, 20, 5)

    if mode == "agentic_react":
        st.warning("⚠️ ReAct 会多轮调用 LLM，免费云模型常见 30–60s；要 <10s 请用 Enhanced")
    if mode == "agentic_react" and llm_backend == "none":
        st.warning("⚠️ ReAct模式需要LLM，纯规则模式会自动降级")
    if mode == "precision" and llm_backend == "none":
        st.warning("⚠️ 精准模式需要LLM做双Agent对比，纯规则模式无法使用")

    # v2.8.6: 精准模式双Agent配置
    if mode == "precision":
        st.markdown("**双Agent配置:**")
        precision_config = st.selectbox(
            "配置方案",
            ["cross_fast", "flash_dual", "cross", "z1_dual"],
            format_func=lambda x: {
                "cross_fast": "🎯 Z1+Flash 极速（推荐~6s, 准确率8.9）",
                "flash_dual": "⚡ Flash × 2（最快~5s）",
                "cross": "🔄 Z1 + Flash 标准（~9s）",
                "z1_dual": "📝 Z1 × 2（质量最优~15s）",
            }.get(x, x),
        )
        st.session_state["precision_config"] = precision_config

        strategy_choice = st.selectbox(
            "提示词策略",
            ["socratic_concise", "direct_analytical", "cot_concise", "socratic_analytical"],
            format_func=lambda x: {
                "socratic_concise": "苏格拉底+精简（推荐, 准确率最高）",
                "direct_analytical": "直接+分析（均衡）",
                "cot_concise": "思维链+精简（推理强）",
                "socratic_analytical": "苏格拉底+分析（深度好）",
            }.get(x, x),
        )
        st.session_state["precision_strategy"] = strategy_choice

        fast_mode = st.checkbox("极速模式（跳过LLM对比，本地检测矛盾）", value=True)
        st.session_state["precision_fast"] = fast_mode

    st.markdown("---")

    # === Reranker 配置（v2.8.3）===
    st.subheader("🎯 Reranker 精排")
    enable_reranker = st.checkbox(
        "启用 Reranker",
        value=saved_env.get("ENABLE_RERANKER", "false").lower() == "true",
        help="开启后对检索结果做Cross-Encoder精排，提升准确率"
    )
    os.environ["ENABLE_RERANKER"] = "true" if enable_reranker else "false"

    if enable_reranker:
        reranker_api_key = st.text_input(
            "SiliconFlow API Key（可选，启用API毫秒级rerank）",
            value=saved_env.get("SILICONFLOW_API_KEY", ""),
            type="password",
            placeholder="在 https://cloud.siliconflow.cn 注册获取",
            help="有Key走API模式(~200ms)，无Key走CPU模式(~500ms-1s)，均不占GPU显存"
        )
        if reranker_api_key:
            os.environ["SILICONFLOW_API_KEY"] = reranker_api_key
            st.caption("🟢 API模式（~200ms，不占VRAM）")
        else:
            st.caption("🟡 CPU模式（~500ms-1s，不占VRAM）")

    st.markdown("---")

    # === 知识库管理 ===
    st.subheader("📚 知识库管理")

    # v2.6: 动态获取可用collection列表
    def get_available_collections():
        """获取 collection 列表 + 点数（有数据优先）"""
        try:
            from src.retrieval.qdrant_retriever import list_collection_stats
            stats = list_collection_stats()
            stats = sorted(stats, key=lambda s: -s.get("points", 0))
            return stats if stats else [{"name": "proj_psychology", "points": 0}]
        except Exception:
            try:
                from src.config import get_chroma_client
                client = get_chroma_client()
                cols = client.list_collections()
                return [{"name": c.name, "points": c.count()} for c in cols] or [
                    {"name": "general_kb", "points": 0}
                ]
            except Exception:
                return [{"name": "proj_psychology", "points": 0}]

    _col_stats = get_available_collections()
    _col_names = [s["name"] for s in _col_stats]
    # 心理库更适合 MBTI/人格问题；论文库选中时给醒目提示
    _default_idx = 0
    for i, n in enumerate(_col_names):
        if n == "proj_psychology":
            _default_idx = i
            break
    collection = st.selectbox(
        "知识库",
        _col_names,
        index=_default_idx if _col_names else 0,
        format_func=lambda n: f"{n} ({next((s['points'] for s in _col_stats if s['name']==n), 0)} 条)",
    )
    if collection == "proj_thesis":
        st.warning("⚠️ 当前是**论文库**，问 INTJ/MBTI 会检索到算法/实验文档 → 答非所问。人格问题请选 **proj_psychology**。")
    elif collection and "thesis" in collection:
        st.warning("⚠️ 该库偏论文技术内容，人格/MBTI 问题请换心理/工作区知识库。")
    doc_dir = st.text_input("文档目录", value="", placeholder="输入文档目录路径，如 data/docs")

    if st.button("📥 索引文档"):
        if not doc_dir:
            st.warning("请先输入文档目录路径")
        elif not os.path.isdir(doc_dir):
            st.error(f"目录不存在: {doc_dir}")
        else:
            indexer = get_indexer_cached(collection)
            with st.spinner("索引中..."):
                count = indexer.index_directory(doc_dir)
            st.success(f"已索引 {count} 个文档块")
            st.session_state["indexed"] = True


# === 延迟导入（缓存重型模块，避免每次交互重新加载）===
@st.cache_resource
def get_rag_query():
    from src.graph import query as rag_query
    return rag_query

@st.cache_resource
def get_rag_stream_query():
    from src.graph import stream_query
    return stream_query

@st.cache_resource
def get_precision_query():
    from src.graph import precision_query
    return precision_query


# === 主界面 ===
st.title("🔍 DeepRAG v2.9 知识问答")

# v2.8: 向量数据库健康检查（优先 Qdrant，回退 ChromaDB）
def check_vector_db_health():
    """检查向量数据库状态（汇总全部 collection）"""
    try:
        from src.retrieval.qdrant_retriever import list_collection_stats
        stats = list_collection_stats()
        total_docs = sum(int(s.get("points") or 0) for s in stats)
        return True, "Qdrant", len(stats), total_docs
    except Exception:
        try:
            from src.config import get_chroma_client
            client = get_chroma_client()
            client.heartbeat()
            cols = client.list_collections()
            total_docs = sum(c.count() for c in cols)
            return True, "ChromaDB", len(cols), total_docs
        except Exception:
            return False, "None", 0, 0

db_ok, db_type, db_cols, db_docs = check_vector_db_health()

# 配置状态栏
config_cols = st.columns(5)
config_cols[0].metric("LLM后端", llm_backend)
config_cols[1].metric("模型", llm_model or "N/A")
config_cols[2].metric("检索模式", mode)
_has_key = bool(
    cerebras_key or groq_key or zhipu_key or anthropic_key or openai_key or siliconflow_key
    or llm_backend in ("ollama", "lmstudio", "none")
)
config_cols[3].metric("API Key", "✅ 已配置" if _has_key else "❌ 未配置")
if db_ok:
    config_cols[4].metric(db_type, f"✅ {db_cols}集合/{db_docs}篇")
else:
    config_cols[4].metric("向量DB", "❌ 未连接")
    st.error("⚠️ 向量数据库未连接！请确保 Qdrant 本地数据目录可访问。")

st.markdown("---")

# === 6 个 Tab ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 问答", "📜 历史记录", "🗄️ 向量DB", "📊 评估报告", "🧪 测试集", "📕 错题集"])

# ========================================
# Tab 1: 💬 问答
# ========================================
with tab1:
    # 多轮会话：相关追问可借鉴前几轮答案（如 INTJ 堆栈一致性）
    if "dialog_turns" not in st.session_state:
        st.session_state["dialog_turns"] = []

    input_mode = st.radio(
        "输入模式",
        ["单次查询", "队列模式"],
        format_func=lambda x: {
            "单次查询": "💬 单次查询",
            "队列模式": "📋 队列模式（可中途追问）",
        }.get(x, x),
        horizontal=True,
    )
    use_stream = st.checkbox("流式输出", value=True)
    c_mem1, c_mem2 = st.columns([3, 1])
    with c_mem1:
        n_turns = len(st.session_state.get("dialog_turns") or [])
        if n_turns:
            st.caption(f"🧠 会话记忆：已缓存最近 {n_turns} 轮，相关追问会自动借鉴并防矛盾")
        else:
            st.caption("🧠 会话记忆：空（答完第一题后，相关第二题会自动沿用已确认事实）")
    with c_mem2:
        if st.button("清空会话记忆", key="clear_dialog_turns"):
            st.session_state["dialog_turns"] = []
            st.rerun()

    # 单次查询模式才显示主输入框
    if input_mode == "单次查询":
        question = st.text_input("请输入问题", value="INTJ的主导功能是什么？")

    # === 队列模式（v2.9.2：完整增删改查 + 勾选处理）===
    if input_mode == "队列模式":
        # 初始化队列状态
        if "q_queue" not in st.session_state:
            st.session_state["q_queue"] = []
        if "q_results" not in st.session_state:
            st.session_state["q_results"] = []
        if "q_processing" not in st.session_state:
            st.session_state["q_processing"] = False

        # 添加问题（带去重 + 相似度检测 + 快速多次点击确认）
        def _text_similarity(a: str, b: str) -> tuple:
            """文本相似度（综合：Jaccard + 子串包含）

            Returns:
                (similarity: float, reason: str)
            """
            if not a or not b:
                return 0.0, ""

            a_lower = a.lower().strip()
            b_lower = b.lower().strip()

            # 完全相同
            if a_lower == b_lower:
                return 1.0, "完全相同"

            # 子串包含检测（只要一方包含另一方就触发）
            if a_lower in b_lower or b_lower in a_lower:
                shorter = min(len(a_lower), len(b_lower))
                longer = max(len(a_lower), len(b_lower))
                ratio = shorter / longer if longer > 0 else 0
                # 短串>=3字符且占比>30%，或短串>=2字符且占比>50%
                if (shorter >= 3 and ratio >= 0.3) or (shorter >= 2 and ratio >= 0.5):
                    return 0.95, "包含关系"

            # Jaccard 字符级相似度
            set_a = set(a_lower)
            set_b = set(b_lower)
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union if union > 0 else 0.0

            return jaccard, "字符相似"

        def _find_most_similar(query: str, queue: list) -> tuple:
            """找出队列中与 query 最相似的问题"""
            best_sim = 0.0
            best_text = ""
            best_reason = ""
            for q in queue:
                q_text = q["text"] if isinstance(q, dict) else q
                sim, reason = _text_similarity(query, q_text)
                if sim > best_sim:
                    best_sim = sim
                    best_text = q_text
                    best_reason = reason
            return best_sim, best_text, best_reason

        # 初始化点击状态
        if "q_click_state" not in st.session_state:
            st.session_state["q_click_state"] = {"count": 0, "query": "", "timestamp": 0}

        col_input, col_add = st.columns([4, 1])
        with col_input:
            new_q = st.text_input("输入问题", key="queue_input", placeholder="输入问题后点击添加")
        with col_add:
            st.write("")  # 对齐
            if st.button("➕ 添加", type="primary") and new_q.strip():
                existing_texts = [q["text"] if isinstance(q, dict) else q for q in st.session_state["q_queue"]]

                # 完全重复
                if new_q.strip() in existing_texts:
                    st.warning("该问题已在队列中")
                else:
                    # 相似度检测
                    sim, sim_text, reason = _find_most_similar(new_q.strip(), st.session_state["q_queue"])

                    # 根据相似度决定需要点击次数
                    if sim >= 0.75:
                        required_clicks = 3 if sim >= 0.9 else 2
                        cs = st.session_state["q_click_state"]
                        now = time.time()

                        if cs["query"] == new_q.strip() and now - cs["timestamp"] < 1.0:
                            # 同一问题，1秒内连续点击
                            cs["count"] += 1
                            cs["timestamp"] = now

                            if cs["count"] >= required_clicks:
                                # 点击完成，添加
                                st.session_state["q_queue"].append({"text": new_q.strip(), "selected": True})
                                cs["count"] = 0
                                cs["query"] = ""
                                st.rerun()
                            else:
                                remaining = required_clicks - cs["count"]
                                st.warning(f"⚠️ 相似度 {sim:.0%}（{reason}，与「{sim_text[:20]}」），快速点击 {remaining} 次确认（1秒内）")
                        else:
                            # 新问题或超时（>1秒），重置计数
                            cs["count"] = 1
                            cs["query"] = new_q.strip()
                            cs["timestamp"] = now
                            remaining = required_clicks - 1
                            st.warning(f"⚠️ 相似度 {sim:.0%}（{reason}，与「{sim_text[:20]}」），快速点击 {remaining} 次确认（1秒内）")
                    else:
                        # 正常添加
                        st.session_state["q_queue"].append({"text": new_q.strip(), "selected": True})
                        st.session_state["q_click_state"] = {"count": 0, "query": "", "timestamp": 0}
                        st.rerun()

        n_pending = len(st.session_state["q_queue"])
        n_selected = sum(1 for q in st.session_state["q_queue"] if q.get("selected", True))

        # 操作按钮行
        col_sel_all, col_sel_none, col_del_sel, col_clear_done = st.columns(4)
        with col_sel_all:
            if st.button("☑️ 全选", disabled=(n_pending == 0)):
                for q in st.session_state["q_queue"]:
                    q["selected"] = True
                st.rerun()
        with col_sel_none:
            if st.button("☐ 全不选", disabled=(n_pending == 0)):
                for q in st.session_state["q_queue"]:
                    q["selected"] = False
                st.rerun()
        with col_del_sel:
            if st.button("🗑️ 删除选中", disabled=(n_selected == 0)):
                st.session_state["q_queue"] = [q for q in st.session_state["q_queue"] if not q.get("selected", True)]
                st.rerun()
        with col_clear_done:
            if st.button("🧹 清空已完成"):
                st.session_state["q_results"] = []
                st.rerun()

        # 处理按钮行
        col_process_sel, col_process_all = st.columns(2)
        with col_process_sel:
            if st.button(f"▶️ 处理选中 ({n_selected}个)", disabled=(n_selected == 0), type="primary"):
                st.session_state["q_process_mode"] = "selected"
                st.session_state["q_processing"] = True
                st.rerun()
        with col_process_all:
            if st.button(f"🔄 全部处理 ({n_pending}个)", disabled=(n_pending == 0)):
                st.session_state["q_process_mode"] = "all"
                st.session_state["q_processing"] = True
                st.rerun()

        # 执行处理
        if st.session_state.get("q_processing"):
            import importlib
            import src.config
            importlib.reload(src.config)

            mode_sel = st.session_state.get("q_process_mode", "all")
            if mode_sel == "selected":
                to_process = [q for q in st.session_state["q_queue"] if q.get("selected", True)]
            else:
                to_process = list(st.session_state["q_queue"])

            if not to_process:
                st.warning("没有需要处理的问题")
                st.session_state["q_processing"] = False
                st.rerun()

            if mode == "precision":
                rag_fn = get_precision_query()
            else:
                rag_fn = get_rag_query()

            total = len(to_process)
            progress = st.progress(0, f"处理中... 0/{total}")

            for i, q_item in enumerate(to_process):
                q = q_item["text"] if isinstance(q_item, dict) else q_item
                t0 = time.time()
                with st.spinner(f"[{i+1}/{total}] {q[:50]}..."):
                    try:
                        if mode == "precision":
                            _pc = st.session_state.get("precision_config", "cross_fast")
                            _pmodels = {"cross_fast": ("THUDM/GLM-Z1-9B-0414","glm-4-flash","glm-4-flash"),
                                        "flash_dual": ("glm-4-flash","glm-4-flash","glm-4-flash"),
                                        "cross": ("THUDM/GLM-Z1-9B-0414","glm-4-flash","glm-4-flash"),
                                        "z1_dual": ("THUDM/GLM-Z1-9B-0414","THUDM/GLM-Z1-9B-0414","glm-4-flash")}
                            _ma,_mb,_mc = _pmodels.get(_pc, _pmodels["cross_fast"])
                            _sc = st.session_state.get("precision_strategy", "socratic_concise")
                            _smap = {"socratic_concise": ("socratic","concise"),
                                     "direct_analytical": ("direct","analytical"),
                                     "cot_concise": ("chain_of_thought","concise"),
                                     "socratic_analytical": ("socratic","analytical")}
                            _sa,_sb = _smap.get(_sc, _smap["socratic_concise"])
                            _fast = st.session_state.get("precision_fast", True)
                            result = rag_fn(q, collection_name=collection, max_retries=max_retries,
                                           model_a=_ma, model_b=_mb, compare_model=_mc,
                                           strategy_a=_sa, strategy_b=_sb, fast_mode=_fast)
                        else:
                            result = rag_fn(
                                q,
                                collection_name=collection,
                                max_retries=max_retries,
                                mode=mode,
                                dialog_turns=st.session_state.get("dialog_turns") or [],
                            )
                    except Exception as e:
                        result = {"question": q, "answer": f"查询失败：{e}", "error": str(e)}
                result["_elapsed"] = time.time() - t0
                st.session_state["q_results"].append((q, result))
                # 写入会话记忆，供后续队列问题借鉴
                _ans = (result or {}).get("answer") or ""
                if _ans and "查询失败" not in _ans:
                    turns = list(st.session_state.get("dialog_turns") or [])
                    turns.append({"q": q, "a": _ans})
                    st.session_state["dialog_turns"] = turns[-5:]

                # 从队列移除已处理的
                st.session_state["q_queue"] = [x for x in st.session_state["q_queue"]
                                                if (x["text"] if isinstance(x, dict) else x) != q]

                # 保存到历史
                try:
                    from src.ui.history_manager import save_qa
                    save_qa(q, result.get("answer", ""), mode,
                            metrics=result, citations=result.get("citations", []))
                except Exception:
                    pass

                progress.progress((i + 1) / total, f"处理中... {i+1}/{total}")

            st.session_state["q_processing"] = False
            st.rerun()

        # 显示待处理队列（带勾选框和删除按钮）
        if st.session_state["q_queue"]:
            st.markdown(f"### 📋 待处理队列 ({n_pending}个，已选{n_selected}个)")
            for i, q_item in enumerate(st.session_state["q_queue"]):
                q_text = q_item["text"] if isinstance(q_item, dict) else q_item
                col_check, col_text, col_edit, col_del = st.columns([0.5, 3, 0.5, 0.5])
                with col_check:
                    new_sel = st.checkbox("", value=q_item.get("selected", True), key=f"sel_{i}", label_visibility="collapsed")
                    if new_sel != q_item.get("selected", True):
                        q_item["selected"] = new_sel
                        st.rerun()
                with col_text:
                    st.text(f"{i+1}. {q_text}")
                with col_edit:
                    if st.button("✏️", key=f"edit_{i}"):
                        st.session_state[f"editing_{i}"] = True
                        st.rerun()
                with col_del:
                    if st.button("❌", key=f"del_{i}"):
                        st.session_state["q_queue"].pop(i)
                        st.rerun()

                # 编辑模式
                if st.session_state.get(f"editing_{i}"):
                    new_text = st.text_input("编辑问题", value=q_text, key=f"edit_input_{i}")
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("💾 保存", key=f"save_{i}"):
                            st.session_state["q_queue"][i]["text"] = new_text
                            st.session_state[f"editing_{i}"] = False
                            st.rerun()
                    with col_cancel:
                        if st.button("↩️ 取消", key=f"cancel_{i}"):
                            st.session_state[f"editing_{i}"] = False
                            st.rerun()

        # 显示已完成结果
        if st.session_state["q_results"]:
            st.markdown(f"### ✅ 已完成 ({len(st.session_state['q_results'])}个)")
            for i, (q, result) in enumerate(st.session_state["q_results"]):
                answer = result.get("answer", "") if result else "（无结果）"
                r_elapsed = result.get("_elapsed", 0) if result else 0
                with st.expander(f"Q{i+1}: {q[:60]} ({r_elapsed:.1f}s)", expanded=(i == len(st.session_state["q_results"]) - 1)):
                    render_answer(answer)

                    if result:
                        m1, m2, m3, m4 = st.columns(4)
                        r_answer = result.get("answer", "")
                        r_relevant = result.get("relevant_count", 0)
                        r_halluc = result.get("hallucination_score", 0)
                        if r_relevant == 0 and (len(r_answer) < 20 or "未找到" in r_answer):
                            r_cred = 0.0
                        else:
                            r_cred = 1 - r_halluc
                        m1.metric("可信度", f"{r_cred:.0%}")
                        m2.metric("引用数", len(result.get("citations", [])))
                        m3.metric("相关文档", r_relevant)
                        m4.metric("耗时", f"{r_elapsed:.1f}s")

                        # v2.9.1: 用户反馈按钮
                        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 3])
                        if fb_col1.button("👍 有帮助", key=f"fb_up_{i}"):
                            from src.evaluation.llm_judge import record_feedback
                            record_feedback(q, answer, "up", result.get("mode", "enhanced"))
                            st.success("感谢反馈！")
                        if fb_col2.button("👎 需改进", key=f"fb_down_{i}"):
                            from src.evaluation.llm_judge import record_feedback
                            record_feedback(q, answer, "down", result.get("mode", "enhanced"))
                            st.info("感谢反馈，我们会持续改进")

                        # 思考过程：执行历史
                        history = result.get("history", [])
                        if history:
                            with st.expander(f"🧠 思考过程（{len(history)}步）"):
                                for h in history:
                                    st.text(f"  → {map_step_name(h)}")

                        # 引用详情
                        citations = result.get("citations", [])

        st.stop()

    # === 单次查询模式 ===
    if st.button("🚀 查询", type="primary"):
        # 确保索引（跳过已有数据的集合）
        indexer = get_indexer_cached(collection)
        if not indexer.is_already_indexed():
            if doc_dir and os.path.isdir(doc_dir):
                with st.spinner("首次索引中..."):
                    count = indexer.index_directory(doc_dir)
                if count > 0:
                    st.info(f"索引了 {count} 个文档块")

        # 重新加载配置
        import importlib
        import src.config
        importlib.reload(src.config)

        start_time = time.time()
        final_state = None
        collected_answer = ""

        _dialog_turns = list(st.session_state.get("dialog_turns") or [])
        try:
            if use_stream and mode not in ("agentic_react", "precision"):
                # === 流式输出 ===
                stream_fn = get_rag_stream_query()
                answer_placeholder = st.empty()

                with st.spinner("检索中..."):
                    for chunk in stream_fn(
                        question,
                        collection_name=collection,
                        max_retries=max_retries,
                        mode=mode,
                        dialog_turns=_dialog_turns,
                    ):
                        if chunk["type"] == "token":
                            collected_answer += chunk["content"]
                            answer_placeholder.markdown(collected_answer + "▌")
                        elif chunk["type"] == "metadata":
                            final_state = chunk["state"]

                answer_placeholder.empty()
                render_answer(collected_answer)
            elif mode == "precision":
                # === 精准模式（非流式，双Agent并行）===
                _pc = st.session_state.get("precision_config", "cross_fast")
                _precision_models = {
                    "cross_fast": ("THUDM/GLM-Z1-9B-0414", "glm-4-flash", "glm-4-flash"),
                    "flash_dual": ("glm-4-flash", "glm-4-flash", "glm-4-flash"),
                    "cross": ("THUDM/GLM-Z1-9B-0414", "glm-4-flash", "glm-4-flash"),
                    "z1_dual": ("THUDM/GLM-Z1-9B-0414", "THUDM/GLM-Z1-9B-0414", "glm-4-flash"),
                }
                _ma, _mb, _mc = _precision_models.get(_pc, _precision_models["cross_fast"])

                _sc = st.session_state.get("precision_strategy", "socratic_concise")
                _strategy_map = {
                    "socratic_concise": ("socratic", "concise"),
                    "direct_analytical": ("direct", "analytical"),
                    "cot_concise": ("chain_of_thought", "concise"),
                    "socratic_analytical": ("socratic", "analytical"),
                }
                _sa, _sb = _strategy_map.get(_sc, _strategy_map["socratic_concise"])
                _fast = st.session_state.get("precision_fast", True)

                with st.spinner("🎯 双Agent并行生成中..."):
                    precision_fn = get_precision_query()
                    final_state = precision_fn(question, collection_name=collection,
                                               max_retries=max_retries,
                                               model_a=_ma, model_b=_mb, compare_model=_mc,
                                               strategy_a=_sa, strategy_b=_sb,
                                               fast_mode=_fast)
                collected_answer = final_state.get("answer", "")
                render_answer(collected_answer)
            else:
                # === 非流式输出 ===
                # 错库自动纠正（与 graph 内路由一致，UI 先提示）
                try:
                    from src.retrieval.collection_router import collection_conflicts_with_query
                    _c, _msg, _sug = collection_conflicts_with_query(question, collection)
                    if _c and _sug and _sug != collection:
                        st.warning(f"🔀 已自动换库：**{collection}** → **{_sug}**（{_msg}）")
                        collection = _sug
                except Exception:
                    pass
                with st.spinner(f"{'🤖 ReAct Agent' if mode == 'agentic_react' else '🔄 Pipeline'} 运行中..."):
                    rag_query_fn = get_rag_query()
                    final_state = rag_query_fn(
                        question,
                        collection_name=collection,
                        max_retries=max_retries,
                        mode=mode,
                        dialog_turns=_dialog_turns,
                    )
                collected_answer = final_state.get("answer", "")
                if final_state.get("routed_collection") and final_state.get("routed_collection") != collection:
                    st.info(f"实际使用知识库：`{final_state.get('routed_collection')}`")
                render_answer(collected_answer)

        except Exception as e:
            st.error(f"查询失败: {str(e)[:300]}")
            import traceback
            with st.expander("详细错误"):
                st.code(traceback.format_exc())
            st.stop()

        elapsed = time.time() - start_time

        # 保存到历史 + 会话记忆（相关追问用）
        if final_state and collected_answer:
            try:
                from src.ui.history_manager import save_qa
                save_qa(question, collected_answer, mode,
                        metrics=final_state, citations=final_state.get("citations", []))
            except Exception as e:
                st.warning(f"历史保存失败: {e}")
            turns = list(st.session_state.get("dialog_turns") or [])
            turns.append({"q": question, "a": collected_answer})
            st.session_state["dialog_turns"] = turns[-5:]
            if any(h for h in (final_state.get("history") or []) if "prior_context" in str(h) or "dialog" in str(h)):
                st.caption("✅ 本轮已注入相关前轮上下文（一致性约束生效）")

        # 指标
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)

        # 可信度：拒答/无引用/无相关文档绝不能显示 100%
        answer_text = final_state.get("answer", "") or ""
        relevant_count = int(final_state.get("relevant_count") or 0)
        halluc_score = float(final_state.get("hallucination_score") or 0)
        citations = final_state.get("citations") or []
        no_knowledge = bool(final_state.get("no_knowledge"))
        refuse_marks = (
            "未找到", "无法回答", "未明确提及", "无关", "无法提供",
            "不能回答", "没有相关", "未检索到", "知识库与外部", "未提及",
            "答非所问", "无法基于",
        )
        is_refuse = no_knowledge or any(m in answer_text for m in refuse_marks)

        fact_ok = final_state.get("fact_check_passed", True)
        if is_refuse or relevant_count <= 0:
            credibility = 0.0
            cred_delta = "无可靠依据"
            cred_delta_color = "inverse"
        elif fact_ok is False:
            # 事实校验未通过：可信度上限压到 50%
            credibility = min(0.50, max(0.0, 1.0 - halluc_score))
            cred_delta = "事实校验未通过"
            cred_delta_color = "inverse"
        elif len(citations) == 0:
            # 有「相关」文档但模型没落引用 → 最多 30%
            credibility = min(0.30, max(0.0, 1.0 - halluc_score))
            cred_delta = "无引用·低可信"
            cred_delta_color = "inverse"
        else:
            credibility = max(0.0, min(1.0, 1.0 - halluc_score))
            # 引用很少时再打折
            if len(citations) < 2:
                credibility = min(credibility, 0.75)
            cred_delta = "越高越好"
            cred_delta_color = "normal"

        col1.metric("可信度", f"{credibility:.0%}", delta=cred_delta,
                     delta_color=cred_delta_color)
        col2.metric("引用数", len(citations))
        col3.metric("相关文档", relevant_count)
        col4.metric("检索轮次", final_state.get("retrieval_round", 0) or final_state.get("retry_count", 0))
        col5.metric("耗时", f"{elapsed:.1f}s")
        if elapsed > 15 and mode == "agentic_react":
            st.info("💡 本次偏慢主要因 **Agentic ReAct 多轮 LLM**。下次侧栏改成 **Enhanced RAG**，并选对知识库，通常可压到 10s 内（视模型而定）。")

        # ReAct Agent 决策信息
        if mode == "agentic_react":
            used_tools = final_state.get("used_tools", [])
            if used_tools:
                st.markdown("---")
                st.subheader("🤖 Agent 决策轨迹")
                st.markdown(f"**使用工具**: {' → '.join(used_tools)}")
                reason = final_state.get("agent_reason", "")
                if reason:
                    st.caption(f"决策理由: {reason}")

        # 精准模式双Agent对比结果
        if mode == "precision":
            verdict = final_state.get("verdict", "")
            answer_a = final_state.get("answer_a", "")
            answer_b = final_state.get("answer_b", "")
            re_searched = final_state.get("re_searched", False)
            conflict_points = final_state.get("conflict_points", [])
            recommendation = final_state.get("recommendation", "")
            show_both = final_state.get("show_both", False)
            strategy_a = final_state.get("strategy_a", "direct")
            strategy_b = final_state.get("strategy_b", "analytical")
            elapsed_a = final_state.get("elapsed_a", 0)
            elapsed_b = final_state.get("elapsed_b", 0)

            st.markdown("---")
            st.subheader("🎯 双Agent对比结果")

            # 对比结论
            verdict_colors = {"agree": "✅", "conflict": "❌", "partial": "⚠️", "skipped": "⏭️"}
            verdict_labels = {"agree": "一致", "conflict": "矛盾", "partial": "部分一致", "skipped": "跳过"}
            verdict_icon = verdict_colors.get(verdict, "❓")
            verdict_label = verdict_labels.get(verdict, verdict)

            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            vcol1.metric("对比结果", f"{verdict_icon} {verdict_label}")
            vcol2.metric("策略", recommendation)
            vcol3.metric("重新搜索", "是" if re_searched else "否")
            vcol4.metric("双答案展示", "是" if show_both else "否")

            if conflict_points:
                st.warning(f"**检测到差异**: {' / '.join(conflict_points)}")

            # 双答案并排展示（矛盾时高亮显示让用户分辨）
            if show_both:
                st.info("💡 两个Agent给出了不同答案，请自行分辨哪个更准确：")

            strategy_labels = {
                "direct": "直接策略", "analytical": "分析策略",
                "socratic": "苏格拉底策略", "chain_of_thought": "思维链策略",
                "concise": "精简策略",
            }
            sa_label = strategy_labels.get(strategy_a, strategy_a)
            sb_label = strategy_labels.get(strategy_b, strategy_b)

            col_a, col_b = st.columns(2)
            with col_a:
                exp_label = f"📝 Agent A（{sa_label}）| {elapsed_a}s | {len(answer_a)}字"
                with st.expander(exp_label, expanded=show_both):
                    if show_both:
                        st.markdown(f"**⚠️ 此答案可能与Agent B存在差异**")
                    st.markdown(answer_a)
            with col_b:
                exp_label = f"📝 Agent B（{sb_label}）| {elapsed_b}s | {len(answer_b)}字"
                with st.expander(exp_label, expanded=show_both):
                    if show_both:
                        st.markdown(f"**⚠️ 此答案可能与Agent A存在差异**")
                    st.markdown(answer_b)

        # 引用详情
        citations = final_state.get("citations", [])
        if citations:
            st.markdown("---")
            st.subheader("📎 引用来源")
            for c in citations:
                source = c.get("source", "?")
                page = c.get("page", "?")
                text = c.get("text", c.get("content", ""))[:200]
                st.caption(f"[{source} p.{page}] {text}")

        # 执行历史（步骤名映射为中文）
        history = final_state.get("history", [])
        if history:
            with st.expander(f"📜 执行历史（{len(history)}步）"):
                for h in history:
                    st.text(f"  → {map_step_name(h)}")

        # 事实校验
        unsupported = final_state.get("unsupported_claims", [])
        if unsupported:
            st.markdown("---")
            st.error(f"⚠️ 未被文档支持的断言: {unsupported}")

        # 冲突
        conflicts = final_state.get("conflicts", [])
        if conflicts:
            st.subheader("⚠️ 多源冲突")
            for cf in conflicts:
                st.warning(f"**{cf['topic']}**: {cf.get('resolution', '')}")

    # 架构说明
    st.markdown("---")
    with st.expander("🏗️ 架构说明"):
        arch_mode = st.radio("查看架构", ["Enhanced Pipeline", "Agentic ReAct"], key="arch_radio")
        if arch_mode == "Enhanced Pipeline":
            st.markdown("""
            ```
            用户提问 → [1.查询分析] → [2.混合检索] → [3.文档评分(CRAG)]
                → 有relevant → [5.生成] → [6.事实校验(Self-RAG)] → [7.冲突检测] → 输出
                → 无relevant → [4.查询改写] → 回到[2]（纠错循环）
                → 重试耗尽 → [4b.Web兜底] → [5.生成]
            ```
            **技术决策**：CRAG纠正检索质量，Self-RAG纠正生成质量，RRF融合不需归一化分数
            """)
        else:
            st.markdown("""
            ```
            用户提问 → ┌→ [Agent决策] → 选择工具(vector/exact/graph/web/generate)
                       │    ↓
                       │  [执行检索] → 结果加入文档列表
                       └← 返回Agent决策（ReAct循环，最多N轮）
                            ↓
                        [生成答案] → [事实校验] → 输出
            ```
            **v2.4新增**：LLM自主决策+4工具全部可用+ReAct循环
            """)


# ========================================
# Tab 2: 📜 历史记录
# ========================================
with tab2:
    st.header("📜 问答历史")

    from src.ui.history_manager import load_history, search_history, clear_history, get_history_stats

    # 统计信息
    stats = get_history_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("总记录数", stats.get("total", 0))
    col2.metric("检索模式", ", ".join(stats.get("modes", {}).keys()) or "N/A")
    avg_hallu = stats.get("avg_metrics", {}).get("hallucination_score", 0)
    # v2.6修复：平均可信度也考虑无答案记录
    if isinstance(avg_hallu, (int, float)):
        avg_cred = max(0, 1 - avg_hallu)
    else:
        avg_cred = None
    col3.metric("平均可信度", f"{avg_cred:.0%}" if avg_cred is not None else "N/A")

    st.markdown("---")

    # 搜索栏
    search_col, clear_col = st.columns([4, 1])
    with search_col:
        keyword = st.text_input("🔍 搜索历史记录", placeholder="输入关键词...")
    with clear_col:
        st.write("")  # 占位对齐
        confirm_clear = st.checkbox("确认操作", key="confirm_clear_hist")
        if st.button("🗑️ 清空历史", disabled=not confirm_clear):
            clear_history()
            st.success("历史记录已清空")
            st.rerun()

    # 历史列表
    if keyword:
        records = search_history(keyword)
        st.caption(f"找到 {len(records)} 条匹配记录")
    else:
        records = load_history(limit=50)

    if not records:
        st.info("暂无历史记录，在「问答」Tab中提问后会自动保存")
    else:
        for record in reversed(records):
            ts = record.get("timestamp", "")[:19]
            q = record.get("question", "")[:60]
            with st.expander(f"[{ts}] {q}..."):
                st.markdown(f"**问题**: {record.get('question', '')}")
                st.markdown(f"**回答**: {record.get('answer', '')}")
                st.caption(f"模式: {record.get('mode', 'unknown')}")

                metrics = record.get("metrics", {})
                if metrics:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    # v2.6修复：历史记录可信度计算
                    hist_relevant = metrics.get("relevant_count", 0)
                    hist_halluc = metrics.get("hallucination_score", 0)
                    hist_answer = record.get("answer", "")
                    if hist_relevant == 0 and (len(hist_answer) < 20 or "未找到" in hist_answer):
                        hist_cred = 0.0
                    else:
                        hist_cred = 1 - hist_halluc
                    mc1.metric("可信度", f"{hist_cred:.0%}")
                    mc2.metric("引用数", len(metrics.get("citations", [])))
                    mc3.metric("相关文档", metrics.get("relevant_count", 0))
                    mc4.metric("事实校验", "✅" if metrics.get("fact_check_passed", True) else "❌")


# ========================================
# Tab 3: 🗄️ 向量DB管理
# ========================================
with tab3:
    st.header("🗄️ 向量数据库管理")

    from src.ui.vector_db_manager import (
        get_db_info, list_collections, get_collection_docs,
        delete_collection, delete_document
    )

    # 数据库概要
    db_info = get_db_info()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("数据库路径", db_info.get("db_path", "unknown").split("/")[-1] if db_info.get("db_path") else "N/A")
    col2.metric("Collection数", db_info.get("total_collections", 0))
    col3.metric("文档总数", db_info.get("total_docs", 0))
    col4.metric("DB大小", f"{db_info.get('db_size_mb', 0)} MB")

    st.markdown("---")

    # Collection 列表
    st.subheader("📋 Collections")
    collections = list_collections()

    if not collections:
        st.info("暂无Collection，请在侧边栏索引文档")
    else:
        for col_info in collections:
            name = col_info.get("name", "unknown")
            count = col_info.get("count", 0)
            with st.expander(f"📦 {name} ({count} 文档)"):
                if col_info.get("sample_doc"):
                    st.text(f"样本文档: {col_info['sample_doc'][:100]}...")

                # 文档预览
                docs = get_collection_docs(name, limit=10)
                if docs:
                    st.markdown("**文档预览:**")
                    for doc in docs:
                        doc_text = doc.get("document", "")[:100]
                        st.text(f"  ID: {doc.get('id', '?')} | {doc_text}...")

                # 删除操作
                confirm_del = st.checkbox("确认删除", key=f"confirm_del_{name}")
                if st.button(f"🗑️ 删除Collection", key=f"del_{name}", disabled=not confirm_del):
                    if delete_collection(name):
                        st.success(f"已删除 {name}")
                        st.rerun()
                    else:
                        st.error("删除失败")


# ========================================
# Tab 4: 📊 评估报告
# ========================================
with tab4:
    st.header("📊 评估报告")

    from src.ui.evaluation_engine import EvaluationEngine, METRIC_WEIGHTS, METRIC_GROUPS

    engine = EvaluationEngine()

    # 指标权重展示
    st.subheader("📐 22指标权重体系")
    weights_df = pd.DataFrame(
        [(k, v) for k, v in METRIC_WEIGHTS.items()],
        columns=["指标", "权重(%)"]
    )
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 单次评估
    st.subheader("🔬 单次查询评估")
    eval_question = st.text_input("输入评估问题", value="INTJ的主导功能是什么？", key="eval_q")

    if st.button("运行评估"):
        # 确保索引（跳过已有数据的集合）
        indexer = get_indexer_cached(collection)
        if not indexer.is_already_indexed():
            if doc_dir and os.path.isdir(doc_dir):
                with st.spinner("首次索引中..."):
                    indexer.index_directory(doc_dir)

        with st.spinner("执行查询并评估..."):
            import importlib
            import src.config
            importlib.reload(src.config)

            start = time.time()
            try:
                rag_query_fn = get_rag_query()
                state = rag_query_fn(eval_question, collection_name=collection, mode=mode)
                elapsed = time.time() - start

                contexts = [d.get("content", d.get("text", "")) for d in state.get("graded_docs", [])]
                eval_result = engine.evaluate(
                    state=state,
                    response_time=elapsed,
                    question=eval_question,
                    answer=state.get("answer", ""),
                    contexts=contexts,
                )
            except Exception as e:
                st.error(f"评估失败: {str(e)[:300]}")
                st.stop()

        # 总分
        st.metric("综合评分", f"{eval_result['overall_score']}/100",
                   delta=f"{eval_result['metric_count']}/{eval_result['total_metrics']} 指标可用")

        # 分组得分雷达图
        group_scores = eval_result.get("group_scores", {})
        if group_scores:
            import plotly.graph_objects as go

            labels = list(group_scores.keys())
            values = list(group_scores.values())

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]] if values else [0],
                theta=labels + [labels[0]] if labels else ["N/A"],
                fill='toself',
                name='DeepRAG'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="分组评分雷达图",
                height=400,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # 详细指标
        st.markdown("**详细指标:**")
        metrics_dict = eval_result.get("metrics", {})
        if metrics_dict:
            metrics_df = pd.DataFrame(
                [(k, round(v * 100, 1)) for k, v in metrics_dict.items()],
                columns=["指标", "得分(0-100)"]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # 历史趋势线（可信度趋势）
    st.markdown("---")
    st.subheader("📈 历史评估趋势")
    from src.ui.history_manager import load_history
    history = load_history(limit=100)
    if history:
        scores = [r.get("metrics", {}).get("hallucination_score", 0) for r in history if r.get("metrics")]
        if scores:
            import plotly.graph_objects as go
            fig_line = go.Figure(data=go.Scatter(
                y=[1 - s for s in scores],
                mode='lines+markers',
                name='可信度',
                line=dict(color='#1f77b4'),
            ))
            fig_line.update_layout(
                title="可信度趋势",
                xaxis_title="查询序号",
                yaxis_title="可信度",
                height=300,
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("暂无评估数据")
    else:
        st.info("暂无历史记录")

    # v2.9.1: LLM-as-Judge 评测
    st.markdown("---")
    st.subheader("⚖️ LLM-as-Judge 5维度评测")

    judge_col1, judge_col2 = st.columns([2, 1])

    with judge_col1:
        judge_question = st.text_input(
            "输入评测问题",
            value="INTJ的主导功能是什么？",
            key="judge_q",
            help="LLM裁判将从5个维度评估答案质量",
        )

    with judge_col2:
        use_golden = st.checkbox("使用Golden Test Set", value=False, key="use_golden_judge")

    if st.button("🚀 运行LLM-as-Judge", key="run_judge"):
        if use_golden:
            # 批量评估Golden Test Set
            with st.spinner("正在批量评估Golden Test Set（可能需要几分钟）..."):
                try:
                    import json
                    from pathlib import Path
                    golden_path = Path("evaluation/golden_test_set.json")
                    if not golden_path.exists():
                        st.error("Golden Test Set 不存在，请先创建")
                        st.stop()

                    with open(golden_path, "r", encoding="utf-8") as f:
                        test_cases = json.load(f)

                    # 运行RAG获取答案
                    from src.evaluation.llm_judge import LLMJudge
                    judge = LLMJudge()

                    # 对前10条做评估（避免耗时过长）
                    sample_cases = test_cases[:10]
                    rag_query_fn = get_rag_query()

                    for case in sample_cases:
                        state = rag_query_fn(case["query"], collection_name=collection, mode=mode)
                        case["actual_answer"] = state.get("answer", "")
                        case["context"] = " ".join(
                            d.get("content", "")[:200] for d in state.get("graded_docs", [])[:2]
                        )

                    results = judge.batch_evaluate(sample_cases)

                    # 展示结果
                    st.metric("平均总分", f"{results['avg_overall']}/10")

                    # 5维度评分
                    dim_scores = results.get("avg_by_dimension", {})
                    if dim_scores:
                        dim_df = pd.DataFrame(
                            [(k, v) for k, v in dim_scores.items()],
                            columns=["维度", "平均分(0-10)"]
                        )
                        st.dataframe(dim_df, use_container_width=True, hide_index=True)

                    # 按类别统计
                    cat_scores = results.get("avg_by_category", {})
                    if cat_scores:
                        st.markdown("**按类别统计:**")
                        cat_df = pd.DataFrame(
                            [(k, v) for k, v in cat_scores.items()],
                            columns=["类别", "平均分"]
                        )
                        st.dataframe(cat_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"批量评估失败: {str(e)[:300]}")

        else:
            # 单问题评估
            with st.spinner("LLM裁判评估中..."):
                try:
                    rag_query_fn = get_rag_query()
                    state = rag_query_fn(judge_question, collection_name=collection, mode=mode)
                    answer = state.get("answer", "")
                    context = " ".join(
                        d.get("content", "")[:200] for d in state.get("graded_docs", [])[:2]
                    )

                    from src.evaluation.llm_judge import LLMJudge
                    judge = LLMJudge()
                    result = judge.evaluate(
                        question=judge_question,
                        answer=answer,
                        reference="",
                        context=context,
                    )

                    # 展示答案
                    render_answer(answer)

                    # 展示5维度评分
                    st.metric("总分", f"{result['overall']}/10")

                    scores = result["scores"]
                    score_cols = st.columns(5)
                    dim_names = {
                        "relevancy": "切题度",
                        "faithfulness": "忠实度",
                        "completeness": "完整度",
                        "conciseness": "简洁度",
                        "citation_accuracy": "引用准确度",
                    }
                    for i, (dim, score) in enumerate(scores.items()):
                        score_cols[i].metric(dim_names.get(dim, dim), f"{score}/10")

                except Exception as e:
                    st.error(f"评估失败: {str(e)[:300]}")

    # v2.9.1: 用户反馈统计
    st.markdown("---")
    st.subheader("💬 用户反馈统计")

    try:
        from src.evaluation.llm_judge import get_feedback_stats
        stats = get_feedback_stats()

        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
        fb_col1.metric("总反馈数", stats["total"])
        fb_col2.metric("👍 好评", stats["up"])
        fb_col3.metric("👎 差评", stats["down"])
        fb_col4.metric("满意度", f"{stats['satisfaction_rate']}%")

    except Exception:
        st.info("反馈统计功能初始化中...")


# ========================================
# Tab 5: 🧪 测试集
# ========================================
with tab5:
    st.header("🧪 测试集管理")

    # 构建测试集
    st.subheader("📋 构建测试集")
    st.markdown("从 xlsx 文件构建测试集，80/20划分训练/验证集")

    xlsx_path = st.text_input("xlsx文件路径", value="rag_evaluation_template.xlsx")
    sample_size = st.number_input("采样数量", value=100, min_value=1, max_value=4000)

    if st.button("🔨 构建测试集"):
        from scripts.build_test_set import build_test_set
        try:
            with st.spinner("构建中..."):
                result = build_test_set(xlsx_path, sample_size=int(sample_size))
            st.success(f"构建完成: 训练集{result['train_count']}条, 验证集{result['val_count']}条, 总计{result['total']}条")
        except Exception as e:
            st.error(f"构建失败: {str(e)[:300]}")

    st.markdown("---")

    # 运行测试
    st.subheader("🏃 运行准确率测试")
    test_mode = st.selectbox("测试模式", ["enhanced", "agentic_react"])
    test_subset = st.selectbox("数据集", ["validation", "train"])
    test_limit = st.number_input("测试数量", value=20, min_value=1, max_value=1000, key="test_limit")

    if st.button("🚀 开始测试"):
        from tests.test_accuracy import AccuracyTester
        tester = AccuracyTester()

        # 确保索引（跳过已有数据的集合）
        indexer = get_indexer_cached(collection)
        if not indexer.is_already_indexed():
            if doc_dir and os.path.isdir(doc_dir):
                with st.spinner("首次索引中..."):
                    indexer.index_directory(doc_dir)

        with st.spinner(f"运行{test_limit}条测试..."):
            try:
                result = tester.run_test_set(
                    mode=test_mode,
                    subset=test_subset,
                    limit=int(test_limit),
                )
            except Exception as e:
                st.error(f"测试失败: {str(e)[:300]}")
                st.stop()

        if "error" in result:
            st.error(result["error"])
        else:
            # 结果展示
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总数", result["total"])
            col2.metric("通过", result["passed"])
            col3.metric("准确率", f"{result['accuracy']}%")
            col4.metric("平均耗时", f"{result['avg_time']}s")

            # 失败案例
            failures = result.get("failures", [])
            if failures:
                st.markdown("---")
                st.subheader(f"❌ 失败案例 ({len(failures)} 个)")
                for f in failures[:10]:
                    q = f.get("question", "")[:60]
                    sim = f.get("similarity", "N/A")
                    with st.expander(f"Q: {q}... (相似度: {sim})"):
                        st.markdown(f"**期望答案**: {f.get('expected', 'N/A')[:200]}")
                        st.markdown(f"**实际答案**: {f.get('actual', 'N/A')[:200]}")
            else:
                st.success("🎉 全部通过！")


# ========================================
# Tab 6: 📕 错题集
# ========================================
with tab6:
    st.header("📕 错题集")
    st.caption("记录低质量回答，按错误类型分类，支持标记修正")

    from src.agents.error_book import ErrorBook

    error_book = ErrorBook()
    records = error_book.records

    # 错误类型分布统计
    st.subheader("📊 错误类型分布")
    type_counts = {
        "knowledge_gap": 0,
        "grading_error": 0,
        "hallucination": 0,
        "fact_check_fail": 0,
    }
    for r in records:
        etype = r.get("error_type", "unknown")
        if etype in type_counts:
            type_counts[etype] += 1

    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("知识库无匹配", type_counts["knowledge_gap"])
    ec2.metric("文档评分失败", type_counts["grading_error"])
    ec3.metric("幻觉", type_counts["hallucination"])
    ec4.metric("事实校验失败", type_counts["fact_check_fail"])

    st.markdown("---")

    # 错题列表
    st.subheader(f"📋 错题记录（共 {len(records)} 条）")

    if not records:
        st.info("暂无错题记录。当系统产生低质量回答时会自动记录到此。")
    else:
        for rec in reversed(records):
            rid = rec.get("id", "?")
            ts = rec.get("timestamp", "")[:19]
            q = rec.get("question", "")[:60]
            etype = rec.get("error_type", "unknown")
            corrected = rec.get("corrected", False)
            status_icon = "✅" if corrected else "❌"

            type_label = {
                "knowledge_gap": "知识库无匹配",
                "grading_error": "文档评分失败",
                "hallucination": "幻觉",
                "fact_check_fail": "事实校验失败",
                "unknown": "未知",
            }.get(etype, etype)

            with st.expander(f"{status_icon} [{ts}] {q}... （{type_label}）"):
                st.markdown(f"**问题**: {rec.get('question', '')}")
                st.markdown(f"**回答**: {rec.get('answer', '')[:500]}")
                st.caption(f"错误类型: {type_label} | ID: {rid}")

                metrics = rec.get("metrics", {})
                if metrics:
                    em1, em2, em3, em4 = st.columns(4)
                    em1.metric("检索文档数", metrics.get("retrieved_count", 0))
                    em2.metric("相关文档数", metrics.get("relevant_count", 0))
                    # v2.6修复：错题集可信度计算
                    err_relevant = metrics.get("relevant_count", 0)
                    err_halluc = metrics.get("hallucination_score", 0)
                    if err_relevant == 0:
                        err_cred = 0.0
                    else:
                        err_cred = 1 - err_halluc
                    em3.metric("可信度", f"{err_cred:.0%}")
                    em4.metric("事实校验", "✅" if metrics.get("fact_check_passed", True) else "❌")

                if corrected:
                    st.success(f"**已修正**: {rec.get('correction', '')}")
                else:
                    with st.form(f"correct_form_{rid}"):
                        correction_text = st.text_area(
                            "输入修正内容（正确回答或建议）",
                            key=f"correction_input_{rid}",
                            height=100,
                        )
                        if st.form_submit_button("标记为已修正"):
                            if correction_text.strip():
                                error_book.mark_corrected(rid, correction_text.strip())
                                st.success("已标记为修正")
                                st.rerun()
                            else:
                                st.warning("请输入修正内容")

    st.markdown("---")
    st.caption("错题集文件: data/error_book.json | 修正提示会在后续查询中自动应用")
