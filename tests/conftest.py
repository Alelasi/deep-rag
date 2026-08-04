"""Pytest 全局配置：CI 环境自动跳过依赖外部服务的测试 + 测试金字塔分层标记

在 GitHub Actions 中，无 ChromaDB/Qdrant/Ollama/LM Studio 等外部服务，
这些测试会无限等待连接，导致 CI 挂死。本文件在 collection 阶段自动跳过它们。

测试金字塔分层：
  L1 — 单元测试（无外部依赖，纯逻辑验证）
  L2 — 集成测试（需要外部服务：Qdrant/Ollama/API）
  L3 — 端到端测试（完整 RAG 管道）
  L4 — 性能基准（响应时间、吞吐量、召回率）
"""
import os
import pytest

# pyarrow 预热：规避 sklearn→pandas→pyarrow 在特定加载顺序下的 Windows access violation。
# 在 pytest 启动早期加载可让其 C 扩展在干净环境下初始化，避免后续崩溃。
try:
    import pyarrow  # noqa: F401
except Exception:
    pass

# CI 环境标记
IN_CI = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"


def pytest_configure(config):
    """注册测试金字塔分层标记"""
    config.addinivalue_line("markers", "L1: 单元测试（无外部依赖）")
    config.addinivalue_line("markers", "L2: 集成测试（需要外部服务）")
    config.addinivalue_line("markers", "L3: 端到端测试（完整RAG管道）")
    config.addinivalue_line("markers", "L4: 性能基准（响应时间/吞吐量/召回率）")


def chroma_reachable() -> bool:
    """探测 Chroma 服务是否可达（一次性 socket 连接后立即关闭，不影响服务）。

    用于本地判定：未启动 Chroma 时，Chroma 依赖测试应 skip 而非 fail。
    """
    import socket

    host = os.getenv("CHROMA_SERVER_HOST", "localhost")
    port = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


# 依赖 Chroma HttpClient 的测试（本地无 Chroma 服务时应 skip，而非 fail）
CHROMA_DEPENDENT = {
    "test_agentic_rag_agent",
    "test_ragas",
}


def pytest_collection_modifyitems(config, items):
    """CI 自动跳过需要外部服务的测试模块；
    Chroma 依赖模块在本地无 Chroma 服务时也自动跳过（避免无谓失败）。"""
    skip_external = pytest.mark.skip(reason="无外部服务（Chroma/CI），跳过")

    # 这些测试文件需要真实 ChromaDB/Qdrant/Ollama/LM Studio/数据文件
    external_markers = [
        "test_e2e",
        "test_qdrant_real_integration",
        "test_qdrant_e2e_real",
        "test_lm_studio_models",
        "test_realtime_query",
        "test_full_agent",
        "test_agentic_integration",
        "test_model_router_integration",
        "test_web_fallback",
        "test_web_fallback_free",
        "test_qdrant_stability",
        "test_app_features",
        "test_agent_complete",
        "test_multi_agent_optimization",
        "test_v2_4_react",
        "test_integration_v2_2",
        # 依赖 Chroma HttpClient 服务（本地/CI 无服务时连接拒绝）
        "test_agentic_rag_agent",
        "test_ragas",
    ]

    chroma_up = chroma_reachable()
    for item in items:
        module_name = item.module.__name__.split(".")[-1]
        if module_name in external_markers:
            if module_name in CHROMA_DEPENDENT:
                # Chroma 依赖：CI 或 本地无 Chroma 服务时跳过
                if IN_CI or not chroma_up:
                    item.add_marker(skip_external)
            elif IN_CI:
                item.add_marker(skip_external)
