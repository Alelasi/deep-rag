"""Pytest 全局配置：CI 环境自动跳过依赖外部服务的测试

在 GitHub Actions 中，无 ChromaDB/Qdrant/Ollama/LM Studio 等外部服务，
这些测试会无限等待连接，导致 CI 挂死。本文件在 collection 阶段自动跳过它们。
"""
import os
import pytest

# CI 环境标记
IN_CI = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"


def pytest_collection_modifyitems(config, items):
    """CI 中自动跳过需要外部服务的测试模块"""
    if not IN_CI:
        return

    skip_external = pytest.mark.skip(reason="CI 环境无外部服务，跳过")

    # 这些测试文件需要真实 ChromaDB/Qdrant/Ollama/LM Studio/数据文件
    external_markers = [
        "test_e2e",
        "test_qdrant_real_integration",
        "test_qdrant_e2e_real",
        "test_qdrant_retriever",
        "test_lm_studio_models",
        "test_realtime_query",
        "test_agent_vs_real_db",
        "test_full_agent",
        "test_agentic_integration",
        "test_model_router_integration",
        "test_pgvector_retriever",
        "test_web_fallback",
        "test_web_fallback_free",
        "test_qdrant_stability",
        "test_app_features",
        "test_agent_complete",
        "test_multi_agent_optimization",
        "test_v2_4_react",
        "test_integration_v2_2",
    ]

    for item in items:
        module_name = item.module.__name__.split(".")[-1]
        if module_name in external_markers:
            item.add_marker(skip_external)
