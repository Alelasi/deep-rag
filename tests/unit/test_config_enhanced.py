"""L1 单元测试 — config 模块增强点（环境变量驱动路径 + LLM_REGISTRY）

覆盖：
- DEEP_RAG_HOME / CHROMA_DB_PATH 的环境变量解析（含默认推导）
- LLM_REGISTRY 注册表完整性与分发（backend="none" 返回 None，无需网络）

说明：config 在 import 时读取环境变量并构建全局常量；路径相关测试通过
reload 模块 + 控制环境变量来断言推导关系，避免写死绝对路径。
"""
import importlib

import pytest

src_config = pytest.importorskip("src.config")


@pytest.mark.L1
@pytest.mark.unit
class TestLLMRegistry:
    """LLM_REGISTRY 注册表"""

    def test_registry_completeness(self):
        expected = {
            "anthropic", "zhipu", "ollama", "lmstudio", "siliconcloud",
            "groq", "cerebras", "openrouter", "openai", "none",
        }
        assert set(src_config.LLM_REGISTRY.keys()) == expected

    def test_registry_values_callable(self):
        for factory in src_config.LLM_REGISTRY.values():
            assert callable(factory)

    def test_registry_dispatch_none_returns_none(self, monkeypatch):
        # backend="none" 对应 _build_none，不触发任何网络/Key 依赖
        monkeypatch.setattr(src_config, "LLM_BACKEND", "none")
        monkeypatch.setattr(src_config, "ENABLE_MODEL_ROUTING", False)
        monkeypatch.setattr(src_config, "MODEL_CANDIDATES", "")
        assert src_config.get_llm() is None


@pytest.mark.L1
@pytest.mark.unit
class TestEnvDrivenPaths:
    """环境变量驱动的路径解析"""

    def _reload(self, monkeypatch, **env):
        for k in ("DEEP_RAG_HOME", "CHROMA_DB_PATH"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        importlib.reload(src_config)
        return src_config

    def test_default_home_equals_project_root(self, monkeypatch):
        cfg = self._reload(monkeypatch)
        assert cfg.DEEP_RAG_HOME == cfg.PROJECT_ROOT

    def test_default_chroma_path_under_home(self, monkeypatch):
        cfg = self._reload(monkeypatch)
        assert cfg.CHROMA_DB_PATH == str(cfg.DEEP_RAG_HOME / "data" / "chroma")

    def test_deep_rag_home_override(self, monkeypatch, tmp_path):
        cfg = self._reload(monkeypatch, DEEP_RAG_HOME=str(tmp_path))
        assert cfg.DEEP_RAG_HOME == tmp_path
        assert cfg.CHROMA_DB_PATH == str(tmp_path / "data" / "chroma")

    def test_chroma_db_path_explicit_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "my_chroma"
        cfg = self._reload(monkeypatch, CHROMA_DB_PATH=str(custom))
        assert cfg.CHROMA_DB_PATH == str(custom)
