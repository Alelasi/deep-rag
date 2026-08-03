"""L1 单元测试 — 配置模块（无外部依赖）

测试 src/config.py 的纯逻辑部分：
- 温度策略
- 路径解析
- 环境变量读取
"""
import os
import pytest
from pathlib import Path


@pytest.mark.L1
class TestTemperatureStrategy:
    """温度策略测试"""

    def test_temperature_strategy_keys(self):
        """温度策略包含所有必要键"""
        from src.config import TEMPERATURE_STRATEGY
        expected_keys = {"fact_check", "doc_grading", "generation",
                         "query_rewrite", "creative", "comparison", "arbitration"}
        assert set(TEMPERATURE_STRATEGY.keys()) == expected_keys

    def test_fact_check_temperature_is_zero(self):
        """事实校验温度为0（确定性）"""
        from src.config import TEMPERATURE_STRATEGY
        assert TEMPERATURE_STRATEGY["fact_check"] == 0.0

    def test_doc_grading_temperature_is_zero(self):
        """文档评分温度为0（确定性）"""
        from src.config import TEMPERATURE_STRATEGY
        assert TEMPERATURE_STRATEGY["doc_grading"] == 0.0

    def test_generation_temperature_moderate(self):
        """答案生成温度适中（0.2-0.5）"""
        from src.config import TEMPERATURE_STRATEGY
        temp = TEMPERATURE_STRATEGY["generation"]
        assert 0.2 <= temp <= 0.5

    def test_creative_temperature_highest(self):
        """创意场景温度最高"""
        from src.config import TEMPERATURE_STRATEGY
        temps = TEMPERATURE_STRATEGY.values()
        assert TEMPERATURE_STRATEGY["creative"] == max(temps)

    def test_all_temperatures_in_valid_range(self):
        """所有温度值在0-1范围内"""
        from src.config import TEMPERATURE_STRATEGY
        for name, temp in TEMPERATURE_STRATEGY.items():
            assert 0.0 <= temp <= 1.0, f"{name} temperature {temp} out of range"


@pytest.mark.L1
class TestPathConfiguration:
    """路径配置测试"""

    def test_project_root_exists(self):
        """项目根目录存在"""
        from src.config import PROJECT_ROOT
        assert Path(PROJECT_ROOT).exists()

    def test_data_dir_defined(self):
        """数据目录已定义"""
        from src.config import DATA_DIR
        assert DATA_DIR is not None

    def test_chroma_db_path_has_default(self):
        """ChromaDB 路径有默认值"""
        from src.config import CHROMA_DB_PATH
        assert CHROMA_DB_PATH  # 非空

    def test_chroma_server_host_default_localhost(self):
        """ChromaDB 服务器默认地址为 localhost"""
        from src.config import CHROMA_SERVER_HOST
        assert CHROMA_SERVER_HOST == "localhost"

    def test_chroma_server_port_default_8000(self):
        """ChromaDB 服务器默认端口为 8000"""
        from src.config import CHROMA_SERVER_PORT
        assert CHROMA_SERVER_PORT == 8000


@pytest.mark.L1
class TestLLMConfig:
    """LLM 配置测试"""

    def test_llm_backend_has_default(self):
        """LLM 后端有默认值"""
        from src.config import LLM_BACKEND
        assert LLM_BACKEND  # 非空

    def test_llm_temperature_default(self):
        """LLM 温度有默认值"""
        from src.config import LLM_TEMPERATURE
        assert isinstance(LLM_TEMPERATURE, (int, float))
        assert 0.0 <= LLM_TEMPERATURE <= 1.0

    def test_api_keys_are_strings(self):
        """API Key 均为字符串类型"""
        from src.config import (ANTHROPIC_API_KEY, OPENAI_API_KEY,
                                ZHIPU_API_KEY, SILICONFLOW_API_KEY)
        for key in [ANTHROPIC_API_KEY, OPENAI_API_KEY, ZHIPU_API_KEY, SILICONFLOW_API_KEY]:
            assert isinstance(key, str)

    def test_device_is_cpu_or_cuda(self):
        """设备类型为 cpu 或 cuda"""
        from src.config import DEVICE
        assert DEVICE in ("cpu", "cuda")


@pytest.mark.L1
class TestEnvironmentSetup:
    """环境变量设置测试"""

    def test_kmp_duplicate_lib_ok_set(self):
        """KMP_DUPLICATE_LIB_OK 已设置（防止 OpenMP 冲突）"""
        assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"

    def test_no_proxy_includes_localhost(self):
        """no_proxy 包含 localhost"""
        no_proxy = os.environ.get("no_proxy", "")
        assert "localhost" in no_proxy or "127.0.0.1" in no_proxy
