"""L4 性能基准测试 — 通过 pytest 运行基准测试

此文件是 scripts/benchmark_test.py 的 pytest 包装器，
使 run_pyramid_tests.py 能通过 -m L4 标记过滤运行。

需要外部服务：Qdrant (6333) + Ollama (11434)
"""
import socket
import pytest
import sys
from pathlib import Path

# 添加 scripts 目录到 path
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))


def _check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """检查端口是否可连接"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# 外部服务检查 fixture
@pytest.fixture(scope="module")
def external_services():
    """检查外部服务是否可用"""
    qdrant_ok = _check_port("localhost", 6333)
    ollama_ok = _check_port("localhost", 11434)
    if not (qdrant_ok and ollama_ok):
        pytest.skip("需要 Qdrant (6333) 和 Ollama (11434) 服务运行中")
    return True


@pytest.mark.L4
class TestBenchmark:
    """性能基准测试 — L4"""

    def test_import_benchmark(self, external_services):
        """能正确导入 benchmark_test 模块"""
        import benchmark_test
        assert hasattr(benchmark_test, "BENCHMARK_QUESTIONS")
        assert len(benchmark_test.BENCHMARK_QUESTIONS) == 20

    def test_percentile_calculation(self, external_services):
        """百分位计算正确"""
        from benchmark_test import calculate_percentiles
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = calculate_percentiles(latencies)
        assert result["min"] == 1.0
        assert result["max"] == 10.0
        assert result["p50"] <= result["p90"] <= result["p99"]
        assert result["mean"] == 5.5

    def test_percentile_empty_list(self, external_services):
        """空列表百分位计算返回零值"""
        from benchmark_test import calculate_percentiles
        result = calculate_percentiles([])
        assert result["p50"] == 0.0
        assert result["p90"] == 0.0
        assert result["p99"] == 0.0

    def test_percentile_single_value(self, external_services):
        """单值百分位计算"""
        from benchmark_test import calculate_percentiles
        result = calculate_percentiles([3.5])
        assert result["p50"] == 3.5
        assert result["p90"] == 3.5
        assert result["p99"] == 3.5

    def test_hit_at_k_calculation(self, external_services):
        """Hit@K 召回率计算"""
        from benchmark_test import calculate_hit_at_k
        answer = "RAG是检索增强生成技术，使用BM25和向量检索"
        keywords = ["检索增强生成", "BM25", "向量", "Cross-Encoder", "unknown"]
        result = calculate_hit_at_k(answer, keywords, k=5)
        assert result["hit_count"] == 3
        assert result["total"] == 5
        assert result["hit_at_k"] == 0.6

    def test_hit_at_k_empty_keywords(self, external_services):
        """空关键词列表 Hit@K"""
        from benchmark_test import calculate_hit_at_k
        result = calculate_hit_at_k("answer", [], k=5)
        assert result["hit_at_k"] == 0.0
        assert result["hit_count"] == 0

    def test_estimate_tokens(self, external_services):
        """Token 估算函数"""
        from benchmark_test import estimate_tokens
        # 纯中文
        cn_tokens = estimate_tokens("检索增强生成技术")
        assert cn_tokens > 0
        # 纯英文
        en_tokens = estimate_tokens("Retrieval Augmented Generation")
        assert en_tokens > 0
        # 混合
        mix_tokens = estimate_tokens("RAG是Retrieval Augmented Generation的缩写")
        assert mix_tokens > 0

    def test_rag_call_single_question(self, external_services):
        """真实 RAG 调用单题基准"""
        from benchmark_test import run_single_benchmark, BENCHMARK_QUESTIONS
        # 选取第一题，1轮
        result = run_single_benchmark(BENCHMARK_QUESTIONS[0], rounds=1)
        assert "latencies" in result
        assert len(result["latencies"]) == 1
        assert "percentiles" in result
        assert "hit_at_k" in result
        assert isinstance(result["simulation"], bool)

    def test_benchmark_summary(self, external_services):
        """基准测试汇总统计"""
        from benchmark_test import run_benchmark
        # 运行小规模基准（2题×1轮）
        report = run_benchmark(rounds=1)
        assert "summary" in report
        assert "latency" in report["summary"]
        assert "p50" in report["summary"]["latency"]
        assert "p90" in report["summary"]["latency"]
        assert "p99" in report["summary"]["latency"]
        assert "avg_hit_at_k" in report["summary"]
        assert "total_errors" in report["summary"]
