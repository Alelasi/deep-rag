"""DeepRAG完整测试运行器
按测试金字塔顺序执行所有测试套件
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

TEST_SUITES = [
    # (test_file, description, level)
    ("tests/test_agentic_tools.py",  "Agentic RAG工具箱", "L1"),
    ("tests/test_agent_router.py",   "Agent决策路由器",  "L1"),
    ("tests/test_v2_4_react.py",     "v2.4 ReAct循环+免费LLM", "L1"),
    ("tests/test_qdrant_retriever.py", "Qdrant检索器",     "L1"),
    ("tests/test_reranker.py",       "Reranker重排序",   "L1"),
    ("tests/test_mcp_server.py",     "MCP Server",       "L1"),
    ("tests/test_e2e.py",            "端到端Pipeline",   "L2"),
    ("tests/test_agentic_integration.py", "Agentic集成",  "L2"),
    ("tests/test_api.py",            "FastAPI接口",      "L2"),
]


def run_suite(test_file, description, level):
    """运行单个测试套件，返回(passed, failed)"""
    print(f"\n{'='*60}")
    print(f"  [{level}] {description}")
    print(f"  File: {test_file}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, test_file],
        cwd=str(PROJECT_ROOT),
        env={"PYTHONIOENCODING": "utf-8", **__import__("os").environ},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (result.stdout or "") + (result.stderr or "")
    # 提取最后的 "Results: X/Y passed"
    for line in reversed(output.split("\n")):
        if "Results:" in line and "passed" in line:
            print(f"  {line.strip()}")
            return result.returncode == 0
        if "全部测试通过" in line:
            print(f"  All e2e tests passed")
            return True

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        # 打印tail供debug
        tail_lines = output.strip().split("\n")[-5:]
        print(f"  Last lines: {tail_lines}")
        return False
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DeepRAG 完整测试运行器")
    print("=" * 60)

    results = []
    for test_file, description, level in TEST_SUITES:
        success = run_suite(test_file, description, level)
        results.append((description, level, success))

    print("\n" + "=" * 60)
    print("  测试套件汇总")
    print("=" * 60)
    for desc, level, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{level}] {desc}: {status}")

    total = len(results)
    passed = sum(1 for _, _, s in results if s)
    print(f"\n  总计: {passed}/{total} 套件通过")
    sys.exit(0 if passed == total else 1)
