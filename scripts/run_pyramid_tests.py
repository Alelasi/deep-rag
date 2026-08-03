#!/usr/bin/env python3
"""DeepRAG 测试金字塔运行器 — L1/L2/L3/L4 四层测试

用法：
    python scripts/run_pyramid_tests.py                    # 运行所有层级
    python scripts/run_pyramid_tests.py --level L1         # 只运行单元测试
    python scripts/run_pyramid_tests.py --level L3 --html  # 端到端测试+HTML报告
    python scripts/run_pyramid_tests.py --level all --cov  # 全部测试+覆盖率
"""
import argparse
import json
import socket
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# 项目根目录
ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "tests" / "reports"

# 测试层级配置
PYRAMID_LEVELS = {
    "L1": {
        "name": "单元测试",
        "desc": "无外部依赖，纯逻辑验证",
        "marker": "L1",
        "paths": ["tests/test_chunker.py", "tests/test_graph_nodes.py", "tests/test_config.py"],
    },
    "L2": {
        "name": "集成测试",
        "desc": "需要外部服务（Qdrant/Ollama/API）",
        "marker": "L2",
        "paths": ["tests/test_qdrant_retriever.py", "tests/test_qdrant_real_integration.py"],
    },
    "L3": {
        "name": "端到端测试",
        "desc": "完整 RAG 管道",
        "marker": "L3",
        "paths": ["tests/test_e2e_comprehensive.py"],
    },
    "L4": {
        "name": "性能基准",
        "desc": "响应时间/吞吐量/召回率",
        "marker": "L4",
        "paths": ["tests/test_e2e_benchmark.py"],
    },
}

# ============================================================
# ANSI 颜色码
# ============================================================
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def colorize(text: str, color: str) -> str:
    """给文本添加颜色"""
    return f"{color}{text}{Color.RESET}"


# ============================================================
# 外部服务检查
# ============================================================

def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """检查指定端口是否可连接"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_external_services() -> dict:
    """检查 Qdrant (6333) 和 Ollama (11434) 是否可用

    Returns:
        {"qdrant": bool, "ollama": bool}
    """
    return {
        "qdrant": check_port("localhost", 6333),
        "ollama": check_port("localhost", 11434),
    }


# ============================================================
# 单层运行
# ============================================================

def run_level(level_key: str, args: argparse.Namespace) -> dict:
    """运行指定层级的测试

    Args:
        level_key: 层级键 (L1/L2/L3/L4)
        args: 命令行参数

    Returns:
        {"level": str, "name": str, "passed": int, "failed": int,
         "skipped": int, "duration": float, "status": str, "message": str}
    """
    cfg = PYRAMID_LEVELS[level_key]
    result = {
        "level": level_key,
        "name": cfg["name"],
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "duration": 0.0,
        "status": "unknown",
        "message": "",
    }

    print()
    print(colorize("=" * 70, Color.GRAY))
    print(colorize(f"  {level_key} — {cfg['name']}", Color.BOLD + Color.CYAN))
    print(colorize(f"  {cfg['desc']}", Color.GRAY))
    print(colorize("=" * 70, Color.GRAY))

    # --- 检查测试文件是否存在 ---
    existing_paths = []
    missing_paths = []
    for p in cfg["paths"]:
        full = ROOT / p
        if full.exists():
            existing_paths.append(p)
        else:
            missing_paths.append(p)

    if missing_paths:
        for mp in missing_paths:
            print(colorize(f"  [WARN] 测试文件不存在: {mp}", Color.YELLOW))

    if not existing_paths:
        result["status"] = "skipped"
        result["message"] = "该层级无可用测试文件"
        print(colorize(f"  [SKIP] {level_key} 无可用测试文件，跳过", Color.YELLOW))
        return result

    # --- L3/L4 需要检查外部服务 ---
    if level_key in ("L3", "L4"):
        services = check_external_services()
        unavailable = [name for name, ok in services.items() if not ok]
        if unavailable:
            result["status"] = "skipped"
            result["message"] = f"外部服务不可用: {', '.join(unavailable)}"
            print(colorize(f"  [SKIP] 外部服务不可用:", Color.YELLOW))
            for name, ok in services.items():
                status = colorize("OK", Color.GREEN) if ok else colorize("OFFLINE", Color.RED)
                port = 6333 if name == "qdrant" else 11434
                print(f"    {name:>8} (:{port})  {status}")
            print(colorize(f"  跳过 {level_key} 测试。请先启动 Qdrant 和 Ollama。", Color.YELLOW))
            return result
        else:
            print(colorize("  外部服务检查:", Color.GREEN))
            for name, ok in services.items():
                port = 6333 if name == "qdrant" else 11434
                print(f"    {name:>8} (:{port})  {colorize('OK', Color.GREEN)}")

    # --- 构建 pytest 命令 ---
    cmd = [sys.executable, "-m", "pytest"]

    # 测试路径
    cmd.extend(existing_paths)

    # 标记过滤
    cmd.extend(["-m", cfg["marker"]])

    # 快速失败
    if args.failfast:
        cmd.append("-x")

    # 覆盖率
    if args.cov:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if args.html:
            cmd.append("--cov-report=html:tests/reports/coverage_html")

    # HTML 报告
    if args.html:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = REPORTS_DIR / f"{level_key}_report_{ts}.html"
        cmd.extend([f"--html={html_path}", "--self-contained-html"])

    # 常用选项
    cmd.extend(["-v", "--tb=short", f"--rootdir={ROOT}"])

    print(colorize(f"  运行命令: {' '.join(cmd)}", Color.GRAY))
    print()

    # --- 执行 ---
    import time
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - start
    result["duration"] = round(elapsed, 2)

    # 解析退出码
    # pytest 退出码：0=全部通过, 1=有失败, 2=被中断, 5=没有收集到测试
    if proc.returncode == 0:
        result["status"] = "passed"
        result["passed"] = 1  # 精确数量从输出解析
    elif proc.returncode == 5:
        result["status"] = "skipped"
        result["message"] = "没有收集到测试（标记不匹配）"
    else:
        result["status"] = "failed"
        result["failed"] = 1

    # 尝试从 JUnit XML 获取精确数字（如果可用）
    # 这里用简化方式：通过 --co 收集计数
    collect_cmd = [sys.executable, "-m", "pytest"] + existing_paths + ["-m", cfg["marker"], "--co", "-q"]
    try:
        collect_proc = subprocess.run(
            collect_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=30
        )
        lines = collect_proc.stdout.strip().split("\n") if collect_proc.stdout.strip() else []
        # 最后几行通常有 "X tests selected" 或错误信息
        for line in lines[-5:]:
            line = line.strip()
            if "test" in line.lower() and ("selected" in line.lower() or "error" in line.lower()):
                # 尝试提取数字
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        if "selected" in line.lower():
                            result["passed"] = int(part) if result["status"] == "passed" else result["passed"]
                        break
    except (subprocess.TimeoutExpired, Exception):
        pass

    # 状态输出
    if result["status"] == "passed":
        print(colorize(f"  [PASS] {level_key} 全部通过 ({elapsed:.1f}s)", Color.GREEN))
    elif result["status"] == "failed":
        print(colorize(f"  [FAIL] {level_key} 存在失败 ({elapsed:.1f}s)", Color.RED))
    elif result["status"] == "skipped":
        print(colorize(f"  [SKIP] {level_key} 已跳过", Color.YELLOW))

    return result


# ============================================================
# 汇总输出
# ============================================================

def print_summary(results: list):
    """打印彩色汇总表"""
    print()
    print(colorize("=" * 70, Color.BOLD + Color.BLUE))
    print(colorize("  测试金字塔汇总报告", Color.BOLD + Color.BLUE))
    print(colorize("=" * 70, Color.BOLD + Color.BLUE))
    print()

    # 表头
    header = f"  {'层级':<6} {'名称':<12} {'状态':<10} {'耗时':<10} {'说明'}"
    print(colorize(header, Color.BOLD))
    print(colorize("  " + "-" * 66, Color.GRAY))

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for r in results:
        level = r["level"]
        name = r["name"]
        status = r["status"]
        duration = f"{r['duration']:.1f}s"
        message = r.get("message", "")

        if status == "passed":
            status_str = colorize("PASS", Color.GREEN)
            total_pass += 1
        elif status == "failed":
            status_str = colorize("FAIL", Color.RED)
            total_fail += 1
        elif status == "skipped":
            status_str = colorize("SKIP", Color.YELLOW)
            total_skip += 1
        else:
            status_str = colorize(status.upper(), Color.GRAY)

        row = f"  {level:<6} {name:<12} {status_str:<18} {duration:<10} {message}"
        print(row)

    print(colorize("  " + "-" * 66, Color.GRAY))
    print()

    # 总结
    total = len(results)
    summary_parts = []
    if total_pass:
        summary_parts.append(colorize(f"通过 {total_pass}", Color.GREEN))
    if total_fail:
        summary_parts.append(colorize(f"失败 {total_fail}", Color.RED))
    if total_skip:
        summary_parts.append(colorize(f"跳过 {total_skip}", Color.YELLOW))

    summary_str = " | ".join(summary_parts) if summary_parts else "无结果"
    print(f"  总计: {total} 个层级 — {summary_str}")
    print()

    # 最终状态
    if total_fail > 0:
        print(colorize("  [RESULT] 测试金字塔存在失败层级", Color.BOLD + Color.RED))
    elif total_skip == total:
        print(colorize("  [RESULT] 所有层级均被跳过", Color.YELLOW))
    else:
        print(colorize("  [RESULT] 测试金字塔全部通过", Color.BOLD + Color.GREEN))

    print(colorize("=" * 70, Color.GRAY))


# ============================================================
# 保存 JSON 报告
# ============================================================

def save_json_report(results: list):
    """保存 JSON 格式的测试结果到 tests/reports/"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"pyramid_report_{ts}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "levels": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["status"] == "passed"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(colorize(f"\n  JSON 报告已保存: {report_path}", Color.CYAN))


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepRAG 测试金字塔运行器 — L1/L2/L3/L4 四层测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_pyramid_tests.py                    # 运行所有层级
  python scripts/run_pyramid_tests.py --level L1         # 只运行单元测试
  python scripts/run_pyramid_tests.py --level L3 --html  # 端到端测试+HTML报告
  python scripts/run_pyramid_tests.py --level all --cov  # 全部测试+覆盖率
  python scripts/run_pyramid_tests.py --level L1 --failfast  # 快速失败
        """,
    )
    parser.add_argument(
        "--level",
        choices=["L1", "L2", "L3", "L4", "all"],
        default="all",
        help="测试层级 (默认: all)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="生成 HTML 报告 (需要 pytest-html)",
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="生成覆盖率报告 (需要 pytest-cov)",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="遇到第一个失败即停止",
    )
    args = parser.parse_args()

    # 确保报告目录存在
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(colorize("\n" + "=" * 70, Color.BOLD + Color.MAGENTA))
    print(colorize("  DeepRAG 测试金字塔运行器", Color.BOLD + Color.MAGENTA))
    print(colorize(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Color.GRAY))
    print(colorize(f"  层级: {args.level}  |  HTML: {args.html}  |  覆盖率: {args.cov}  |  快速失败: {args.failfast}",
                   Color.GRAY))
    print(colorize("=" * 70, Color.BOLD + Color.MAGENTA))

    # 确定要运行的层级
    if args.level == "all":
        levels_to_run = list(PYRAMID_LEVELS.keys())
    else:
        levels_to_run = [args.level]

    results = []
    for level_key in levels_to_run:
        r = run_level(level_key, args)
        results.append(r)
        # failfast: 如果失败则不继续后续层级
        if args.failfast and r["status"] == "failed":
            print(colorize(f"\n  [FAILFAST] {level_key} 失败，停止后续层级", Color.RED))
            break

    # 打印汇总
    print_summary(results)

    # 保存 JSON 报告
    save_json_report(results)

    # 返回码：有失败则返回1
    has_failure = any(r["status"] == "failed" for r in results)
    sys.exit(1 if has_failure else 0)


if __name__ == "__main__":
    main()
