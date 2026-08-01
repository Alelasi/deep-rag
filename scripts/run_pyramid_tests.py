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

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "tests" / "reports"

PYRAMID_LEVELS = {
    "L1": {"name": "单元测试", "desc": "无外部依赖，纯逻辑验证", "marker": "L1",
            "paths": ["tests/test_chunker.py", "tests/test_graph_nodes.py", "tests/test_config.py"]},
    "L2": {"name": "集成测试", "desc": "需要外部服务（Qdrant/Ollama/API）", "marker": "L2",
            "paths": ["tests/test_qdrant_retriever.py", "tests/test_qdrant_real_integration.py"]},
    "L3": {"name": "端到端测试", "desc": "完整 RAG 管道", "marker": "L3",
            "paths": ["scripts/comprehensive_test.py"]},
    "L4": {"name": "性能基准", "desc": "响应时间/吞吐量/召回率", "marker": "L4",
            "paths": ["scripts/benchmark_test.py"]},
}

class Color:
    RESET = "\033[0m"; BOLD = "\033[1m"; RED = "\033[91m"; GREEN = "\033[92m"
    YELLOW = "\033[93m"; BLUE = "\033[94m"; MAGENTA = "\033[95m"; CYAN = "\033[96m"; GRAY = "\033[90m"

def colorize(text, color): return f"{color}{text}{Color.RESET}"

def check_port(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout): return True
    except (socket.timeout, ConnectionRefusedError, OSError): return False

def check_external_services():
    return {"qdrant": check_port("localhost", 6333), "ollama": check_port("localhost", 11434)}

def run_level(level_key, args):
    cfg = PYRAMID_LEVELS[level_key]
    result = {"level": level_key, "name": cfg["name"], "passed": 0, "failed": 0,
              "skipped": 0, "duration": 0.0, "status": "unknown", "message": ""}
    print(f"\n{'='*70}\n  {level_key} — {cfg['name']}\n  {cfg['desc']}\n{'='*70}")
    existing_paths = [p for p in cfg["paths"] if (ROOT / p).exists()]
    missing = [p for p in cfg["paths"] if not (ROOT / p).exists()]
    for mp in missing: print(f"  [WARN] 测试文件不存在: {mp}")
    if not existing_paths:
        result["status"] = "skipped"; result["message"] = "该层级无可用测试文件"
        return result
    if level_key in ("L3", "L4"):
        services = check_external_services()
        unavailable = [n for n, ok in services.items() if not ok]
        if unavailable:
            result["status"] = "skipped"; result["message"] = f"外部服务不可用: {', '.join(unavailable)}"
            return result
    cmd = [sys.executable, "-m", "pytest"] + existing_paths + ["-m", cfg["marker"]]
    if args.failfast: cmd.append("-x")
    if args.cov: cmd.extend(["--cov=src", "--cov-report=term-missing"])
    if args.html:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd.extend([f"--html={REPORTS_DIR}/{level_key}_report_{ts}.html", "--self-contained-html"])
    cmd.extend(["-v", "--tb=short", f"--rootdir={ROOT}"])
    import time; start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - start; result["duration"] = round(elapsed, 2)
    if proc.returncode == 0: result["status"] = "passed"; result["passed"] = 1
    elif proc.returncode == 5: result["status"] = "skipped"; result["message"] = "没有收集到测试"
    else: result["status"] = "failed"; result["failed"] = 1
    return result

def print_summary(results):
    print(f"\n{'='*70}\n  测试金字塔汇总报告\n{'='*70}")
    for r in results:
        status = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(r["status"], r["status"])
        print(f"  {r['level']:<6} {r['name']:<12} {status:<10} {r['duration']:.1f}s  {r.get('message','')}")
    fails = sum(1 for r in results if r["status"] == "failed")
    print(f"\n  {'[FAIL] 存在失败' if fails else '[PASS] 全部通过'}")

def save_json_report(results):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"pyramid_report_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "levels": results}, f, ensure_ascii=False, indent=2)
    print(f"  JSON 报告: {path}")

def main():
    parser = argparse.ArgumentParser(description="DeepRAG 测试金字塔运行器")
    parser.add_argument("--level", choices=["L1","L2","L3","L4","all"], default="all")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--cov", action="store_true")
    parser.add_argument("--failfast", action="store_true")
    args = parser.parse_args()
    levels = list(PYRAMID_LEVELS.keys()) if args.level == "all" else [args.level]
    results = []
    for lk in levels:
        r = run_level(lk, args); results.append(r)
        if args.failfast and r["status"] == "failed": break
    print_summary(results); save_json_report(results)
    sys.exit(1 if any(r["status"] == "failed" for r in results) else 0)

if __name__ == "__main__": main()
