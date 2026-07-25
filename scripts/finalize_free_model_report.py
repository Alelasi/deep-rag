#!/usr/bin/env python3
"""汇总已测免费模型 + 补测智谱，写出 v3 报告。"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "docs" / "free_model_benchmark_report_v3.md"
OUT_JSON = ROOT / "docs" / "free_model_benchmark_data_v3.json"

# load .env
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

RESULTS = [
    {
        "name": "Groq Llama3.1-8B",
        "provider": "Groq",
        "model_id": "llama-3.1-8b-instant",
        "success": 6,
        "total": 6,
        "avg_latency_ms": 1071.1,
        "avg_speed": 219.7,
        "avg_score": 8.64,
    },
    {
        "name": "Groq Qwen3.6-27B",
        "provider": "Groq",
        "model_id": "qwen/qwen3.6-27b",
        "success": 6,
        "total": 6,
        "avg_latency_ms": 1941.7,
        "avg_speed": 359.7,
        "avg_score": 7.92,
    },
    {
        "name": "Cerebras GPT-OSS-120B",
        "provider": "Cerebras",
        "model_id": "gpt-oss-120b",
        "success": 6,
        "total": 6,
        "avg_latency_ms": 804.7,
        "avg_speed": 610.5,
        "avg_score": 8.82,
    },
    {
        "name": "Silicon GLM-Z1-9B",
        "provider": "SiliconFlow",
        "model_id": "THUDM/GLM-Z1-9B-0414",
        "success": 6,
        "total": 6,
        "avg_latency_ms": round(statistics.mean([15620, 14492, 28230, 7905, 4973, 19461]), 1),
        "avg_speed": 0,
        "avg_score": round(statistics.mean([8.25, 7.87, 8.33, 7.28, 8.07, 7.9]), 2),
    },
    {
        "name": "Silicon Qwen2.5-7B",
        "provider": "SiliconFlow",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "success": 6,
        "total": 6,
        "avg_latency_ms": round(statistics.mean([2591, 1604, 2329, 6004, 1244, 4137]), 1),
        "avg_speed": 0,
        "avg_score": round(statistics.mean([7.78, 6.16, 7.45, 8.48, 6.83, 6.17]), 2),
    },
]

QUESTIONS = [
    ("rag", "什么是RAG检索增强生成？分点说明相对纯LLM的优势。", ["检索", "生成", "知识", "幻觉"]),
    ("math", "求解 2x+5=17，写出步骤并给出 x 的值。", ["x", "6", "12"]),
    ("code", "用Python写 is_prime(n: int) -> bool，含类型注解。", ["def", "is_prime", "return", "bool"]),
    ("zh", "解释成语画蛇添足，并举一个生活例子。", ["多余", "蛇", "足"]),
    ("json", "从'2025年3月15日，张三在北京开会'提取人名地点时间，只输出JSON。", ["张三", "北京", "2025"]),
    ("debug", "Python IndentationError: unexpected indent 常见原因与修复？", ["缩进", "空格", "Tab"]),
]


def score(text: str, kw: list[str], lat: float, cat: str) -> float:
    if not text:
        return 0.0
    hits = sum(1 for k in kw if k.lower() in text.lower())
    kws = hits / max(1, len(kw)) * 10
    n = len(text)
    length = 9 if 60 <= n <= 1500 else (6 if n >= 30 else 3)
    struct = 8 if any(x in text for x in ("\n", "1.", "-", "：", "{")) else 5
    zh = min(10.0, 4 + sum(1 for c in text if "一" <= c <= "鿿") / max(1, n) * 10)
    code = 5.0
    if cat == "code":
        code = min(10.0, sum(2 for tip in ("def ", "return", "int", "bool") if tip in text))
    instr = 9 if (cat != "json" or ("{" in text and "}" in text)) else 3
    if cat == "math" and ("x=6" in text.replace(" ", "") or "x = 6" in text):
        instr = 10
    lat_s = 10 if lat <= 1000 else 8 if lat <= 2000 else 6 if lat <= 5000 else 4 if lat <= 12000 else 2
    return round(
        kws * 0.36 + length * 0.12 + struct * 0.10 + zh * 0.12 + code * 0.10 + instr * 0.10 + lat_s * 0.10,
        2,
    )


def test_chat(name: str, provider: str, model: str, url: str, key: str, sleep_s: float = 2.5) -> dict:
    print(f"=== {name}", flush=True)
    scores, lats = [], []
    ok = 0
    for qid, q, kw in QUESTIONS:
        t0 = time.time()
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": q}],
                    "temperature": 0.2,
                    "max_tokens": 700,
                },
                timeout=60,
            )
            lat = (time.time() - t0) * 1000
            if r.status_code == 200:
                content = (
                    r.json().get("choices", [{}])[0].get("message", {}).get("content") or ""
                ).strip()
                sc = score(content, kw, lat, qid)
                scores.append(sc)
                lats.append(lat)
                ok += 1
                print(f"  {qid} OK {lat:.0f}ms {sc}", flush=True)
            else:
                print(f"  {qid} FAIL {r.status_code} {r.text[:80]}", flush=True)
                scores.append(0.0)
        except Exception as e:
            print(f"  {qid} FAIL {e}", flush=True)
            scores.append(0.0)
        time.sleep(sleep_s)
    return {
        "name": name,
        "provider": provider,
        "model_id": model,
        "success": ok,
        "total": 6,
        "avg_latency_ms": round(statistics.mean(lats), 1) if lats else 0,
        "avg_speed": 0,
        "avg_score": round(statistics.mean(scores), 2),
    }


def main() -> int:
    results = list(RESULTS)
    zkey = os.getenv("ZHIPU_API_KEY", "").strip()
    if zkey:
        for mid, name in [
            ("glm-4.5-flash", "Zhipu GLM-4.5-Flash"),
            ("glm-4-flash", "Zhipu GLM-4-Flash"),
        ]:
            results.append(
                test_chat(
                    name,
                    "Zhipu",
                    mid,
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    zkey,
                    2.5,
                )
            )
            time.sleep(2)

    results.sort(key=lambda x: (-x["avg_score"], x["avg_latency_ms"]))

    lines = [
        "# 免费 LLM 模型横向对比报告 v3",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 模型数：{len(results)} · 每模型 6 题（知识/数学/代码/中文/JSON/排错）",
        "- 评分：关键词+结构+中文+代码+指令+延迟 规则启发式；失败计 0；**非人工金标**",
        "- 免费档易 429：本报告仅收录完整 6/6 或已补测完成的结果",
        "- 模型 ID 以 2026-07-18 各平台 `/models` 实测为准",
        "",
        "## 综合排名",
        "",
        "| 排名 | 模型 | 厂商 | 综合分 | 成功率 | 平均延迟 |",
        "|-----:|------|------|-------:|-------:|---------:|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['provider']} | **{r['avg_score']}** | "
            f"{r['success']}/{r['total']} | {r['avg_latency_ms']:.0f}ms |"
        )

    lines += ["", "## 分厂商最佳", ""]
    best: dict = {}
    for r in results:
        if r["success"] <= 0:
            continue
        if r["provider"] not in best or r["avg_score"] > best[r["provider"]]["avg_score"]:
            best[r["provider"]] = r
    for p, r in sorted(best.items(), key=lambda kv: -kv[1]["avg_score"]):
        lines.append(f"- **{p}**: {r['name']} · 分 {r['avg_score']} · {r['avg_latency_ms']:.0f}ms")

    lines += [
        "",
        "## 结论与 DeepRAG 建议",
        "",
        "| 场景 | 推荐 | 理由 |",
        "|------|------|------|",
        "| **极速演示** | Cerebras GPT-OSS-120B / Groq Llama3.1-8B | 延迟约 0.8–1.1s |",
        "| **中文问答** | Groq Llama3.1-8B / Silicon GLM-Z1-9B / Zhipu Flash | 中文题更稳 |",
        "| **当前默认（本机）** | Silicon GLM-Z1-9B | 与 DeepRAG 现配置一致，质量尚可但偏慢(约 15s) |",
        "| **降级链** | Groq → Cerebras → Silicon → Zhipu | 多厂商冗余，避开单家 429 |",
        "",
        "## 原始数据",
        "",
        f"- JSON：`docs/free_model_benchmark_data_v3.json`",
        "- 旧版 v2：`docs/free_model_benchmark_report_v2.md`（2026-07-15）",
        "- 脚本：`scripts/benchmark_free_models_v3.py` / `scripts/finalize_free_model_report.py`",
        "- **密钥仅从环境变量读取**（已清理 v1/v2 硬编码）",
        "",
    ]
    md = "\n".join(lines)
    OUT_MD.write_text(md, encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps({"time": datetime.now().isoformat(), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(md)
    print("WROTE", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
