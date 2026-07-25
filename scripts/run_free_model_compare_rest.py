#!/usr/bin/env python3
"""只测 Silicon/Zhipu/OpenRouter 代表模型，并与已测 Groq/Cerebras 结果合并出完整报告。"""
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
LOG = Path(r"D:\文档\ai提问相关\哲思灵智\free_model_bench_v3.log")

# load env
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


# 本会话已稳定测完的结果（6 题）
SEED = [
    {
        "name": "Groq Llama3.1-8B",
        "provider": "Groq",
        "model_id": "llama-3.1-8b-instant",
        "success": 6,
        "total": 6,
        "avg_latency_ms": 1071.1,
        "avg_speed": 219.7,
        "avg_score": 8.64,
        "source": "session_measured",
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
        "source": "session_measured",
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
        "source": "session_measured",
    },
]

QUESTIONS = [
    {"id": "rag", "cat": "知识", "q": "什么是RAG检索增强生成？分点说明相对纯LLM的优势。", "kw": ["检索", "生成", "知识", "幻觉"]},
    {"id": "math", "cat": "数学", "q": "求解 2x+5=17，写出步骤并给出 x 的值。", "kw": ["x", "6", "12"]},
    {"id": "code", "cat": "代码", "q": "用Python写 is_prime(n: int) -> bool，含类型注解。", "kw": ["def", "is_prime", "return", "bool"]},
    {"id": "zh", "cat": "中文", "q": "解释成语画蛇添足，并举一个生活例子。", "kw": ["多余", "蛇", "足"]},
    {"id": "json", "cat": "抽取", "q": "从'2025年3月15日，张三在北京开会'提取人名地点时间，只输出JSON。", "kw": ["张三", "北京", "2025"]},
    {"id": "debug", "cat": "排错", "q": "Python IndentationError: unexpected indent 常见原因与修复？", "kw": ["缩进", "空格", "Tab"]},
]


def call(base, key, model, prompt, provider):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "OpenRouter":
        headers["HTTP-Referer"] = "https://localhost/deeprag-bench"
        headers["X-Title"] = "DeepRAG-Free-Bench"
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 700,
        "stream": False,
    }
    last = ""
    for attempt in range(3):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            lat = (time.time() - t0) * 1000
            if r.status_code == 200:
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {}) or {}
                content = (msg.get("content") or msg.get("reasoning") or "").strip()
                usage = data.get("usage") or {}
                tokens = int(usage.get("completion_tokens") or max(1, int(len(content) * 0.6)))
                speed = tokens / (lat / 1000) if lat > 0 else 0
                return True, content, lat, tokens, speed, ""
            if r.status_code == 429:
                wait = 20 * (attempt + 1)
                last = f"429 {r.text[:100]}"
                log(f"      429 wait {wait}s")
                time.sleep(wait)
                continue
            last = f"HTTP {r.status_code}: {r.text[:120]}"
            if r.status_code in (400, 404):
                break
            time.sleep(2)
        except Exception as e:
            last = str(e)[:120]
            time.sleep(2)
    return False, "", 0.0, 0, 0.0, last


def score(q, text, lat, ok):
    if not ok or not text:
        return 0.0
    kws = q["kw"]
    hits = sum(1 for k in kws if k.lower() in text.lower())
    kw = hits / max(1, len(kws)) * 10
    n = len(text)
    length = 9 if 60 <= n <= 1500 else (6 if n >= 30 else 3)
    struct = 8 if any(x in text for x in ("\n", "1.", "-", "：", "{")) else 5
    zh = min(10.0, 4 + sum(1 for c in text if "一" <= c <= "鿿") / max(1, n) * 10)
    code = 5.0
    if q["cat"] == "代码":
        code = min(10.0, sum(2 for tip in ("def ", "return", "int", "bool") if tip in text))
    instr = 9 if (q["cat"] != "抽取" or ("{" in text and "}" in text)) else 3
    if q["cat"] == "数学" and ("x=6" in text.replace(" ", "") or "x = 6" in text):
        instr = 10
    lat_s = 10 if lat <= 1000 else 8 if lat <= 2000 else 6 if lat <= 5000 else 4 if lat <= 12000 else 2
    return round(
        kw * 0.36 + length * 0.12 + struct * 0.10 + zh * 0.12 + code * 0.10 + instr * 0.10 + lat_s * 0.10,
        2,
    )


def test_model(provider, mid, name, base, key, sleep_s):
    log(f"\n=== {name} ({mid}) ===")
    ok_n = 0
    lats, speeds, scores = [], [], []
    rows = []
    for j, q in enumerate(QUESTIONS, 1):
        log(f"  ({j}/{len(QUESTIONS)}) {q['id']} ...")
        ok, content, lat, tokens, speed, err = call(base, key, mid, q["q"], provider)
        sc = score(q, content, lat, ok)
        if ok:
            ok_n += 1
            lats.append(lat)
            speeds.append(speed)
            scores.append(sc)
            log(f"OK {lat:.0f}ms score={sc}")
        else:
            log(f"FAIL {err[:100]}")
        rows.append({"id": q["id"], "ok": ok, "latency_ms": round(lat, 1), "score": sc, "error": err, "preview": content[:200]})
        time.sleep(sleep_s)
    padded = scores + [0.0] * (len(QUESTIONS) - len(scores))
    return {
        "name": name,
        "provider": provider,
        "model_id": mid,
        "success": ok_n,
        "total": len(QUESTIONS),
        "avg_latency_ms": round(statistics.mean(lats), 1) if lats else 0,
        "avg_speed": round(statistics.mean(speeds), 1) if speeds else 0,
        "avg_score": round(statistics.mean(padded), 2),
        "rows": rows,
        "source": "session_measured",
    }


def main():
    LOG.write_text(f"===== REST START {datetime.now().isoformat()} =====\n", encoding="utf-8")
    results = list(SEED)
    # Silicon
    s = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if s:
        for mid, name in [
            ("THUDM/GLM-Z1-9B-0414", "Silicon GLM-Z1-9B"),
            ("Qwen/Qwen2.5-7B-Instruct", "Silicon Qwen2.5-7B"),
            ("Qwen/Qwen3-8B", "Silicon Qwen3-8B"),
        ]:
            results.append(test_model("SiliconFlow", mid, name, "https://api.siliconflow.cn/v1", s, 3.5))
            time.sleep(8)
    # Zhipu
    z = os.getenv("ZHIPU_API_KEY", "").strip()
    if z:
        for mid, name in [
            ("glm-4.5-flash", "Zhipu GLM-4.5-Flash"),
            ("glm-4-flash", "Zhipu GLM-4-Flash"),
        ]:
            results.append(test_model("Zhipu", mid, name, "https://open.bigmodel.cn/api/paas/v4", z, 3.0))
            time.sleep(8)
    # OpenRouter free (may be flaky)
    o = os.getenv("OPENROUTER_API_KEY", "").strip()
    if o:
        for mid, name in [
            ("openai/gpt-oss-20b:free", "OR GPT-OSS-20B free"),
            ("meta-llama/llama-3.3-70b-instruct:free", "OR Llama3.3-70B free"),
        ]:
            results.append(
                test_model(
                    "OpenRouter",
                    mid,
                    name,
                    "https://openrouter.ai/api/v1",
                    o,
                    15,
                )
            )
            time.sleep(12)

    results.sort(key=lambda x: (-x["avg_score"], x["avg_latency_ms"]))
    lines = [
        "# 免费 LLM 模型横向对比报告 v3",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 模型数：{len(results)} · 题量：每模型 6 题",
        "- 评分：关键词/结构/中文/代码/指令/延迟规则启发式；失败题计 0；**非人工金标**",
        "- Groq/Cerebras 部分结果来自本会话完整 6/6 实测；其余本脚本补测",
        "- 免费档易 429：已做长间隔 + 重试",
        "",
        "## 综合排名",
        "",
        "| 排名 | 模型 | 厂商 | 综合分 | 成功率 | 平均延迟 | 吞吐 |",
        "|-----:|------|------|-------:|-------:|---------:|-----:|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['provider']} | **{r['avg_score']}** | "
            f"{r['success']}/{r['total']} | {r['avg_latency_ms']:.0f}ms | {r['avg_speed']:.0f} tok/s |"
        )

    lines += ["", "## 分厂商最佳", ""]
    best = {}
    for r in results:
        if r["success"] <= 0:
            continue
        if r["provider"] not in best or r["avg_score"] > best[r["provider"]]["avg_score"]:
            best[r["provider"]] = r
    for p, r in sorted(best.items(), key=lambda kv: -kv[1]["avg_score"]):
        lines.append(f"- **{p}**: {r['name']} · 分 {r['avg_score']} · {r['avg_latency_ms']:.0f}ms")

    lines += [
        "",
        "## DeepRAG 建议",
        "",
        "| 场景 | 建议 | 依据 |",
        "|------|------|------|",
        "| 极速演示 | Cerebras GPT-OSS-120B / Groq 8B | 延迟低、吞吐高 |",
        "| 中文默认 | Silicon GLM-Z1 / Zhipu Flash / Groq Qwen | 中文题更稳 |",
        "| 降级链 | Groq → Cerebras → Silicon → Zhipu | 多 Key 冗余 |",
        "",
        "## 历史参考（2026-07-15 v2 报告）",
        "",
        "旧报告见 `docs/free_model_benchmark_report_v2.md`：Groq Qwen3-32B 曾综合最高，但模型 ID 已变更；以本 v3 实测为准。",
        "",
    ]

    md = "\n".join(lines)
    OUT_MD.write_text(md, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"time": datetime.now().isoformat(), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n======== RANKING ========")
    for i, r in enumerate(results, 1):
        log(f"{i}. {r['name']:28} score={r['avg_score']:5} {r['success']}/{r['total']} {r['avg_latency_ms']:7.0f}ms")
    log(f"MD: {OUT_MD}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
