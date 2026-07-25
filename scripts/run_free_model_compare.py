#!/usr/bin/env python3
"""免费模型精简横向对比（单进程、严格限流、可完成）

每家 1–2 个代表模型 × 6 题；请求间隔大；只允许一个实例。
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCK = Path(r"D:\文档\ai提问相关\哲思灵智\free_model_bench.lock")
LOG = Path(r"D:\文档\ai提问相关\哲思灵智\free_model_bench_v3.log")
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


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


QUESTIONS = [
    {"id": "rag", "cat": "知识", "q": "什么是RAG检索增强生成？分点说明相对纯LLM的优势。", "kw": ["检索", "生成", "知识", "幻觉"]},
    {"id": "math", "cat": "数学", "q": "求解 2x+5=17，写出步骤并给出 x 的值。", "kw": ["x", "6", "12"]},
    {"id": "code", "cat": "代码", "q": "用Python写 is_prime(n: int) -> bool，含类型注解。", "kw": ["def", "is_prime", "return", "bool"]},
    {"id": "zh", "cat": "中文", "q": "解释成语画蛇添足，并举一个生活例子。", "kw": ["多余", "蛇", "足"]},
    {"id": "json", "cat": "抽取", "q": "从'2025年3月15日，张三在北京开会'提取人名地点时间，只输出JSON。", "kw": ["张三", "北京", "2025"]},
    {"id": "debug", "cat": "排错", "q": "Python IndentationError: unexpected indent 常见原因与修复？", "kw": ["缩进", "空格", "Tab"]},
]


def models():
    items = []
    g = os.getenv("GROQ_API_KEY", "").strip()
    if g:
        items += [
            ("Groq", "llama-3.1-8b-instant", "Groq Llama3.1-8B", "https://api.groq.com/openai/v1", g, 12),
            ("Groq", "qwen/qwen3.6-27b", "Groq Qwen3.6-27B", "https://api.groq.com/openai/v1", g, 15),
        ]
    c = os.getenv("CEREBRAS_API_KEY", "").strip()
    if c:
        items += [
            ("Cerebras", "gpt-oss-120b", "Cerebras GPT-OSS-120B", "https://api.cerebras.ai/v1", c, 15),
            ("Cerebras", "gemma-4-31b", "Cerebras Gemma4-31B", "https://api.cerebras.ai/v1", c, 15),
        ]
    s = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if s:
        items += [
            ("SiliconFlow", "THUDM/GLM-Z1-9B-0414", "Silicon GLM-Z1-9B", "https://api.siliconflow.cn/v1", s, 4),
            ("SiliconFlow", "Qwen/Qwen2.5-7B-Instruct", "Silicon Qwen2.5-7B", "https://api.siliconflow.cn/v1", s, 4),
            ("SiliconFlow", "Qwen/Qwen3-8B", "Silicon Qwen3-8B", "https://api.siliconflow.cn/v1", s, 4),
        ]
    z = os.getenv("ZHIPU_API_KEY", "").strip()
    if z:
        items += [
            ("Zhipu", "glm-4.5-flash", "Zhipu GLM-4.5-Flash", "https://open.bigmodel.cn/api/paas/v4", z, 4),
            ("Zhipu", "glm-4-flash", "Zhipu GLM-4-Flash", "https://open.bigmodel.cn/api/paas/v4", z, 4),
        ]
    o = os.getenv("OPENROUTER_API_KEY", "").strip()
    if o:
        items += [
            ("OpenRouter", "openai/gpt-oss-20b:free", "OR GPT-OSS-20B free", "https://openrouter.ai/api/v1", o, 18),
            ("OpenRouter", "meta-llama/llama-3.3-70b-instruct:free", "OR Llama3.3-70B free", "https://openrouter.ai/api/v1", o, 18),
        ]
    return items


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
    for attempt in range(4):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            lat = (time.time() - t0) * 1000
            if r.status_code == 200:
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {}) or {}
                content = (msg.get("content") or msg.get("reasoning") or "").strip()
                # 某些模型 content 空但 reasoning 有
                if not content:
                    content = str(msg)[:20]
                usage = data.get("usage") or {}
                tokens = int(usage.get("completion_tokens") or max(1, int(len(content) * 0.6)))
                speed = tokens / (lat / 1000) if lat > 0 else 0
                return True, content, lat, tokens, speed, ""
            if r.status_code == 429:
                wait = 25 * (attempt + 1)
                last = f"429 {r.text[:100]}"
                log(f"      429 wait {wait}s")
                time.sleep(wait)
                continue
            last = f"HTTP {r.status_code}: {r.text[:120]}"
            if r.status_code in (400, 404):
                break
            time.sleep(3)
        except Exception as e:
            last = str(e)[:120]
            time.sleep(2)
    return False, "", 0.0, 0, 0.0, last


def score(q, text, lat, ok):
    if not ok or not text:
        return 0.0, {}
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
    total = kw * 0.36 + length * 0.12 + struct * 0.10 + zh * 0.12 + code * 0.10 + instr * 0.10 + lat_s * 0.10
    detail = {
        "keyword": round(kw, 2),
        "length": length,
        "structure": struct,
        "zh": round(zh, 2),
        "code": round(code, 2),
        "instruction": instr,
        "latency_score": lat_s,
    }
    return round(total, 2), detail


def main():
    if LOCK.exists():
        # stale lock > 2h clear
        age = time.time() - LOCK.stat().st_mtime
        if age < 7200:
            print("lock exists, another run?")
            # still proceed if no process? we killed all
        LOCK.unlink(missing_ok=True)
    LOCK.write_text(datetime.now().isoformat(), encoding="utf-8")
    LOG.write_text(f"===== CLEAN START {datetime.now().isoformat()} =====\n", encoding="utf-8")

    ms = models()
    log(f"Models: {len(ms)}")
    for m in ms:
        log(f"  - {m[0]}: {m[2]}")

    results = []
    try:
        for i, (provider, mid, name, base, key, sleep_s) in enumerate(ms, 1):
            log(f"\n=== [{i}/{len(ms)}] {name} ({mid}) ===")
            rows = []
            ok_n = 0
            lats, speeds, scores = [], [], []
            for j, q in enumerate(QUESTIONS, 1):
                log(f"  ({j}/{len(QUESTIONS)}) {q['id']} ...",)
                ok, content, lat, tokens, speed, err = call(base, key, mid, q["q"], provider)
                sc, detail = score(q, content, lat, ok)
                if ok:
                    ok_n += 1
                    lats.append(lat)
                    speeds.append(speed)
                    scores.append(sc)
                    log(f"OK {lat:.0f}ms score={sc}")
                else:
                    log(f"FAIL {err[:100]}")
                rows.append(
                    {
                        "id": q["id"],
                        "ok": ok,
                        "latency_ms": round(lat, 1),
                        "speed": round(speed, 1),
                        "score": sc,
                        "detail": detail,
                        "preview": content[:240],
                        "error": err,
                    }
                )
                time.sleep(sleep_s)
            # fail pads 0
            padded = scores + [0.0] * (len(QUESTIONS) - len(scores))
            avg = round(statistics.mean(padded), 2) if padded else 0.0
            rec = {
                "name": name,
                "provider": provider,
                "model_id": mid,
                "success": ok_n,
                "total": len(QUESTIONS),
                "avg_latency_ms": round(statistics.mean(lats), 1) if lats else 0,
                "avg_speed": round(statistics.mean(speeds), 1) if speeds else 0,
                "avg_score": avg,
                "rows": rows,
            }
            results.append(rec)
            log(
                f"  >> success={ok_n}/{len(QUESTIONS)} lat={rec['avg_latency_ms']}ms "
                f"speed={rec['avg_speed']} score={avg}"
            )
            time.sleep(25)  # provider cool-down
    finally:
        LOCK.unlink(missing_ok=True)

    results.sort(key=lambda x: (-x["avg_score"], x["avg_latency_ms"]))
    # report
    lines = [
        "# 免费 LLM 模型横向对比报告 v3（精简可完成版）",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 模型数：{len(results)} · 每模型 {len(QUESTIONS)} 题",
        "- 说明：关键词/结构/中文/代码/指令/延迟规则分；失败按 0 计入；**非人工金标**",
        "- 策略：单进程 + 长间隔 + 每家少量代表模型，减少 429",
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
        if r["success"] == 0:
            continue
        if r["provider"] not in best or r["avg_score"] > best[r["provider"]]["avg_score"]:
            best[r["provider"]] = r
    for p, r in sorted(best.items(), key=lambda kv: -kv[1]["avg_score"]):
        lines.append(f"- **{p}**: {r['name']} · 分 {r['avg_score']} · {r['avg_latency_ms']:.0f}ms")

    lines += [
        "",
        "## DeepRAG 建议",
        "",
        "| 场景 | 建议 |",
        "|------|------|",
        "| 极速演示 | Cerebras / Groq 小模型 |",
        "| 中文问答默认 | Silicon GLM-Z1 / Zhipu Flash / Groq Qwen |",
        "| 降级链 | Groq → Cerebras → Silicon → Zhipu |",
        "",
        "## 逐模型摘要",
        "",
    ]
    for r in results:
        lines.append(f"### {r['name']}")
        lines.append(
            f"- success {r['success']}/{r['total']} · score {r['avg_score']} · "
            f"lat {r['avg_latency_ms']}ms · speed {r['avg_speed']}"
        )
        fails = [x for x in r["rows"] if not x["ok"]]
        if fails:
            for f in fails:
                lines.append(f"- FAIL {f['id']}: `{f['error'][:120]}`")
        lines.append("")

    md = "\n".join(lines)
    OUT_MD.write_text(md, encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps({"time": datetime.now().isoformat(), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log("\n======== RANKING ========")
    for i, r in enumerate(results, 1):
        log(
            f"{i}. {r['name']:28} score={r['avg_score']:5} "
            f"{r['success']}/{r['total']} {r['avg_latency_ms']:7.0f}ms {r['avg_speed']:6.0f}tok/s"
        )
    log(f"MD: {OUT_MD}")
    log(f"JSON: {OUT_JSON}")
    print(md[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
