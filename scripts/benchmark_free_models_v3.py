#!/usr/bin/env python3
"""免费 LLM 模型横向对比 v3

- 密钥仅从 .env / 环境变量读取（禁止写死）
- 覆盖：Groq / Cerebras / SiliconFlow / 智谱 / OpenRouter 免费档
- 维度：成功率、延迟、吞吐、关键词准确、完整性、指令遵循、中文、代码
- 输出：docs/free_model_benchmark_report_v3.md + data json

用法：
  cd deep-rag
  .venv\\Scripts\\python.exe scripts/benchmark_free_models_v3.py
  .venv\\Scripts\\python.exe scripts/benchmark_free_models_v3.py --limit-models 4 --limit-q 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 加载 .env
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


QUESTIONS = [
    {
        "id": "q_rag",
        "cat": "知识",
        "q": "什么是RAG（检索增强生成）？相对纯LLM有什么优势？请分点说明。",
        "kw": ["检索", "生成", "知识", "幻觉"],
    },
    {
        "id": "q_math",
        "cat": "数学",
        "q": "求解 2x+5=17，给出步骤并给出最终 x=？",
        "kw": ["x", "6", "12"],
    },
    {
        "id": "q_logic",
        "cat": "逻辑",
        "q": "小明有5个苹果，给了小红2个，又买了3个，最后有几个？逐步推理。",
        "kw": ["6", "5", "2", "3"],
    },
    {
        "id": "q_code",
        "cat": "代码",
        "q": "用Python写函数 is_prime(n: int) -> bool 判断素数，含类型注解与简短docstring。",
        "kw": ["def", "is_prime", "return", "int", "bool"],
    },
    {
        "id": "q_zh",
        "cat": "中文",
        "q": "解释成语「画蛇添足」，并给一个生活中的例子。",
        "kw": ["多余", "蛇", "足"],
    },
    {
        "id": "q_json",
        "cat": "抽取",
        "q": "从文本提取人名地点时间，只输出JSON：'2025年3月15日，张三在北京参加了人工智能大会。'",
        "kw": ["张三", "北京", "2025"],
    },
    {
        "id": "q_compare",
        "cat": "对比",
        "q": "比较 Python 与 JavaScript 至少 4 点区别，用列表。",
        "kw": ["Python", "JavaScript", "类型", "运行"],
    },
    {
        "id": "q_debug",
        "cat": "排错",
        "q": "Python 报 IndentationError: unexpected indent，最常见原因与修复方法？",
        "kw": ["缩进", "空格", "Tab"],
    },
]


def env_key(*names: str) -> str:
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v:
            return v
    return ""


def build_models() -> List[dict]:
    """根据已配置 Key 组装免费/准免费模型列表。"""
    models: List[dict] = []

    # 模型 ID 以各平台 /models 实测为准（2026-07-18）
    # 每家 2 个代表模型，请求间隔拉大，避免免费 RPM 打爆
    groq = env_key("GROQ_API_KEY")
    if groq:
        for mid, name, sleep in [
            ("llama-3.1-8b-instant", "Groq Llama3.1-8B", 8.0),
            ("llama-3.3-70b-versatile", "Groq Llama3.3-70B", 10.0),
            ("qwen/qwen3.6-27b", "Groq Qwen3.6-27B", 10.0),
        ]:
            models.append(
                {
                    "provider": "Groq",
                    "model_id": mid,
                    "name": name,
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key": groq,
                    "sleep": sleep,
                    "tier": "free",
                }
            )

    cerebras = env_key("CEREBRAS_API_KEY")
    if cerebras:
        for mid, name, sleep in [
            ("gpt-oss-120b", "Cerebras GPT-OSS-120B", 12.0),
            ("gemma-4-31b", "Cerebras Gemma4-31B", 12.0),
        ]:
            models.append(
                {
                    "provider": "Cerebras",
                    "model_id": mid,
                    "name": name,
                    "base_url": "https://api.cerebras.ai/v1",
                    "api_key": cerebras,
                    "sleep": sleep,
                    "tier": "free",
                }
            )

    sf = env_key("SILICONFLOW_API_KEY")
    if sf:
        for mid, name, sleep in [
            ("THUDM/GLM-Z1-9B-0414", "Silicon GLM-Z1-9B", 3.0),
            ("Qwen/Qwen2.5-7B-Instruct", "Silicon Qwen2.5-7B", 3.0),
            ("Qwen/Qwen3-8B", "Silicon Qwen3-8B", 3.0),
        ]:
            models.append(
                {
                    "provider": "SiliconFlow",
                    "model_id": mid,
                    "name": name,
                    "base_url": "https://api.siliconflow.cn/v1",
                    "api_key": sf,
                    "sleep": sleep,
                    "tier": "free/quota",
                }
            )

    zhipu = env_key("ZHIPU_API_KEY")
    if zhipu:
        for mid, name, sleep in [
            ("glm-4.5-flash", "Zhipu GLM-4.5-Flash", 3.0),
            ("glm-4-flash", "Zhipu GLM-4-Flash", 3.0),
        ]:
            models.append(
                {
                    "provider": "Zhipu",
                    "model_id": mid,
                    "name": name,
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "api_key": zhipu,
                    "sleep": sleep,
                    "tier": "free/quota",
                    "path": "/chat/completions",
                }
            )

    orouter = env_key("OPENROUTER_API_KEY")
    if orouter:
        for mid, name, sleep in [
            ("openai/gpt-oss-20b:free", "OR GPT-OSS-20B free", 12.0),
            ("meta-llama/llama-3.3-70b-instruct:free", "OR Llama3.3-70B free", 12.0),
            ("qwen/qwen3-coder:free", "OR Qwen3-Coder free", 12.0),
        ]:
            models.append(
                {
                    "provider": "OpenRouter",
                    "model_id": mid,
                    "name": name,
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key": orouter,
                    "sleep": sleep,
                    "tier": "free",
                    "extra_headers": {
                        "HTTP-Referer": "https://localhost/deeprag-bench",
                        "X-Title": "DeepRAG-Free-Bench",
                    },
                }
            )

    return models


def call_chat(m: dict, prompt: str, max_retries: int = 4) -> dict:
    headers = {
        "Authorization": f"Bearer {m['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(m.get("extra_headers") or {})
    url = m["base_url"].rstrip("/") + (m.get("path") or "/chat/completions")
    payload = {
        "model": m["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False,
    }

    last_err = ""
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            lat = (time.time() - t0) * 1000
            if r.status_code == 200:
                data = r.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                    or ""
                )
                usage = data.get("usage") or {}
                tokens = int(
                    usage.get("completion_tokens")
                    or usage.get("total_tokens")
                    or max(1, int(len(content) * 0.6))
                )
                speed = tokens / (lat / 1000) if lat > 0 else 0
                return {
                    "ok": True,
                    "content": content.strip(),
                    "latency_ms": round(lat, 1),
                    "tokens": tokens,
                    "speed": round(speed, 1),
                    "error": "",
                }
            if r.status_code == 429:
                # free 档 RPM 很紧；指数退避
                wait = 30 * (attempt + 1)
                last_err = f"429 {r.text[:120]}"
                print(f"      429, wait {wait}s retry {attempt+1}/{max_retries}", flush=True)
                time.sleep(wait)
                continue
            last_err = f"HTTP {r.status_code}: {r.text[:160]}"
            # 模型不存在等不重试
            if r.status_code in (400, 404):
                break
            time.sleep(1.5)
        except Exception as e:
            last_err = str(e)[:160]
            time.sleep(1.0)
    return {
        "ok": False,
        "content": "",
        "latency_ms": 0.0,
        "tokens": 0,
        "speed": 0.0,
        "error": last_err,
    }


def score_answer(q: dict, text: str, latency_ms: float, ok: bool) -> dict:
    if not ok or not text:
        return {
            "keyword": 0.0,
            "length": 0.0,
            "structure": 0.0,
            "zh": 0.0,
            "code": 0.0,
            "instruction": 0.0,
            "latency_score": 0.0,
            "total": 0.0,
        }

    kws = q.get("kw") or []
    hits = sum(1 for k in kws if k.lower() in text.lower())
    keyword = (hits / max(1, len(kws))) * 10

    n = len(text)
    if 80 <= n <= 1200:
        length = 9.0
    elif 40 <= n < 80 or 1200 < n <= 2000:
        length = 7.0
    else:
        length = 4.0

    structure = 8.0 if any(x in text for x in ("\n", "1.", "-", "：", ":")) else 5.0
    zh_chars = sum(1 for c in text if "一" <= c <= "鿿")
    zh = min(10.0, 4.0 + zh_chars / max(1, n) * 10)

    code = 5.0
    if q["cat"] == "代码":
        code = 0.0
        for tip in ("def ", "return", "int", "bool", '"""', "'''"):
            if tip in text:
                code += 2.0
        code = min(10.0, code)

    instruction = 8.0
    if q["cat"] == "抽取":
        instruction = 9.0 if ("{" in text and "}" in text) else 3.0
    if q["cat"] == "数学" and ("x=6" in text.replace(" ", "") or "x = 6" in text):
        instruction = 10.0

    # 延迟：越低越好
    if latency_ms <= 800:
        lat_s = 10.0
    elif latency_ms <= 1500:
        lat_s = 8.0
    elif latency_ms <= 3000:
        lat_s = 6.0
    elif latency_ms <= 8000:
        lat_s = 4.0
    else:
        lat_s = 2.0

    total = (
        keyword * 0.35
        + length * 0.12
        + structure * 0.10
        + zh * 0.12
        + code * 0.10
        + instruction * 0.11
        + lat_s * 0.10
    )
    return {
        "keyword": round(keyword, 2),
        "length": round(length, 2),
        "structure": round(structure, 2),
        "zh": round(zh, 2),
        "code": round(code, 2),
        "instruction": round(instruction, 2),
        "latency_score": round(lat_s, 2),
        "total": round(total, 2),
    }


@dataclass
class ModelResult:
    name: str
    provider: str
    model_id: str
    tier: str
    success: int = 0
    total_q: int = 0
    avg_latency_ms: float = 0.0
    avg_speed: float = 0.0
    avg_score: float = 0.0
    details: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def run_benchmark(models: List[dict], questions: List[dict]) -> List[ModelResult]:
    out: List[ModelResult] = []
    for mi, m in enumerate(models, 1):
        print(f"\n=== [{mi}/{len(models)}] {m['name']} ({m['model_id']}) ===", flush=True)
        mr = ModelResult(
            name=m["name"],
            provider=m["provider"],
            model_id=m["model_id"],
            tier=m.get("tier", "free"),
        )
        lats, speeds, scores = [], [], []
        for qi, q in enumerate(questions, 1):
            print(f"  ({qi}/{len(questions)}) {q['id']} ...", end=" ", flush=True)
            resp = call_chat(m, q["q"])
            sc = score_answer(q, resp.get("content") or "", resp.get("latency_ms") or 0, resp.get("ok"))
            row = {
                "qid": q["id"],
                "cat": q["cat"],
                "ok": resp["ok"],
                "latency_ms": resp["latency_ms"],
                "tokens": resp["tokens"],
                "speed": resp["speed"],
                "scores": sc,
                "answer_preview": (resp.get("content") or "")[:280],
                "error": resp.get("error") or "",
            }
            mr.details.append(row)
            if resp["ok"]:
                mr.success += 1
                lats.append(resp["latency_ms"])
                speeds.append(resp["speed"])
                scores.append(sc["total"])
                print(f"OK {resp['latency_ms']:.0f}ms score={sc['total']}", flush=True)
            else:
                mr.errors.append(f"{q['id']}: {resp.get('error')}")
                print(f"FAIL {resp.get('error')[:80]}", flush=True)
            time.sleep(float(m.get("sleep") or 1.5))
        mr.total_q = len(questions)
        mr.avg_latency_ms = round(statistics.mean(lats), 1) if lats else 0.0
        mr.avg_speed = round(statistics.mean(speeds), 1) if speeds else 0.0
        # 综合分 0-10：质量均值 × 成功率（失败题按 0 计入）
        if scores:
            # 失败题补 0，保证成功率体现在分数里且不超过 10
            padded = scores + [0.0] * (mr.total_q - len(scores))
            mr.avg_score = round(statistics.mean(padded), 2)
        else:
            mr.avg_score = 0.0
        # 模型之间额外冷却，降低跨模型 429
        time.sleep(20.0)
        out.append(mr)
        print(
            f"  >> success={mr.success}/{mr.total_q} lat={mr.avg_latency_ms}ms "
            f"speed={mr.avg_speed} score={mr.avg_score}",
            flush=True,
        )
    out.sort(key=lambda x: (-x.avg_score, x.avg_latency_ms))
    return out


def to_md(results: List[ModelResult], meta: dict) -> str:
    lines = [
        "# 免费 LLM 模型横向对比报告 v3",
        "",
        f"- 时间：{meta['time']}",
        f"- 题目数：{meta['n_questions']}",
        f"- 模型数：{len(results)}（有 Key 才测）",
        f"- 评分：关键词+结构+中文+代码+指令+延迟 规则启发式；**非人工金标**",
        "",
        "## 综合排名",
        "",
        "| 排名 | 模型 | 厂商 | 综合分 | 成功率 | 平均延迟 | 平均速度 |",
        "|-----:|------|------|-------:|-------:|---------:|---------:|",
    ]
    for i, r in enumerate(results, 1):
        rate = f"{r.success}/{r.total_q}"
        lines.append(
            f"| {i} | {r.name} | {r.provider} | **{r.avg_score}** | {rate} | "
            f"{r.avg_latency_ms:.0f}ms | {r.avg_speed:.0f} tok/s |"
        )

    lines += ["", "## 分厂商最佳", ""]
    by_p: Dict[str, ModelResult] = {}
    for r in results:
        if r.success == 0:
            continue
        if r.provider not in by_p or r.avg_score > by_p[r.provider].avg_score:
            by_p[r.provider] = r
    for p, r in sorted(by_p.items(), key=lambda kv: -kv[1].avg_score):
        lines.append(f"- **{p}**: {r.name} · 分 {r.avg_score} · {r.avg_latency_ms:.0f}ms")

    lines += ["", "## 维度拆解（成功题均值）", ""]
    lines.append("| 模型 | 关键词 | 结构 | 中文 | 指令 | 延迟分 |")
    lines.append("|------|-------:|-----:|-----:|-----:|-------:|")
    for r in results:
        ok_rows = [d for d in r.details if d["ok"]]
        if not ok_rows:
            lines.append(f"| {r.name} | - | - | - | - | - |")
            continue

        def avg(k):
            return round(statistics.mean(d["scores"][k] for d in ok_rows), 2)

        lines.append(
            f"| {r.name} | {avg('keyword')} | {avg('structure')} | {avg('zh')} | "
            f"{avg('instruction')} | {avg('latency_score')} |"
        )

    lines += ["", "## 失败/限流摘要", ""]
    any_err = False
    for r in results:
        if r.errors:
            any_err = True
            lines.append(f"### {r.name}")
            for e in r.errors[:6]:
                lines.append(f"- `{e[:180]}`")
    if not any_err:
        lines.append("- 无明显失败")

    lines += [
        "",
        "## 使用建议（DeepRAG）",
        "",
        "| 场景 | 建议 |",
        "|------|------|",
        "| 极速演示 | Cerebras / Groq 小模型 |",
        "| 中文质量 | Groq Qwen / 硅基 GLM / 智谱 Flash |",
        "| 稳定默认 | Silicon GLM-Z1-9B 或 Zhipu Flash |",
        "| 降级链路 | Groq → Cerebras → Silicon → Zhipu |",
        "",
        "> 免费额度/RPM 会变；以当次实测为准。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-models", type=int, default=0)
    ap.add_argument("--limit-q", type=int, default=0)
    ap.add_argument(
        "--out-prefix",
        default=str(ROOT / "docs" / "free_model_benchmark_v3"),
    )
    args = ap.parse_args()

    models = build_models()
    if not models:
        print("ERROR: 未检测到任何 API Key（GROQ/CEREBRAS/SILICONFLOW/ZHIPU/OPENROUTER）")
        return 2
    if args.limit_models > 0:
        models = models[: args.limit_models]
    questions = QUESTIONS
    if args.limit_q > 0:
        questions = QUESTIONS[: args.limit_q]

    print(f"Models to test: {len(models)}")
    for m in models:
        print(f"  - {m['provider']}: {m['name']}")
    print(f"Questions: {len(questions)}")

    results = run_benchmark(models, questions)
    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "n_models": len(models),
    }
    payload = {
        "meta": meta,
        "ranking": [
            {
                "rank": i + 1,
                **{k: getattr(r, k) for k in ("name", "provider", "model_id", "tier", "success", "total_q", "avg_latency_ms", "avg_speed", "avg_score")},
                "details": r.details,
                "errors": r.errors,
            }
            for i, r in enumerate(results)
        ],
    }

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(prefix) + ".json")
    md_path = Path(str(prefix) + ".md")
    # also latest aliases
    latest_json = ROOT / "docs" / "free_model_benchmark_data_v3.json"
    latest_md = ROOT / "docs" / "free_model_benchmark_report_v3.md"

    data = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(data, encoding="utf-8")
    latest_json.write_text(data, encoding="utf-8")
    md = to_md(results, meta)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")

    print("\n======== RANKING ========", flush=True)
    for i, r in enumerate(results, 1):
        print(
            f"{i}. {r.name:28} score={r.avg_score:5}  "
            f"{r.success}/{r.total_q}  {r.avg_latency_ms:7.0f}ms  {r.avg_speed:6.0f}tok/s",
            flush=True,
        )
    print(f"\nJSON: {latest_json}")
    print(f"MD:   {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
