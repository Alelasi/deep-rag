#!/usr/bin/env python3
"""DeepRAG 多维度实测评分（走真实 src.graph.query）

维度（约 30+ 项汇总）：
  准确率/关键词命中、完整性、相关性、引用质量、响应速度、
  幻觉分、事实校验通过率、拒识正确率、no_knowledge、
  检索相关文档数、延迟分位、分类/难度拆分、错误率等。

用法：
  cd deep-rag
  set QDRANT_MODE=server
  .venv\\Scripts\\python.exe evaluation/run_multidim_eval.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QDRANT_MODE", "server")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REFUSE_MARKERS = (
    "未找到可靠依据",
    "无法基于证据",
    "知识库与外部检索均未找到",
    "无法回答",
    "没有相关",
    "未找到与",
    "不足以回答",
    "无法从知识库",
)

# 免费联网回退：DuckDuckGo 检索 + 项目 get_llm（Groq/Cerebras/硅基/智谱等）
# 不用 CPA grok-4.5（高成本高精度，留给人工/专项；评测默认走免费通道）


def free_web_answer(question: str, timeout: int = 90, max_results: int = 5) -> dict:
    """本地库无答案时：免费搜索 + 免费 LLM 总结。

    搜索：src.retrieval.web_fallback.web_search_fallback（默认 DuckDuckGo，无需 Key）
    生成：src.config.get_llm / get_llm_with_fallback（.env 里十几个免费模型链路）
    """
    # 1) 免费网页检索
    hits: List[dict] = []
    engine = os.environ.get("EVAL_WEB_ENGINE") or "duckduckgo"
    try:
        from src.retrieval.web_fallback import web_search_fallback

        hits = web_search_fallback(question, max_results=max_results, engine=engine) or []
    except Exception as e:
        hits = []
        search_err = str(e)
    else:
        search_err = None

    # 过滤 mock 假结果（无真实 URL 时也允许仅 LLM 常识答，但标记）
    real_hits = []
    for h in hits:
        meta = h.get("metadata") or {}
        if meta.get("engine") == "mock" or str(h.get("doc_id", "")).startswith("web_mock"):
            continue
        real_hits.append(h)

    snippets = []
    urls = []
    for i, h in enumerate(real_hits[:max_results], 1):
        title = (h.get("metadata") or {}).get("title") or ""
        body = (h.get("content") or h.get("metadata", {}).get("snippet") or "").strip()
        src = h.get("source") or ""
        if src:
            urls.append(src)
        snippets.append(f"[{i}] {title}\n{body[:500]}\n来源: {src}")

    context = "\n\n".join(snippets) if snippets else "（未检索到可用网页片段）"

    # 2) 免费 LLM 总结（走项目已有多后端）
    try:
        from src.config import get_llm, get_llm_with_fallback
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as e:
        return {"ok": False, "error": f"import_llm_failed: {e}", "answer": "", "urls": urls}

    llm = None
    try:
        llm = get_llm_with_fallback(temperature=0.2)
    except Exception:
        try:
            llm = get_llm(temperature=0.2)
        except Exception as e:
            return {"ok": False, "error": f"get_llm_failed: {e}", "answer": "", "urls": urls}

    if llm is None:
        return {"ok": False, "error": "llm_unavailable", "answer": "", "urls": urls}

    sys_msg = (
        "你是严谨的学科助教。根据提供的【网页检索片段】回答问题；"
        "片段不足时，可补充公认基础知识，但必须标明哪些来自检索、哪些是常识。"
        "用简明中文，包含关键术语；尽量列出引用编号与URL。"
        "若完全无法作答，只输出 SEARCH_FAILED。"
    )
    user_msg = (
        f"问题：{question}\n\n"
        f"【网页检索片段】\n{context}\n\n"
        "请作答。"
    )

    try:
        # 部分后端不支持 timeout 参数，用线程超时由外层控制
        resp = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
        if isinstance(resp, str):
            answer = resp.strip()
        else:
            answer = (getattr(resp, "content", None) or str(resp)).strip()
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__
        if not answer or answer.strip() == "SEARCH_FAILED":
            return {
                "ok": False,
                "error": search_err or "empty_or_search_failed",
                "answer": answer,
                "urls": urls,
                "hits": len(real_hits),
                "raw_model": model_name,
            }
        return {
            "ok": True,
            "answer": answer,
            "urls": urls,
            "hits": len(real_hits),
            "raw_model": model_name,
            "source": "free_web_llm",
            "engine": engine,
            "search_error": search_err,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "answer": "", "urls": urls, "hits": len(real_hits)}


def _need_web_fallback(case: dict, result: dict) -> bool:
    """本地无依据/拒答时启用联网回退；拒识金标题永不回退。"""
    if case.get("expect_refuse"):
        return False
    if case.get("web_fallback") is False:
        return False
    answer = (result.get("answer") or "") if result else ""
    no_knowledge = bool(result.get("no_knowledge")) if result else True
    refuse = _is_refuse(answer, no_knowledge)
    if no_knowledge or refuse or not answer.strip():
        return True
    return False


def _kw_hits(answer: str, keywords: List[str]) -> tuple[int, float]:
    if not keywords:
        return 0, 1.0
    text = answer or ""
    hits = sum(1 for k in keywords if k and k.lower() in text.lower())
    return hits, hits / len(keywords)


def _is_refuse(answer: str, no_knowledge: bool) -> bool:
    if no_knowledge:
        return True
    a = answer or ""
    return any(m in a for m in REFUSE_MARKERS)


def _time_score(sec: float) -> float:
    if sec < 3:
        return 10.0
    if sec < 6:
        return 8.0
    if sec < 12:
        return 6.0
    if sec < 20:
        return 4.0
    return 2.0


def score_case(case: dict, result: dict, latency_s: float, err: str | None) -> dict:
    answer = (result.get("answer") or "") if result else ""
    keywords = case.get("expected_keywords") or []
    allow_refuse = bool(case.get("allow_refuse"))
    expect_refuse = bool(case.get("expect_refuse"))
    no_knowledge = bool(result.get("no_knowledge")) if result else True
    refuse = _is_refuse(answer, no_knowledge)
    hits, kw_rate = _kw_hits(answer, keywords)

    # pass/fail
    if expect_refuse:
        ok = refuse
    elif allow_refuse and refuse:
        ok = True
    else:
        ok = (kw_rate >= 0.34) and not (refuse and not allow_refuse)
        if keywords and hits == 0 and not allow_refuse:
            ok = False

    # 8 维 0-10
    accuracy = 10.0 if expect_refuse and refuse else round(min(10.0, kw_rate * 10), 2)
    if expect_refuse and not refuse:
        accuracy = 2.0

    length_score = min(10.0, len(answer) / 80.0)
    completeness = round(length_score * 0.45 + accuracy * 0.55, 2)

    q_tokens = [t for t in case["question"].replace("？", " ").replace("?", " ").split() if len(t) > 1]
    rel_hits = sum(1 for t in q_tokens if t in answer)
    relevance = round(min(10.0, (rel_hits / max(len(q_tokens), 1)) * 10 + (3 if kw_rate > 0 else 0)), 2)

    citations = result.get("citations") or []
    has_cite_mark = any(m in answer for m in ("[1]", "[2]", "[来源", "来源", "参考"))
    citation_quality = 9.0 if (citations or has_cite_mark) else (3.0 if answer else 0.0)

    hall_sys = result.get("hallucination_score")
    try:
        hall_sys_f = float(hall_sys) if hall_sys is not None else None
    except Exception:
        hall_sys_f = None
    uncertainty = sum(1 for w in ("可能", "也许", "大概", "不确定") if w in answer)
    hall_heuristic = min(10.0, uncertainty * 2.0 + (0 if hall_sys_f is None else hall_sys_f * 10))
    # 越低越好；转换为“反幻觉分”
    anti_hall = round(max(0.0, 10.0 - hall_heuristic), 2)

    format_score = 8.0 if any(x in answer for x in ("\n", "。", "：", "【")) else (5.0 if answer else 0.0)
    fluency = 8.5 if len(answer) > 60 else (6.0 if len(answer) > 20 else 3.0)
    time_s = _time_score(latency_s)

    weights = {
        "accuracy": 0.28,
        "completeness": 0.16,
        "relevance": 0.12,
        "citation_quality": 0.10,
        "time": 0.10,
        "anti_hall": 0.12,
        "format": 0.06,
        "fluency": 0.06,
    }
    total = (
        accuracy * weights["accuracy"]
        + completeness * weights["completeness"]
        + relevance * weights["relevance"]
        + citation_quality * weights["citation_quality"]
        + time_s * weights["time"]
        + anti_hall * weights["anti_hall"]
        + format_score * weights["format"]
        + fluency * weights["fluency"]
    )

    graded = result.get("graded_docs") or result.get("retrieved_docs") or []
    return {
        "id": case["id"],
        "question": case["question"],
        "category": case.get("category"),
        "difficulty": case.get("difficulty"),
        "collection": case.get("collection"),
        "ok": bool(ok),
        "expect_refuse": expect_refuse,
        "is_refuse": refuse,
        "keyword_hits": hits,
        "keyword_rate": round(kw_rate, 4),
        "accuracy": accuracy,
        "completeness": completeness,
        "relevance": relevance,
        "citation_quality": citation_quality,
        "time_score": time_s,
        "anti_hallucination": anti_hall,
        "format_score": format_score,
        "fluency": fluency,
        "total_score": round(total, 2),
        "latency_s": round(latency_s, 3),
        "answer_len": len(answer),
        "no_knowledge": no_knowledge,
        "fact_check_passed": bool(result.get("fact_check_passed")) if result else False,
        "hallucination_score": hall_sys_f,
        "citation_count": len(citations) if isinstance(citations, list) else 0,
        "retrieved_docs": len(graded) if isinstance(graded, list) else 0,
        "relevant_count": result.get("relevant_count"),
        "retrieval_decision": result.get("retrieval_decision"),
        "used_mock_web": bool(result.get("used_mock_web")) if result else False,
        "used_web_fallback": bool(
            result.get("used_web_fallback") or result.get("used_cpa_web")
        )
        if result
        else False,
        "used_cpa_web": bool(result.get("used_cpa_web")) if result else False,
        "answer_source": result.get("answer_source") or ("local_kb" if result else "none"),
        "regenerate_count": result.get("regenerate_count") or 0,
        "routed_collection": result.get("routed_collection") or result.get("collection_name"),
        "answer_preview": answer[:220],
        "answer": answer,  # 完整回答，便于逐题审阅
        "error": err,
        "discipline": case.get("discipline"),
        "subdiscipline": case.get("subdiscipline"),
    }


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def aggregate(rows: List[dict]) -> dict:
    n = len(rows) or 1
    oks = [r for r in rows if r["ok"]]
    lats = [r["latency_s"] for r in rows]
    totals = [r["total_score"] for r in rows]
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    by_diff: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category") or "na"].append(r)
        by_diff[r.get("difficulty") or "na"].append(r)

    def avg(key: str) -> float:
        return round(sum(r[key] for r in rows) / n, 3)

    dims = {
        "准确率_pass": round(len(oks) / n, 4),
        "综合分_mean": round(statistics.mean(totals), 3) if totals else 0,
        "综合分_median": round(statistics.median(totals), 3) if totals else 0,
        "关键词命中率_mean": avg("keyword_rate"),
        "准确性_mean": avg("accuracy"),
        "完整性_mean": avg("completeness"),
        "相关性_mean": avg("relevance"),
        "引用质量_mean": avg("citation_quality"),
        "速度分_mean": avg("time_score"),
        "反幻觉_mean": avg("anti_hallucination"),
        "格式_mean": avg("format_score"),
        "流畅度_mean": avg("fluency"),
        "事实校验通过率": round(sum(1 for r in rows if r["fact_check_passed"]) / n, 4),
        "no_knowledge率": round(sum(1 for r in rows if r["no_knowledge"]) / n, 4),
        "拒识题正确率": _refuse_acc(rows),
        "mock_web率": round(sum(1 for r in rows if r["used_mock_web"]) / n, 4),
        "联网回退率": round(
            sum(1 for r in rows if r.get("used_web_fallback") or r.get("used_cpa_web")) / n, 4
        ),
        "cpa_web回退率": round(sum(1 for r in rows if r.get("used_cpa_web")) / n, 4),
        "错误率": round(sum(1 for r in rows if r.get("error")) / n, 4),
        "空答率": round(sum(1 for r in rows if r["answer_len"] == 0) / n, 4),
        "有引用率": round(sum(1 for r in rows if r["citation_count"] > 0 or r["citation_quality"] >= 8) / n, 4),
        "延迟_mean_s": round(statistics.mean(lats), 3) if lats else 0,
        "延迟_p50_s": round(percentile(lats, 0.5), 3),
        "延迟_p90_s": round(percentile(lats, 0.9), 3),
        "延迟_max_s": round(max(lats), 3) if lats else 0,
        "3秒内占比": round(sum(1 for x in lats if x <= 3) / n, 4),
        "10秒内占比": round(sum(1 for x in lats if x <= 10) / n, 4),
        "20秒内占比": round(sum(1 for x in lats if x <= 20) / n, 4),
        "平均答案长度": round(sum(r["answer_len"] for r in rows) / n, 1),
        "平均检索文档数": round(sum(r["retrieved_docs"] for r in rows) / n, 2),
        "平均相关文档数": round(
            sum((r.get("relevant_count") or 0) for r in rows) / n, 2
        ),
        "系统幻觉分_mean": round(
            statistics.mean([r["hallucination_score"] for r in rows if r["hallucination_score"] is not None])
            if any(r["hallucination_score"] is not None for r in rows)
            else 0.0,
            4,
        ),
        "按类别pass": {
            k: round(sum(1 for r in v if r["ok"]) / len(v), 4) for k, v in sorted(by_cat.items())
        },
        "按类别综合分": {
            k: round(statistics.mean([r["total_score"] for r in v]), 3) for k, v in sorted(by_cat.items())
        },
        "按难度pass": {
            k: round(sum(1 for r in v if r["ok"]) / len(v), 4) for k, v in sorted(by_diff.items())
        },
        "按难度综合分": {
            k: round(statistics.mean([r["total_score"] for r in v]), 3) for k, v in sorted(by_diff.items())
        },
        "按知识库pass": _by_field(rows, "collection"),
    }
    # 等级
    pass_rate = dims["准确率_pass"]
    overall = dims["综合分_mean"]
    if pass_rate >= 0.85 and overall >= 7.5:
        grade = "A"
    elif pass_rate >= 0.7 and overall >= 6.5:
        grade = "B"
    elif pass_rate >= 0.55 and overall >= 5.5:
        grade = "C"
    else:
        grade = "D"
    dims["等级"] = grade
    dims["说明"] = (
        "关键词+规则启发式评分，含系统 hallucination/fact_check 字段；"
        "非人工金标，也非官方 RAGAS 全量；勿直接写简历 95%。"
    )
    return dims


def _refuse_acc(rows: List[dict]) -> float:
    ref = [r for r in rows if r.get("expect_refuse")]
    if not ref:
        return 0.0
    return round(sum(1 for r in ref if r["ok"]) / len(ref), 4)


def _by_field(rows: List[dict], field: str) -> dict:
    g: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        g[str(r.get(field) or "na")].append(r)
    return {k: round(sum(1 for r in v if r["ok"]) / len(v), 4) for k, v in sorted(g.items())}


def to_md(report: dict) -> str:
    m = report["metrics"]
    lines = [
        f"# DeepRAG 多维度实测报告",
        "",
        f"- 时间：{report['generated_at']}",
        f"- 样本数：{report['total']}",
        f"- 等级：**{m['等级']}**",
        f"- 准确率(pass)：**{m['准确率_pass']:.1%}**（{report['hit']}/{report['total']}）",
        f"- 综合分 mean/median：{m['综合分_mean']} / {m['综合分_median']}（满分 10）",
        f"- 说明：{m['说明']}",
        "",
        "## 核心指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    core_keys = [
        "准确率_pass",
        "综合分_mean",
        "关键词命中率_mean",
        "准确性_mean",
        "完整性_mean",
        "相关性_mean",
        "引用质量_mean",
        "速度分_mean",
        "反幻觉_mean",
        "事实校验通过率",
        "no_knowledge率",
        "拒识题正确率",
        "有引用率",
        "错误率",
        "空答率",
        "延迟_mean_s",
        "延迟_p50_s",
        "延迟_p90_s",
        "延迟_max_s",
        "3秒内占比",
        "10秒内占比",
        "20秒内占比",
        "系统幻觉分_mean",
        "平均检索文档数",
        "平均相关文档数",
        "平均答案长度",
        "mock_web率",
        "cpa_web回退率",
    ]
    for k in core_keys:
        v = m[k]
        if isinstance(v, float) and k.endswith(("率", "占比", "pass")):
            lines.append(f"| {k} | {v:.1%} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += ["", "## 分类 pass / 综合分", "", "| 类别 | pass | 综合分 |", "|---|---:|---:|"]
    for k, v in m["按类别pass"].items():
        lines.append(f"| {k} | {v:.1%} | {m['按类别综合分'].get(k, 0)} |")

    lines += ["", "## 难度", "", "| 难度 | pass | 综合分 |", "|---|---:|---:|"]
    for k, v in m["按难度pass"].items():
        lines.append(f"| {k} | {v:.1%} | {m['按难度综合分'].get(k, 0)} |")

    lines += ["", "## 知识库", "", "| collection | pass |", "|---|---:|"]
    for k, v in m["按知识库pass"].items():
        lines.append(f"| {k} | {v:.1%} |")

    lines += [
        "",
        "## 分题明细",
        "",
        "| id | ok | total | acc | lat_s | nk | fc | q |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for r in report["cases"]:
        lines.append(
            f"| {r['id']} | {r['ok']} | {r['total_score']} | {r['accuracy']} | "
            f"{r['latency_s']} | {r['no_knowledge']} | {r['fact_check_passed']} | {r['question'][:28]} |"
        )
    lines += ["", "## 失败样例（最多 12）", ""]
    fails = [r for r in report["cases"] if not r["ok"]][:12]
    if not fails:
        lines.append("- 无失败")
    for r in fails:
        lines.append(
            f"- **{r['id']}** `{r['question']}` total={r['total_score']} "
            f"kw={r['keyword_rate']} refuse={r['is_refuse']} preview={r['answer_preview'][:120]!r}"
        )

    # 完整 AI 回答逐题（用户审阅）
    lines += ["", "## 逐题完整回答", ""]
    for i, r in enumerate(report["cases"], 1):
        flag = "✅" if r["ok"] else "❌"
        ans = (r.get("answer") or r.get("answer_preview") or "").strip() or "（无正文）"
        lines += [
            f"### {i}. {flag} `{r['id']}` {r['question']}",
            "",
            f"- 类别/难度：{r.get('category')}/{r.get('difficulty')} · 库：{r.get('collection')}",
            f"- 综合分 **{r['total_score']}** · 准确 {r['accuracy']} · 关键词 {r['keyword_rate']} · 延迟 {r['latency_s']}s",
            f"- 拒识={r['is_refuse']} expect_refuse={r['expect_refuse']} · nk={r['no_knowledge']} · 事实校验={r['fact_check_passed']} · 幻觉分={r.get('hallucination_score')}",
            "",
            "```",
            ans,
            "```",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        default=str(ROOT / "evaluation" / "multidim_cases.json"),
    )
    ap.add_argument(
        "--out",
        default=str(
            ROOT
            / "evaluation"
            / "reports"
            / f"multidim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    ap.add_argument("--timeout", type=int, default=90, help="单题超时秒数")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.8,
        help="题间休眠，降低 API 限流",
    )
    ap.add_argument(
        "--web-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="本地库无依据时：免费网页检索(DuckDuckGo)+项目 get_llm 补答（默认开；拒识题永不回退）",
    )
    ap.add_argument(
        "--web-timeout",
        type=int,
        default=120,
        help="联网补答单题超时秒数（外层线程控制）",
    )
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    from src.graph import query

    rows = []
    print(
        f"=== DeepRAG multidim eval n={len(cases)} timeout={args.timeout}s "
        f"web_fallback={args.web_fallback} (free DDG+LLM) ===",
        flush=True,
    )
    for i, case in enumerate(cases, 1):
        q = case["question"]
        col = case.get("collection") or "proj_work"
        t0 = time.time()
        err = None
        result: Dict[str, Any] = {}
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(query, q, collection_name=col)
                result = fut.result(timeout=args.timeout) or {}
        except FuturesTimeout:
            err = f"timeout>{args.timeout}s"
            result = {"answer": "", "errors": [err], "no_knowledge": True}
        except Exception as e:
            err = str(e)
            result = {"answer": "", "errors": [err], "no_knowledge": True}

        # 本地无依据 → 免费联网 + LLM（拒识题不回退）
        if args.web_fallback and _need_web_fallback(case, result):
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(free_web_answer, q, args.web_timeout)
                try:
                    fb = fut.result(timeout=args.web_timeout + 15) or {}
                except Exception as e:
                    fb = {"ok": False, "error": f"web_timeout_or_err:{e}", "answer": ""}
            if fb.get("ok") and fb.get("answer"):
                local_ans = (result.get("answer") or "").strip()
                model = fb.get("raw_model") or "llm"
                engine = fb.get("engine") or "duckduckgo"
                prefix = (
                    f"【来源：免费联网({engine})+LLM({model})；本地知识库未命中或不足】\n"
                )
                merged = prefix + fb["answer"]
                if local_ans and not _is_refuse(local_ans, bool(result.get("no_knowledge"))):
                    merged = (
                        prefix
                        + fb["answer"]
                        + "\n\n---\n【本地库原文摘要】\n"
                        + local_ans[:500]
                    )
                urls = fb.get("urls") or []
                if urls:
                    merged += "\n\n【网页链接】\n" + "\n".join(f"- {u}" for u in urls[:5])
                result = {
                    **result,
                    "answer": merged,
                    "no_knowledge": False,
                    "used_web_fallback": True,
                    "used_cpa_web": False,
                    "answer_source": "free_web_llm",
                    "web_model": model,
                    "web_engine": engine,
                    "web_hits": fb.get("hits"),
                }
                print(
                    f"  -> free web+LLM OK model={model} hits={fb.get('hits')}",
                    flush=True,
                )
            else:
                result = {
                    **result,
                    "used_web_fallback": False,
                    "used_cpa_web": False,
                    "answer_source": "local_kb_or_empty",
                    "web_fallback_error": fb.get("error"),
                }
                print(f"  -> free web+LLM FAIL: {fb.get('error')}", flush=True)
        else:
            result.setdefault("answer_source", "local_kb")
            result.setdefault("used_web_fallback", False)
            result.setdefault("used_cpa_web", False)

        lat = time.time() - t0
        row = score_case(case, result, lat, err)
        rows.append(row)
        flag = "OK" if row["ok"] else "FAIL"
        src = "web" if row.get("used_web_fallback") else "kb"
        print(
            f"[{i}/{len(cases)}] {flag} {row['id']} total={row['total_score']} "
            f"acc={row['accuracy']} {lat:.1f}s nk={row['no_knowledge']} src={src} | {q[:36]}",
            flush=True,
        )
        # 增量落盘，避免中途挂掉全丢
        try:
            partial = {
                "partial": True,
                "done": i,
                "total": len(cases),
                "hit": sum(1 for r in rows if r["ok"]),
                "cases": rows,
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(str(args.out) + ".partial.json").write_text(
                json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        if args.sleep > 0:
            time.sleep(args.sleep)

    hit = sum(1 for r in rows if r["ok"])
    metrics = aggregate(rows)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(rows),
        "hit": hit,
        "accuracy_pass": round(hit / len(rows), 4) if rows else 0,
        "metrics": metrics,
        "cases": rows,
        "config_hint": {
            "QDRANT_MODE": os.getenv("QDRANT_MODE"),
            "web_fallback": bool(args.web_fallback),
            "web_fallback_mode": "free_duckduckgo_plus_project_llm",
            "note": "以运行时 src.config / .env 为准；本地无依据时免费联网+LLM",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(to_md(report), encoding="utf-8")

    print("\n======== SUMMARY ========", flush=True)
    print(f"pass={hit}/{len(rows)} ({report['accuracy_pass']:.1%}) grade={metrics['等级']}", flush=True)
    print(f"total_score_mean={metrics['综合分_mean']} p50_lat={metrics['延迟_p50_s']}s", flush=True)
    print(f"JSON: {out}", flush=True)
    print(f"MD:   {md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
