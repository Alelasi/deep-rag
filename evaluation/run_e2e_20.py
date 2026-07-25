#!/usr/bin/env python3
"""DeepRAG 主路径 20 题 E2E（强制走 src.graph）

用法：
  cd deep-rag
  python evaluation/run_e2e_20.py --out evaluation/reports/e2e_20_latest.json

说明：
- 无向量库 / 无 LLM 时仍会跑通，但 no_knowledge 与答案质量会偏低
- 报告禁止自动写成「95%」；只输出 N/20 与分题明细
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _keyword_hit(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    text = answer or ""
    return any(k in text for k in keywords if k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(ROOT / "evaluation" / "e2e_20.json"),
        help="用例 JSON 路径",
    )
    parser.add_argument(
        "--out",
        default=str(
            ROOT
            / "evaluation"
            / "reports"
            / f"e2e_20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    parser.add_argument("--collection", default="default")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    from src.graph import query

    rows = []
    hit = 0
    no_knowledge_n = 0
    mock_n = 0

    for case in cases:
        q = case["question"]
        t0 = time.time()
        err = None
        result = {}
        try:
            result = query(q, collection_name=args.collection) or {}
        except Exception as e:
            err = str(e)
            result = {"answer": "", "errors": [err]}
        latency_ms = int((time.time() - t0) * 1000)

        answer = result.get("answer") or ""
        no_knowledge = bool(result.get("no_knowledge"))
        used_mock = bool(result.get("used_mock_web"))
        if no_knowledge:
            no_knowledge_n += 1
        if used_mock:
            mock_n += 1

        keywords = case.get("expected_keywords") or []
        allow_refuse = bool(case.get("allow_refuse"))
        is_refuse = no_knowledge or any(
            w in answer
            for w in ("未找到可靠依据", "无法基于证据", "知识库与外部检索均未找到")
        )
        if allow_refuse and is_refuse:
            ok = True
        else:
            ok = _keyword_hit(answer, keywords) and not (is_refuse and not allow_refuse)
        if ok:
            hit += 1

        rows.append(
            {
                "id": case.get("id"),
                "question": q,
                "ok": ok,
                "latency_ms": latency_ms,
                "no_knowledge": no_knowledge,
                "used_mock_web": used_mock,
                "hallucination_score": result.get("hallucination_score"),
                "answer_preview": answer[:200],
                "error": err,
            }
        )
        print(
            f"[{'OK' if ok else 'FAIL'}] {case.get('id')} {latency_ms}ms "
            f"nk={no_knowledge} mock={used_mock} | {q[:40]}"
        )

    total = len(cases)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "hit": hit,
        "accuracy": round(hit / total, 4) if total else 0.0,
        "no_knowledge_rate": round(no_knowledge_n / total, 4) if total else 0.0,
        "mock_web_rate": round(mock_n / total, 4) if total else 0.0,
        "note": "启发式关键词命中，非官方 RAGAS；勿直接写成简历 95%",
        "cases": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_path.with_suffix(".md")
    md_lines = [
        f"# E2E 20 报告",
        f"",
        f"- 时间：{report['generated_at']}",
        f"- 命中：**{hit}/{total}**（{report['accuracy']:.1%}）",
        f"- no_knowledge 率：{report['no_knowledge_rate']:.1%}",
        f"- mock_web 率：{report['mock_web_rate']:.1%}",
        f"- 说明：{report['note']}",
        f"",
        f"| id | ok | ms | nk | mock | question |",
        f"|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['id']} | {r['ok']} | {r['latency_ms']} | {r['no_knowledge']} | "
            f"{r['used_mock_web']} | {r['question'][:40]} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n=> {hit}/{total} accuracy={report['accuracy']:.1%}")
    print(f"JSON: {out_path}")
    print(f"MD:   {md_path}")
    return 0 if hit >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
