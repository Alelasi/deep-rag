"""Pipeline 路由纯函数 — 从 graph 抽出，便于单测与注释密度达标

与 ``src.graph`` 的约定：
- Corrective：grade 后决定 generate / rewrite / web
- Self-RAG：fact_check 后决定 check_conflicts / regenerate
- 开关来自 ``src.config``，默认 fast（不 regenerate）
"""
from __future__ import annotations

from typing import Any, Mapping

from src.config import ENABLE_SELF_RAG_LOOP, SELF_RAG_MAX_REGENERATE


def route_after_grading(state: Mapping[str, Any]) -> str:
    """Corrective RAG：文档评分后的下一跳。

    规则（与历史 graph 行为对齐）：
    - 至少 1 篇 relevant → generate
    - 已走过 web_search → 强制 generate（避免死循环）
    - retry_count < 1 → rewrite_query 再检索
    - 否则 → web_search 兜底
    """
    # 优先用上游写入的 relevant_count，避免重复扫描
    if "relevant_count" in state:
        relevant = int(state.get("relevant_count") or 0)
    else:
        graded = state.get("graded_docs") or []
        relevant = sum(1 for d in graded if d.get("grade") == "relevant")
    retry_count = int(state.get("retry_count") or 0)

    if relevant >= 1:
        return "generate"
    # 与历史 graph 一致：决策已是 web_search 时仍路由到 web_search 节点
    if state.get("retrieval_decision") == "web_search":
        return "web_search"
    if retry_count < 1:
        return "rewrite_query"
    return "web_search"


def route_after_fact_check(state: Mapping[str, Any]) -> str:
    """Self-RAG：事实校验后的下一跳。

    - ENABLE_SELF_RAG_LOOP=false（默认）→ 直接 check_conflicts
    - 已通过校验 → check_conflicts
    - 未通过且 regenerate_count < MAX → regenerate
    - 否则 → check_conflicts（带失败分数字段由上游写入）
    """
    if not ENABLE_SELF_RAG_LOOP:
        return "check_conflicts"
    if state.get("fact_check_passed", True):
        return "check_conflicts"
    regen = int(state.get("regenerate_count") or 0)
    if regen < SELF_RAG_MAX_REGENERATE:
        return "regenerate"
    return "check_conflicts"


def is_mock_web_doc(doc: Mapping[str, Any]) -> bool:
    """判断单条 Web 结果是否为 mock 占位（不可作证据）。"""
    meta = doc.get("metadata") or {}
    if meta.get("is_mock"):
        return True
    if meta.get("engine") == "mock":
        return True
    src = str(doc.get("source") or "")
    return src.startswith("mock://")


def filter_real_web_results(web_results: list | None) -> list:
    """过滤掉 mock Web 结果，仅保留真实检索片段。"""
    if not web_results:
        return []
    return [w for w in web_results if not is_mock_web_doc(w)]
