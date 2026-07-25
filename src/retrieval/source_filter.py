"""检索结果源过滤 — 去掉易污染答案的路径

典型脏源：
- chat_history.json（历史错误答案被再次检索）
- 构建/缓存/日志类文件
"""
from __future__ import annotations

from typing import Any, List

# 路径子串命中则丢弃（小写比较）
DENY_SOURCE_SUBSTR = (
    "chat_history.json",
    "test_results",
    "evaluation_reports",
    "__pycache__",
    ".venv",
    "node_modules",
    "build_kb",
    "htmlcov",
    "pytest_cache",
)


def source_of(doc: Any) -> str:
    if isinstance(doc, dict):
        meta = doc.get("metadata") or {}
        return str(doc.get("source") or meta.get("source") or "")
    return str(getattr(doc, "source", "") or "")


def is_denied_source(source: str) -> bool:
    s = (source or "").replace("\\", "/").lower()
    return any(x in s for x in DENY_SOURCE_SUBSTR)


def filter_docs(docs: List[Any]) -> List[Any]:
    """过滤脏源；若全被滤掉则退回原列表避免空检索。"""
    if not docs:
        return docs
    kept = [d for d in docs if not is_denied_source(source_of(d))]
    return kept if kept else docs


def prefer_exact_type_stack(question: str, docs: List[Any]) -> List[Any]:
    """MBTI 类型堆栈问题：把含「TYPE: Ni-Te-...」明确行的文档提到前面。"""
    import re

    if not docs or not question:
        return docs
    m = re.search(r"\b([IE][NS][TF][JP])\b", question, re.I)
    if not m:
        return docs
    code = m.group(1).upper()
    # 匹配 INTJ: Ni-Te-Fi-Se 或 INTJ的...Ni-Te-Fi-Se
    pat = re.compile(
        rf"{code}\s*[:：]\s*[A-Z]{{2}}\s*-\s*[A-Z]{{2}}\s*-\s*[A-Z]{{2}}\s*-\s*[A-Z]{{2}}",
        re.I,
    )
    hit, rest = [], []
    for d in docs:
        content = d.get("content") if isinstance(d, dict) else getattr(d, "content", "")
        content = content or ""
        if pat.search(content) or (code in content and "Ni-Te-Fi-Se" in content and code == "INTJ"):
            hit.append(d)
        else:
            rest.append(d)
    return hit + rest if hit else docs
