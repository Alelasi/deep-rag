"""多轮对话上下文 — 相关追问可借鉴前几轮，并做堆栈一致性约束

场景：Q1「INTJ 主导功能」答对 Ni-Te-Fi-Se 后，
     Q2「INTJ 功能排序」不得再编出 Si-Fe。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# 类型码
_TYPE_RE = re.compile(r"\b([IE][NS][TF][JP])\b", re.I)
# 功能堆栈行：Ni-Te-Fi-Se
_STACK_RE = re.compile(
    r"\b([IE][NS][TF][JP])\s*[:：]?\s*"
    r"([NSTF][ie])\s*[-–—/]\s*"
    r"([NSTF][ie])\s*[-–—/]\s*"
    r"([NSTF][ie])\s*[-–—/]\s*"
    r"([NSTF][ie])\b",
    re.I,
)


def extract_type_codes(text: str) -> List[str]:
    return list({m.group(1).upper() for m in _TYPE_RE.finditer(text or "")})


def extract_stacks(text: str) -> Dict[str, str]:
    """从文本抽出 {INTJ: 'Ni-Te-Fi-Se', ...}"""
    out: Dict[str, str] = {}
    for m in _STACK_RE.finditer(text or ""):
        code = m.group(1).upper()
        stack = "-".join(g.capitalize() if len(g) == 2 else g for g in m.groups()[1:])
        # 规范化 Ni/Te 大小写：第一位大写第二位小写
        parts = []
        for g in m.groups()[1:]:
            parts.append(g[0].upper() + g[1].lower())
        out[code] = "-".join(parts)
    return out


def is_related(question: str, prev_q: str, prev_a: str) -> bool:
    """判断当前问题是否与上一轮相关（可借鉴）。"""
    if not question or not prev_q:
        return False
    # 同一 MBTI 类型
    tq = set(extract_type_codes(question))
    tp = set(extract_type_codes(prev_q + " " + (prev_a or "")))
    if tq and tp and (tq & tp):
        return True
    # 关键词重叠（功能/堆栈/主导）
    keys = ("功能", "堆栈", "主导", "辅助", "劣势", "排序", "认知", "MBTI", "人格")
    q_hit = sum(1 for k in keys if k in question)
    p_hit = sum(1 for k in keys if k in prev_q)
    if q_hit >= 1 and p_hit >= 1 and (tq & tp or not tq):
        # 短问题追问更可能相关
        if len(question) < 40 or any(k in question for k in ("排序", "功能", "呢", "那")):
            return True
    return False


def build_prior_context(question: str, turns: List[dict], max_turns: int = 3) -> str:
    """生成注入 prompt 的前轮摘要（仅相关轮次）。"""
    if not turns:
        return ""
    lines = []
    for turn in turns[-max_turns:]:
        pq, pa = turn.get("q", ""), turn.get("a", "")
        if not is_related(question, pq, pa):
            continue
        # 截断答案，保留堆栈行
        stacks = extract_stacks(pa)
        stack_note = ""
        if stacks:
            stack_note = "；已确认堆栈：" + "，".join(f"{k}={v}" for k, v in stacks.items())
        lines.append(f"- 用户曾问：{pq[:80]}\n  系统曾答：{(pa or '')[:200]}{stack_note}")
    if not lines:
        return ""
    return (
        "【对话上下文·相关前轮，必须与之保持一致，禁止矛盾】\n"
        + "\n".join(lines)
        + "\n"
    )


def consistency_hint(question: str, turns: List[dict]) -> str:
    """硬约束：前轮已给出的类型堆栈必须沿用。"""
    if not turns:
        return ""
    known: Dict[str, str] = {}
    for turn in turns[-5:]:
        known.update(extract_stacks(turn.get("a") or ""))
    if not known:
        return ""
    q_types = extract_type_codes(question)
    relevant = {k: v for k, v in known.items() if not q_types or k in q_types}
    if not relevant:
        relevant = known
    rules = "；".join(f"{k} 功能堆栈必须是 {v}（不得改成其他顺序）" for k, v in relevant.items())
    return f"【一致性硬约束】{rules}。若与检索文档冲突，优先采用前轮已确认且文档中出现的明确堆栈行。\n"


def find_contradiction(answer: str, turns: List[dict]) -> Optional[Tuple[str, str, str]]:
    """若新答案堆栈与前轮冲突，返回 (type, old_stack, new_stack)。"""
    if not answer or not turns:
        return None
    known: Dict[str, str] = {}
    for turn in turns[-5:]:
        known.update(extract_stacks(turn.get("a") or ""))
    new = extract_stacks(answer)
    for code, new_s in new.items():
        old = known.get(code)
        if old and old.lower() != new_s.lower():
            return code, old, new_s
    return None
