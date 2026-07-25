"""输入校验与索引路径沙箱

职责：
1. sanitize_question — 长度截断、去 NUL、启发式注入标记
2. validate_index_path — 仅允许白名单根下目录被 /index 灌库（防任意路径读盘）
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

# 默认最大问题长度（可用 MAX_QUESTION_CHARS 覆盖）
DEFAULT_MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))

# 常见 prompt 注入试探（启发式；不能替代模型侧防护）
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"system\s*:\s*you\s+are", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
]


def sanitize_question(
    question: str,
    max_chars: int = DEFAULT_MAX_QUESTION_CHARS,
) -> Tuple[str, List[str]]:
    """清洗用户问题。

    Returns:
        (clean_text, warnings)；非法空输入抛 ValueError
    """
    warnings: List[str] = []
    if question is None:
        raise ValueError("question 不能为空")
    text = str(question).strip()
    if not text:
        raise ValueError("question 不能为空")

    # 超长截断，避免拖垮 LLM context / 费用
    if len(text) > max_chars:
        text = text[:max_chars]
        warnings.append(f"question 截断至 {max_chars} 字符")

    # 注入试探只告警，不直接拒绝（避免误杀正常英文技术问答）
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            warnings.append("detected_possible_prompt_injection")
            break

    if "\x00" in text:
        text = text.replace("\x00", "")
        warnings.append("removed_null_bytes")
    return text, warnings


def _allowed_roots() -> List[Path]:
    """解析允许索引的根路径列表（环境变量 + 项目 data 默认）。"""
    roots: List[Path] = []
    raw = os.getenv("INDEX_ALLOWED_ROOTS", "").strip()
    if raw:
        # 支持 PATH 风格 pathsep 或逗号分隔
        sep = os.pathsep if os.pathsep in raw else ","
        for part in raw.split(sep):
            part = part.strip()
            if part:
                roots.append(Path(part).resolve())

    # 默认始终允许项目内 data / sample_docs，保证 demo 可索引
    project = Path(__file__).resolve().parents[2]
    roots.append((project / "data").resolve())
    roots.append((project / "data" / "sample_docs").resolve())

    # 去重且保序
    uniq: List[Path] = []
    seen = set()
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def validate_index_path(docs_dir: str) -> Path:
    """校验 docs_dir 可被安全索引。

    Raises:
        ValueError: 路径不存在或不是目录
        PermissionError: 不在白名单根之下
    """
    path = Path(docs_dir).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"目录不存在: {docs_dir}")
    if not path.is_dir():
        raise ValueError(f"不是目录: {docs_dir}")

    # 必须落在某一允许根之下（含根自身）
    for root in _allowed_roots():
        try:
            path.relative_to(root)
            return path
        except ValueError:
            if path == root:
                return path

    raise PermissionError(
        f"索引路径不在白名单内。允许根: {[str(r) for r in _allowed_roots()]}。"
        f"可通过 INDEX_ALLOWED_ROOTS 扩展。"
    )
