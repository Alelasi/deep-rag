"""统计指定 Python 文件的注释率（注释行 / 总非空行）

用法：
  python scripts/comment_ratio_check.py src/security src/pipeline_routing.py
要求：目标模块注释率 >= 20%
"""
from __future__ import annotations

import sys
from pathlib import Path


def analyze(path: Path) -> tuple[int, int, float]:
    """返回 (注释行, 非空行, 比率)。"""
    text = path.read_text(encoding="utf-8")
    comment_lines = 0
    code_or_comment = 0
    in_doc = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        code_or_comment += 1
        if s.startswith('"""') or s.startswith("'''"):
            comment_lines += 1
            # 单行 docstring
            if s.count('"""') >= 2 or s.count("'''") >= 2:
                continue
            in_doc = not in_doc
            continue
        if in_doc:
            comment_lines += 1
            continue
        if s.startswith("#"):
            comment_lines += 1
    ratio = comment_lines / code_or_comment if code_or_comment else 0.0
    return comment_lines, code_or_comment, ratio


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    targets: list[Path] = []
    for arg in argv[1:] or ["src/security", "src/pipeline_routing.py"]:
        p = (root / arg).resolve()
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.py")))
        elif p.is_file():
            targets.append(p)

    failed = 0
    for p in targets:
        if p.name == "__init__.py" and p.stat().st_size < 200:
            continue
        c, t, r = analyze(p)
        status = "OK" if r >= 0.20 else "LOW"
        if r < 0.20:
            failed += 1
        print(f"{status} {r:6.1%}  comments={c:3d} non_empty={t:3d}  {p.relative_to(root)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
