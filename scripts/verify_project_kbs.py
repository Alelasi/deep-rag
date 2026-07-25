#!/usr/bin/env python3
"""构建完成后的多项目知识库验收

- 列出各 collection 点数
- 对已就绪库做 1 次向量检索 smoke（可选 --query）

用法：
  # 构建进程结束后再跑（local 模式勿并行写）
  .venv/Scripts/python.exe scripts/verify_project_kbs.py
  .venv/Scripts/python.exe scripts/verify_project_kbs.py --query "什么是MBTI"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QDRANT_MODE", "local")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="", help="对每个非空库跑一条检索")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    from src.retrieval.project_collections import PROJECT_COLLECTIONS, PROJECT_LABELS
    from src.retrieval.qdrant_retriever import get_qdrant_retriever, list_collection_stats

    print("=== Qdrant collections ===")
    stats = {s["name"]: s["points"] for s in list_collection_stats()}
    ready = 0
    for key, col in PROJECT_COLLECTIONS.items():
        n = stats.get(col, 0)
        label = PROJECT_LABELS.get(key, key)
        flag = "OK" if n > 0 else "EMPTY"
        if n > 0:
            ready += 1
        print(f"  [{flag}] {label:8} {col:20} points={n}")

    print(f"\nready {ready}/{len(PROJECT_COLLECTIONS)}")

    if args.query.strip():
        print(f"\n=== smoke query: {args.query!r} ===")
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5"), device=device
        )
        emb = model.encode(
            [args.query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        for key, col in PROJECT_COLLECTIONS.items():
            if stats.get(col, 0) <= 0:
                continue
            r = get_qdrant_retriever(col)
            hits = r.search(emb, top_k=args.top_k)
            print(f"\n-- {key} / {col} --")
            for h in hits:
                src = h.get("source", "")[:60]
                score = h.get("_score", 0)
                snippet = (h.get("content") or "")[:80].replace("\n", " ")
                print(f"  {score:.3f} | {src} | {snippet}")

    return 0 if ready > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
