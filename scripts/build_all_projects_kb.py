#!/usr/bin/env python3
"""为 D:\\文档\\ai提问相关 下各一级项目构建独立 Chroma 集合

数据库中心（服务器数据目录）：
  D:\\文档\\ai提问相关\\哲思灵智\\向量数据库

安全红线：
  - 禁止 PersistentClient 直连已有库
  - 必须先：chroma run --path <中心路径> --port 8000
  - 本脚本仅 HttpClient(localhost:8000)

用法：
  # 终端1
  chroma run --path "D:/文档/ai提问相关/哲思灵智/向量数据库" --port 8000
  # 终端2
  cd deep-rag
  .venv/Scripts/python.exe scripts/build_all_projects_kb.py
  .venv/Scripts/python.exe scripts/build_all_projects_kb.py --project work --force
  .venv/Scripts/python.exe scripts/build_all_projects_kb.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---- 环境：必须在 import chromadb / torch 之前 ----
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 向量库中心（与 CLAUDE / start_deeprag.bat 一致）
CENTER_DB = Path(os.getenv(
    "CHROMA_CENTER_PATH",
    r"D:\文档\ai提问相关\哲思灵智\向量数据库",
))
CHROMA_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_SERVER_PORT", "8000"))

# 一级项目 → collection 名 + 源根
# 注释率：配置即文档
PROJECTS = {
    "work": {
        "path": r"D:\文档\ai提问相关\工作",
        "collection": "proj_work",
        "desc": "工作区：deep-rag/docs/简历/面试等",
    },
    "thesis": {
        "path": r"D:\文档\ai提问相关\论文",
        "collection": "proj_thesis",
        "desc": "毕业论文相关",
    },
    "psychology": {
        "path": r"D:\文档\ai提问相关\心理人际",
        "collection": "proj_psychology",
        "desc": "心理人际资料",
    },
    "social": {
        "path": r"D:\文档\ai提问相关\社科",
        "collection": "proj_social",
        "desc": "社科资料",
    },
    "ideas": {
        "path": r"D:\文档\ai提问相关\奇思妙想",
        "collection": "proj_ideas",
        "desc": "奇思妙想",
    },
    "assistant": {
        "path": r"D:\文档\ai提问相关\助理",
        "collection": "proj_assistant",
        "desc": "助理项目",
    },
    "tools": {
        "path": r"D:\文档\ai提问相关\工具",
        "collection": "proj_tools",
        "desc": "工具脚本与资料",
    },
    "worklog": {
        "path": r"D:\文档\ai提问相关\工作日志",
        "collection": "proj_worklog",
        "desc": "工作日志",
    },
    "zhesi": {
        "path": r"D:\文档\ai提问相关\哲思灵智",
        "collection": "proj_zhesi",
        "desc": "哲思灵智书籍与 rag-docs（不含向量库本体）",
    },
    "api_router": {
        "path": r"D:\文档\ai提问相关\api-router-ui",
        "collection": "proj_api_router",
        "desc": "api-router-ui",
    },
}

# 扫描排除（避免 venv/缓存/向量库本体递归进库）
EXCLUDE_DIR_NAMES = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".claude", ".trae", ".ace-tool", ".streamlit",
    "chroma_db", "chroma_db_corrupted", "htmlcov", "dist", "build",
    ".benchmarks", "向量数据库",  # 中心库自身不入库
}
EXCLUDE_PATH_SUBSTR = [
    "向量数据库",
    "chroma_db",
    ".venv",
    "node_modules",
    "垃圾桶_待确认删除",
    "归档/数据实验",
]
SUPPORTED_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html", ".mdx"}
# 单文件过大则跳过（防词典/全量 dump）
MAX_FILE_BYTES = 2 * 1024 * 1024
# 分块
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH = 64
SLEEP_EVERY_BATCH = 0.15


def log(msg: str) -> None:
    print(msg, flush=True)


def get_http_client():
    """仅 HttpClient — 禁止 PersistentClient。"""
    import chromadb

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    # 探活
    client.heartbeat()
    return client


def should_skip_path(path: Path) -> bool:
    """路径是否应跳过。"""
    parts = set(path.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    s = str(path)
    for sub in EXCLUDE_PATH_SUBSTR:
        if sub in s:
            return True
    return False


def iter_files(root: Path):
    """遍历可索引文本文件。"""
    if not root.exists():
        return
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if should_skip_path(fp):
            continue
        if fp.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if fp.name.startswith("."):
            continue
        try:
            if fp.stat().st_size > MAX_FILE_BYTES or fp.stat().st_size < 32:
                continue
        except OSError:
            continue
        yield fp


def split_text(text: str) -> list[str]:
    """简易重叠分块（不强制 langchain，降低依赖失败率）。"""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + CHUNK_SIZE)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def load_embedder():
    """加载中文 embedding；优先 GPU。"""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    log(f"[embed] model={model_name} device={device}")
    if device == "cuda":
        torch.set_num_threads(2)
    model = SentenceTransformer(model_name, device=device)
    return model, device


def ensure_collection(client, name: str, force: bool):
    """获取或创建集合；force 时先删后建。"""
    if force:
        try:
            client.delete_collection(name)
            log(f"  [force] deleted collection {name}")
        except Exception:
            pass
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:sync_threshold": 50,
                "project": name,
            },
        )


def index_project(client, embedder, key: str, cfg: dict, force: bool) -> dict:
    """索引单个项目，返回统计。"""
    root = Path(cfg["path"])
    col_name = cfg["collection"]
    stats = {
        "project": key,
        "collection": col_name,
        "path": str(root),
        "files": 0,
        "chunks": 0,
        "skipped": False,
        "error": None,
    }
    if not root.exists():
        stats["error"] = "path_not_found"
        log(f"[skip] {key}: 目录不存在 {root}")
        return stats

    col = ensure_collection(client, col_name, force=force)
    if not force:
        try:
            cnt = col.count()
            if cnt > 0:
                stats["skipped"] = True
                stats["chunks"] = cnt
                log(f"[skip] {key}: 已有 {cnt} 条，使用 --force 重建")
                return stats
        except Exception:
            pass

    # 收集块
    all_chunks = []
    for fp in iter_files(root):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(text.strip()) < 20:
            continue
        stats["files"] += 1
        try:
            rel = str(fp.relative_to(root))
        except ValueError:
            rel = str(fp)
        for i, chunk in enumerate(split_text(text)):
            doc_id = hashlib.md5(f"{key}:{rel}:{i}".encode()).hexdigest()[:16]
            all_chunks.append((doc_id, chunk, rel, i + 1))

    log(f"[{key}] files={stats['files']} raw_chunks={len(all_chunks)}")
    if not all_chunks:
        stats["error"] = "no_chunks"
        return stats

    # 批量写入
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]
        docs = [b[1] for b in batch]
        ids = [b[0] for b in batch]
        metas = [
            {"source": b[2], "page": b[3], "project": key}
            for b in batch
        ]
        emb = embedder.encode(docs, show_progress_bar=False)
        # numpy → list
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        col.add(documents=docs, ids=ids, metadatas=metas, embeddings=emb)
        stats["chunks"] += len(batch)
        if (i // BATCH) % 10 == 0:
            log(f"  ... {stats['chunks']}/{len(all_chunks)}")
        time.sleep(SLEEP_EVERY_BATCH)
        if (i // BATCH) % 10 == 9:
            time.sleep(2.0)  # 每 10 批多歇，控 CPU/GPU

    log(f"[ok] {key}: {stats['chunks']} chunks → {col_name}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="多项目向量库构建（中心=哲思灵智）")
    parser.add_argument("--project", choices=list(PROJECTS.keys()) + ["all"], default="all")
    parser.add_argument("--force", action="store_true", help="删除后重建集合")
    parser.add_argument("--dry-run", action="store_true", help="只列计划不写入")
    args = parser.parse_args()

    log("=" * 60)
    log(f"中心路径: {CENTER_DB}")
    log(f"HttpClient: {CHROMA_HOST}:{CHROMA_PORT}")
    log(f"time: {datetime.now().isoformat(timespec='seconds')}")
    log("=" * 60)

    keys = list(PROJECTS.keys()) if args.project == "all" else [args.project]
    if args.dry_run:
        for k in keys:
            c = PROJECTS[k]
            exists = Path(c["path"]).exists()
            log(f"  {k:12} → {c['collection']:18} exists={exists}  {c['path']}")
        return 0

    # 连接服务器
    try:
        client = get_http_client()
    except Exception as e:
        log("ERROR: 无法连接 Chroma 服务器。请先启动：")
        log(f'  chroma run --path "{CENTER_DB}" --port {CHROMA_PORT}')
        log(f"  detail: {e}")
        return 2

    embedder, _ = load_embedder()
    results = []
    t0 = time.time()
    for k in keys:
        log(f"\n>>> 开始项目 {k}: {PROJECTS[k]['desc']}")
        try:
            st = index_project(client, embedder, k, PROJECTS[k], force=args.force)
        except Exception as e:
            st = {
                "project": k,
                "collection": PROJECTS[k]["collection"],
                "error": str(e),
                "chunks": 0,
            }
            log(f"[ERR] {k}: {e}")
        results.append(st)

    # 写注册表到中心旁
    registry = {
        "center": str(CENTER_DB),
        "host": CHROMA_HOST,
        "port": CHROMA_PORT,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 1),
        "projects": results,
    }
    reg_path = CENTER_DB.parent / "向量库注册表.json"
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n注册表: {reg_path}")
    log(f"总耗时: {registry['elapsed_sec']}s")

    # 摘要
    for r in results:
        log(
            f"  - {r.get('project')}: chunks={r.get('chunks')} "
            f"skip={r.get('skipped')} err={r.get('error')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
