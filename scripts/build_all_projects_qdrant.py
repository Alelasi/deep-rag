#!/usr/bin/env python3
"""多项目向量库构建 — Qdrant 版（推荐，避免 Chroma HNSW 损坏）

数据中心默认：
  D:\\文档\\ai提问相关\\哲思灵智\\qdrant_data

两种运行模式：
1) 本地磁盘模式（无需 Docker，推荐本机先用这个）
   --mode local
2) Docker/服务器模式（Qdrant 在 127.0.0.1:6333）
   --mode server

GPU：
  - embedding 走 CUDA（若可用）
  - CPU 线程压到 2，降低扫描/序列化占满整机

用法：
  cd deep-rag
  .venv/Scripts/python.exe scripts/build_all_projects_qdrant.py --dry-run
  .venv/Scripts/python.exe scripts/build_all_projects_qdrant.py --project work
  .venv/Scripts/python.exe scripts/build_all_projects_qdrant.py --project all --force
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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 限制底层 BLAS/OpenMP 线程，避免「看起来在用 GPU 但 CPU 100%」
# BLAS 线程：balanced/fast 用 2；gentle 可在 load 时再压
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 中心目录（与哲思灵智并列 chroma 中心；Qdrant 用独立子目录）
CENTER = Path(os.getenv(
    "QDRANT_CENTER_PATH",
    r"D:\文档\ai提问相关\哲思灵智\qdrant_data",
))
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_SIZE = 768  # bge-base-zh-v1.5

# zhesi（含「书籍/」）提前于 ideas，避免奇思妙想大扫盘拖住书库
PROJECTS = {
    "work": {"path": r"D:\文档\ai提问相关\工作", "collection": "proj_work"},
    "thesis": {"path": r"D:\文档\ai提问相关\论文", "collection": "proj_thesis"},
    "psychology": {"path": r"D:\文档\ai提问相关\心理人际", "collection": "proj_psychology"},
    "social": {"path": r"D:\文档\ai提问相关\社科", "collection": "proj_social"},
    "zhesi": {"path": r"D:\文档\ai提问相关\哲思灵智", "collection": "proj_zhesi"},
    "ideas": {"path": r"D:\文档\ai提问相关\奇思妙想", "collection": "proj_ideas"},
    "assistant": {"path": r"D:\文档\ai提问相关\助理", "collection": "proj_assistant"},
    "tools": {"path": r"D:\文档\ai提问相关\工具", "collection": "proj_tools"},
    "worklog": {"path": r"D:\文档\ai提问相关\工作日志", "collection": "proj_worklog"},
    "api_router": {"path": r"D:\文档\ai提问相关\api-router-ui", "collection": "proj_api_router"},
}

EXCLUDE_DIR_NAMES = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".claude", ".trae", "htmlcov", "dist", "build",
    "向量数据库", "qdrant_data", "chroma_db",
    "site-packages", ".tox", "vendor", ".uploads",
    "trae_聊天记录", "rag-docs-全盘",  # 重复/超大体量，书籍不在这些目录
}
EXCLUDE_SUBSTR = [
    "向量数据库", "qdrant_data", ".venv", "node_modules",
    "垃圾桶_待确认删除", "site-packages", "\\dist\\", "/dist/",
]
SUPPORTED_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html", ".mdx"}
# 普通代码/笔记 2MB；「书籍」目录放宽到 48MB（名著合辑 md 可达数十 MB）
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_BOOK_FILE_BYTES = 48 * 1024 * 1024
# 同名文件只保留一份（见 collect_files）
DEDUP_BY_BASENAME = os.getenv("DEDUP_BY_BASENAME", "1").lower() not in ("0", "false", "no")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# === 速度档位（慢的主因曾是：micro=24 + sleep 0.5s/批 ≈ 纯休息占一半时间）===
# 默认 balanced：接近早期「几十分钟级」；--gentle 才强节流
# 环境变量仍可覆盖
PROFILE = os.getenv("BUILD_PROFILE", "balanced").strip().lower()  # desktop | gentle | balanced | fast
_PROFILES = {
    # 跟手优先：CPU/GPU 尽量压在约 30–45%（明显变慢）
    "gentle": dict(micro=16, throttle=0.55, upsert_bs=96, upsert_sleep=0.25, threads=1),
    # 桌面可用：硬约束目标 ≤50%（批间多睡 + 更小 batch + 单线程 + 亲和性约 40% 核）
    "desktop": dict(micro=24, throttle=0.40, upsert_bs=96, upsert_sleep=0.20, threads=1),
    # 默认：GPU 能吃饱，略让桌面（可能 >50%，仅无人时用）
    "balanced": dict(micro=64, throttle=0.02, upsert_bs=256, upsert_sleep=0.02, threads=2),
    # 无人值守：尽量快
    "fast": dict(micro=96, throttle=0.0, upsert_bs=384, upsert_sleep=0.0, threads=2),
}
_p = _PROFILES.get(PROFILE, _PROFILES["balanced"])
GPU_MICRO_BATCH = int(os.getenv("EMBED_MICRO_BATCH", str(_p["micro"])))
CPU_MICRO_BATCH = 8
THROTTLE_SLEEP = float(os.getenv("THROTTLE_SLEEP", str(_p["throttle"])))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", str(_p["upsert_bs"])))
UPSERT_SLEEP = float(os.getenv("UPSERT_SLEEP", str(_p["upsert_sleep"])))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", str(_p["threads"])))


_LOG_FP = None


def _init_file_log() -> None:
    """同时写文件日志，避免后台任务被杀时只剩截断 stdout。"""
    global _LOG_FP
    try:
        path = CENTER.parent / "build_kb.log"
        _LOG_FP = open(path, "a", encoding="utf-8")
        print(f"[log] appending → {path}", flush=True)
    except Exception:
        _LOG_FP = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG_FP is not None:
        try:
            _LOG_FP.write(msg + "\n")
            _LOG_FP.flush()
        except Exception:
            pass


def _fmt_eta(seconds: float) -> str:
    """秒 → 可读 ETA。"""
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def should_skip(path: Path) -> bool:
    parts_l = {p.lower() for p in path.parts}
    if parts_l & {x.lower() for x in EXCLUDE_DIR_NAMES}:
        return True
    s = str(path)
    return any(x in s for x in EXCLUDE_SUBSTR)


def _max_bytes_for(path: Path) -> int:
    """书籍目录允许更大单文件，确保哲思灵智/书籍 全量入库。"""
    if "书籍" in path.parts:
        return MAX_BOOK_FILE_BYTES
    return MAX_FILE_BYTES


def _file_priority(path: Path, size: int) -> tuple:
    """同名去重排序：书籍优先 > 更大 > 路径更短（更可能是主副本）。"""
    in_books = 1 if "书籍" in path.parts else 0
    return (in_books, size, -len(str(path)))


def collect_files(root: Path) -> tuple[list[Path], dict]:
    """收集待索引文件；默认按文件名去重（同名只留 1 份，优先书籍/更大文件）。

    Returns:
        (files, stats) stats 含 scanned/kept/skipped_dup/skipped_size
    """
    stats = {
        "scanned": 0,
        "kept": 0,
        "skipped_dup": 0,
        "skipped_size": 0,
        "skipped_other": 0,
    }
    if not root.exists():
        return [], stats

    # name_lower -> (priority_tuple, path)
    best: dict[str, tuple[tuple, Path]] = {}
    ordered: list[Path] = []  # 不去重时的顺序

    for fp in root.rglob("*"):
        if not fp.is_file() or should_skip(fp):
            continue
        if fp.suffix.lower() not in SUPPORTED_EXTS or fp.name.startswith("."):
            continue
        stats["scanned"] += 1
        try:
            sz = fp.stat().st_size
        except OSError:
            stats["skipped_other"] += 1
            continue
        if sz < 32:
            stats["skipped_other"] += 1
            continue
        if sz > _max_bytes_for(fp):
            stats["skipped_size"] += 1
            continue

        if not DEDUP_BY_BASENAME:
            ordered.append(fp)
            continue

        key = fp.name.lower()
        pri = _file_priority(fp, sz)
        old = best.get(key)
        if old is None:
            best[key] = (pri, fp)
        else:
            stats["skipped_dup"] += 1
            if pri > old[0]:
                best[key] = (pri, fp)

    if DEDUP_BY_BASENAME:
        # 书籍优先排前面，便于日志与尽早入库
        files = [p for _, p in sorted(best.values(), key=lambda x: x[0], reverse=True)]
    else:
        files = ordered
    stats["kept"] = len(files)
    return files, stats


def iter_files(root: Path):
    """兼容旧调用：yield 去重后的文件。"""
    files, _ = collect_files(root)
    yield from files


def split_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + CHUNK_SIZE)
        out.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return out


def make_client(mode: str):
    """创建 Qdrant 客户端。"""
    from qdrant_client import QdrantClient

    if mode == "local":
        CENTER.mkdir(parents=True, exist_ok=True)
        # 本地持久化，不依赖 Docker
        log(f"[qdrant] local path mode → {CENTER}")
        return QdrantClient(path=str(CENTER))
    log(f"[qdrant] server mode → {QDRANT_HOST}:{QDRANT_PORT}")
    # check_compatibility=False：避免 client 1.18 vs server 1.12 警告/异常中断
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=120,
        check_compatibility=False,
    )


def load_embedder():
    """GPU 优先 embedding + 限制 CPU 线程。"""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, TORCH_THREADS))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    log(f"[embed] {model_name} on {device} threads={TORCH_THREADS} profile={PROFILE}")
    if device == "cuda":
        log(f"[embed] GPU={torch.cuda.get_device_name(0)}")
    model = SentenceTransformer(model_name, device=device)
    return model, device


def ensure_collection(client, name: str, force: bool) -> None:
    from qdrant_client.models import Distance, VectorParams

    names = [c.name for c in client.get_collections().collections]
    if force and name in names:
        client.delete_collection(name)
        names.remove(name)
        log(f"  [force] deleted {name}")
    if name not in names:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log(f"  created collection {name}")


def collection_count(client, name: str) -> int:
    try:
        return int(client.get_collection(name).points_count or 0)
    except Exception:
        return 0


def _ckpt_dir() -> Path:
    """断点目录：与 qdrant_data 同级，重启可续。"""
    d = CENTER / ".checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ckpt_path(collection: str) -> Path:
    return _ckpt_dir() / f"{collection}.json"


def load_checkpoint(collection: str) -> dict:
    """读取断点：{next_start, total, updated_at}。"""
    p = _ckpt_path(collection)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(collection: str, next_start: int, total: int) -> None:
    """每批写完落盘，Ctrl+C / 关机后可 --resume。"""
    payload = {
        "next_start": next_start,
        "total": total,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "done": next_start >= total,
    }
    _ckpt_path(collection).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clear_checkpoint(collection: str) -> None:
    p = _ckpt_path(collection)
    if p.exists():
        p.unlink()


def index_project(client, embedder, device: str, key: str, cfg: dict, force: bool) -> dict:
    """索引单项目：支持暂停后续建（checkpoint + 确定性 point id upsert）。

    慢的主因（占比粗估）：
    1. embedding 编码 ~50–70%（GPU，但有节流 sleep）
    2. Qdrant HNSW 写入 ~20–40%（CPU）
    3. 扫盘分块 ~5–15%（CPU，一次性）
    不是「整库一次性算法」，是「分块后逐批 encode+写入」。
    """
    from qdrant_client.models import PointStruct

    root = Path(cfg["path"])
    col = cfg["collection"]
    stats = {
        "project": key,
        "collection": col,
        "chunks": 0,
        "files": 0,
        "skipped": False,
        "resumed_from": 0,
        "error": None,
    }

    if not root.exists():
        stats["error"] = "path_not_found"
        return stats

    # force：删库 + 删断点，从头来
    if force:
        clear_checkpoint(col)
    ensure_collection(client, col, force=force)

    # ---------- 阶段 0：扫盘分块（相对快；同名去重 + 书籍优先）----------
    file_list, fstats = collect_files(root)
    log(
        f"[{key}] scan kept={fstats['kept']} scanned={fstats['scanned']} "
        f"skip_dup={fstats['skipped_dup']} skip_size={fstats['skipped_size']} "
        f"dedup={'on' if DEDUP_BY_BASENAME else 'off'}"
    )
    # 书籍命中数（zhesi 必看）
    n_books = sum(1 for p in file_list if "书籍" in p.parts)
    if n_books:
        log(f"[{key}] books_included={n_books} (under 书籍/)")

    chunks = []
    for fp in file_list:
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
        for i, ch in enumerate(split_text(text)):
            doc_id = hashlib.md5(f"{key}:{rel}:{i}".encode()).hexdigest()[:16]
            chunks.append((doc_id, ch, rel, i + 1))

    n = len(chunks)
    log(f"[{key}] files={stats['files']} chunks={n}")
    if n == 0:
        stats["error"] = "no_chunks"
        return stats

    # 断点续传：必须以「库里真实点数」为准；server/local 切换后 ckpt 可能失效
    existing = collection_count(client, col)
    ckpt = {} if force else load_checkpoint(col)
    start0 = int(ckpt.get("next_start") or 0)

    # 去重后语料变少，但库里仍是旧「含重复」的点 → 强制重建，避免假 skip
    if (
        not force
        and existing > 0
        and n > 0
        and existing > int(n * 1.15)
    ):
        log(
            f"[{key}] corpus shrunk after dedup (db={existing} > chunks={n}*1.15) "
            "→ force rebuild collection"
        )
        force = True
        clear_checkpoint(col)
        ensure_collection(client, col, force=True)
        existing = 0
        start0 = 0

    # 库已满 ≈ 完成
    if not force and existing >= max(1, int(n * 0.98)):
        save_checkpoint(col, n, n)
        stats["skipped"] = True
        stats["chunks"] = existing
        log(f"[skip] {key}: already ~complete points={existing}/{n}")
        return stats

    # ckpt 说完成但库是空的/很少 → 失效，从头建
    if existing < max(1, int(n * 0.05)) and start0 > 0:
        log(f"[{key}] checkpoint stale (db_points={existing} << {n}), reset to 0")
        start0 = 0
        clear_checkpoint(col)

    # 有部分数据但无可靠 ckpt：从 0 upsert 覆盖（id 确定性，不膨胀）
    if start0 > existing * 2 and existing > 0:
        log(f"[{key}] ckpt next={start0} vs db={existing}, resume from 0 upsert-idempotent")
        start0 = 0

    if start0 > 0:
        stats["resumed_from"] = start0
        log(f"[{key}] RESUME from chunk index {start0}/{n} (checkpoint)")
    micro = GPU_MICRO_BATCH if device == "cuda" else CPU_MICRO_BATCH
    remain = n - start0
    # 纯 sleep 预估（帮助理解「为什么慢」）
    n_batches = max(1, (remain + UPSERT_BATCH - 1) // UPSERT_BATCH)
    sleep_tax = n_batches * (THROTTLE_SLEEP + UPSERT_SLEEP)
    log(
        f"[{key}] profile={PROFILE} micro={micro} upsert_bs={UPSERT_BATCH} "
        f"sleep/批={THROTTLE_SLEEP + UPSERT_SLEEP:.2f}s "
        f"预计空睡≈{_fmt_eta(sleep_tax)}（{n_batches}批）"
    )
    t0 = time.time()

    # 可视化进度条：仅交互终端用 tqdm；重定向到文件时禁用（避免管道/缓冲导致进程异常退出）
    bar = None
    use_tqdm = sys.stdout.isatty() and os.getenv("FORCE_TQDM", "").lower() not in ("0", "false")
    if use_tqdm:
        try:
            from tqdm import tqdm
            bar = tqdm(
                total=remain,
                initial=0,
                unit="chk",
                desc=f"{key}",
                ncols=100,
                mininterval=0.5,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
            )
        except ImportError:
            log(f"[{key}] tqdm 未安装，用文本进度")
    else:
        log(f"[{key}] 文本进度模式（非 TTY / 后台） remain={remain}")

    done_this = 0
    for start in range(start0, n, UPSERT_BATCH):
        end = min(n, start + UPSERT_BATCH)
        part = chunks[start:end]
        docs = [p[1] for p in part]
        emb = embedder.encode(
            docs,
            batch_size=micro,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if THROTTLE_SLEEP > 0:
            time.sleep(THROTTLE_SLEEP)

        points = []
        for j, (doc_id, content, rel, page) in enumerate(part):
            # 确定性 id → 重跑同一批只是覆盖，不会重复膨胀
            pid = int(hashlib.md5(doc_id.encode()).hexdigest()[:15], 16)
            points.append(
                PointStruct(
                    id=pid,
                    vector=emb[j].tolist(),
                    payload={
                        "content": content,
                        "source": rel,
                        "page": page,
                        "doc_id": doc_id,
                        "project": key,
                    },
                )
            )
        # upsert 失败重试（网络/Docker 偶发抖动）
        for attempt in range(3):
            try:
                client.upsert(collection_name=col, points=points)
                break
            except Exception as up_e:
                log(f"  [warn] upsert retry {attempt+1}/3: {up_e}")
                time.sleep(1.5 * (attempt + 1))
                if attempt == 2:
                    raise
        batch_n = len(points)
        stats["chunks"] += batch_n
        done_this += batch_n
        del emb, points, docs
        save_checkpoint(col, end, n)
        if UPSERT_SLEEP > 0:
            time.sleep(UPSERT_SLEEP)

        if bar is not None:
            bar.update(batch_n)
            # 后缀：库名 + 全局百分比 + 速率
            elapsed = max(0.1, time.time() - t0)
            rate = done_this / elapsed
            bar.set_postfix_str(
                f"{end}/{n} ({100 * end / n:.0f}%) {rate:.0f} chk/s", refresh=False
            )
        elif ((start - start0) // UPSERT_BATCH) % 2 == 0:
            elapsed = max(0.1, time.time() - t0)
            rate = done_this / elapsed
            eta = (n - end) / rate if rate > 0 else 0
            log(
                f"  [{key}] {end}/{n} ({100 * end / n:.0f}%) "
                f"{rate:.0f} chk/s ETA {_fmt_eta(eta)}"
            )

    if bar is not None:
        bar.close()
    save_checkpoint(col, n, n)
    elapsed = time.time() - t0
    rate = stats["chunks"] / max(0.1, elapsed)
    log(
        f"[ok] {key} → {col}: wrote {stats['chunks']} this run, "
        f"db_points≈{collection_count(client, col)}, "
        f"{elapsed:.0f}s, {rate:.0f} chk/s"
    )
    return stats

def main() -> int:
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="all", choices=list(PROJECTS) + ["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=["local", "server"], default="server",
                    help="server=Docker可并发(推荐); local=单进程独占磁盘")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="最快档（等同 BUILD_PROFILE=fast）",
    )
    ap.add_argument(
        "--gentle",
        action="store_true",
        help="强节流跟手档（会很慢，仅白天要桌面流畅时用）",
    )
    ap.add_argument(
        "--desktop",
        action="store_true",
        help="桌面可用档（目标占用约40-60%，推荐边用电脑边建库）",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="只查看各库点数与断点，不构建",
    )
    args = ap.parse_args()
    _init_file_log()

    global THROTTLE_SLEEP, UPSERT_SLEEP, GPU_MICRO_BATCH, UPSERT_BATCH, TORCH_THREADS, PROFILE
    if args.fast:
        PROFILE = "fast"
        p = _PROFILES["fast"]
    elif args.gentle:
        PROFILE = "gentle"
        p = _PROFILES["gentle"]
    elif args.desktop:
        PROFILE = "desktop"
        p = _PROFILES["desktop"]
    else:
        p = _PROFILES.get(PROFILE, _PROFILES["balanced"])
    GPU_MICRO_BATCH = int(os.getenv("EMBED_MICRO_BATCH", str(p["micro"])))
    THROTTLE_SLEEP = float(os.getenv("THROTTLE_SLEEP", str(p["throttle"])))
    UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", str(p["upsert_bs"])))
    UPSERT_SLEEP = float(os.getenv("UPSERT_SLEEP", str(p["upsert_sleep"])))
    TORCH_THREADS = int(os.getenv("TORCH_THREADS", str(p["threads"])))
    # desktop/gentle：压低 BLAS 线程，避免 CPU 扫盘+编码时顶满
    if PROFILE in ("desktop", "gentle"):
        for _k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[_k] = "1"
    log(
        f"[mode] profile={PROFILE} micro={GPU_MICRO_BATCH} "
        f"upsert_bs={UPSERT_BATCH} sleep={THROTTLE_SLEEP}+{UPSERT_SLEEP}s/批"
    )
    if PROFILE == "gentle":
        log("[hint] gentle 很慢：每批空睡约 0.5s；要恢复速度请去掉 --gentle 或加 --fast")
    elif PROFILE == "desktop":
        log("[hint] desktop：目标占用约40-60%；更快用 --balanced/--fast，更省用 --gentle")

    keys = list(PROJECTS) if args.project == "all" else [args.project]
    log(f"center={CENTER} mode={args.mode} time={datetime.now().isoformat(timespec='seconds')}")
    if args.dry_run:
        for k in keys:
            p = Path(PROJECTS[k]["path"])
            log(f"  {k:12} {PROJECTS[k]['collection']:18} exists={p.exists()} {p}")
        return 0

    if args.status:
        client = make_client(args.mode)
        log("=== 库状态 / 断点（可暂停后继续）===")
        for k in keys:
            col = PROJECTS[k]["collection"]
            cnt = collection_count(client, col)
            ck = load_checkpoint(col)
            ns = ck.get("next_start", "-")
            tot = ck.get("total", "-")
            done = ck.get("done", False)
            log(f"  {k:12} {col:18} points={cnt:7}  ckpt={ns}/{tot} done={done}")
        log("继续构建：同一命令去掉 --status 即可（默认断点续传，勿加 --force）")
        log("彻底重来：加 --force（会删库+断点）")
        return 0

    # Windows：降优先级；desktop 再限制 CPU 亲和性到约一半逻辑核
    try:
        import psutil

        proc = psutil.Process()
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        log("[os] process priority = below normal")
        if PROFILE in ("desktop", "gentle"):
            try:
                cores = list(range(psutil.cpu_count(logical=True) or 4))
                # 用户要求 CPU 不超过约 50%：desktop≈40% 核，gentle≈30% 核
                ratio = 0.3 if PROFILE == "gentle" else 0.4
                keep = max(1, int(len(cores) * ratio))
                # 优先用后半段核，前半段留给前台
                affinity = cores[-keep:] if len(cores) > 1 else cores
                proc.cpu_affinity(affinity)
                log(f"[os] cpu_affinity={affinity} (≈{100 * keep / len(cores):.0f}% cores, cap≤50%)")
            except Exception as aff_e:
                log(f"[os] cpu_affinity skip: {aff_e}")
    except Exception as e:
        log(f"[os] nice skip: {e}")

    try:
        client = make_client(args.mode)
        embedder, device = load_embedder()
    except Exception as e:
        log(f"[FATAL] init failed: {e}\n{traceback.format_exc()}")
        return 2

    results = []
    t0 = time.time()
    for k in keys:
        log(f"\n>>> {k}")
        try:
            results.append(index_project(client, embedder, device, k, PROJECTS[k], args.force))
        except Exception as e:
            log(f"[ERR] {k}: {e}\n{traceback.format_exc()}")
            results.append({"project": k, "error": str(e), "chunks": 0})
        if PROFILE == "gentle":
            time.sleep(1.0)
        elif PROFILE == "desktop":
            time.sleep(0.6)
        elif PROFILE == "balanced":
            time.sleep(0.2)

    reg = {
        "backend": "qdrant",
        "mode": args.mode,
        "center": str(CENTER),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 1),
        "device": device,
        "projects": results,
    }
    out = CENTER.parent / "向量库注册表_qdrant.json"
    out.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"registry → {out} elapsed={reg['elapsed_sec']}s")
    ok_n = sum(1 for r in results if not r.get("error"))
    log(f"[done] ok={ok_n}/{len(results)}")
    return 0 if ok_n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
