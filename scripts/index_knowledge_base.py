"""知识库索引脚本 — 扫描 .md 文件并索引到 ChromaDB

功能：
1. 扫描心理人际目录和哲思灵智目录下的所有 .md 文件
2. 使用 RecursiveCharacterTextSplitter 分片 (chunk_size=800, overlap=200)
3. 存入 ChromaDB 的两个 collection：psychology_kb / general_kb
4. 支持增量更新（已索引的文件路径跳过）
5. 显示进度和统计信息
"""

# 必须在所有 import 之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'       # 使用已缓存的模型，避免在线检查失败
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # transformers 也设为离线模式

import sys
import gc
import hashlib
import time
import traceback
from pathlib import Path

# 设置项目根目录并加入 sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置工作目录
os.chdir(str(PROJECT_ROOT))

# 启用行缓冲，确保 print 输出实时刷新（避免管道模式下缓冲延迟）
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from src.retrieval.indexer import Indexer
from src.config import EMBEDDING_MODEL, DEVICE

# ============ 配置 ============

# 待索引的目录
PSYCHOLOGY_DIR = r"d:\文档\ai提问相关\心理人际"
GENERAL_DIR = r"d:\文档\ai提问相关\哲思灵智\rag-docs\converted"

# 排除的目录名（不扫描这些目录下的文件）
EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', '.claude', '.trae',
    '.ace-tool', '.venv', 'venv', '.streamlit',
}

# 分块配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# 单文件最大分片数（超过则跳过，避免字典类大文件耗时过长）
MAX_CHUNKS_PER_FILE = 5000


def scan_md_files(dir_path: str) -> list:
    """递归扫描目录下所有 .md 文件，排除隐藏文件和指定目录

    Args:
        dir_path: 要扫描的目录路径

    Returns:
        .md 文件 Path 列表（已排序）
    """
    root = Path(dir_path)
    if not root.exists():
        print(f"  [警告] 目录不存在: {dir_path}")
        return []

    md_files = []
    for filepath in sorted(root.rglob("*.md")):
        # 跳过隐藏文件（以 . 开头）
        if filepath.name.startswith("."):
            continue
        # 跳过排除目录中的文件
        parts = set(filepath.parts)
        if parts & EXCLUDE_DIRS:
            continue
        # 跳过备份文件
        if filepath.suffix == ".md" and filepath.name.endswith((".bak", ".backup", ".bak2")):
            continue
        md_files.append(filepath)

    return md_files


def get_indexed_filepaths(collection) -> set:
    """从 ChromaDB collection 中获取已索引的 file_path 集合

    Args:
        collection: ChromaDB collection 对象

    Returns:
        已索引文件路径的集合
    """
    indexed = set()
    try:
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
        for meta in metadatas:
            if meta and "file_path" in meta:
                indexed.add(meta["file_path"])
    except Exception as e:
        print(f"  [警告] 获取已有索引元数据失败: {e}")
    return indexed


def index_directory(indexer: Indexer, dir_path: str, collection_name: str) -> dict:
    """索引目录下所有 .md 文件到指定 ChromaDB collection

    Args:
        indexer: Indexer 实例
        dir_path: 要扫描的目录路径
        collection_name: ChromaDB collection 名称

    Returns:
        统计信息字典
    """
    print(f"\n{'=' * 60}")
    print(f"  目录: {dir_path}")
    print(f"  Collection: {collection_name}")
    print(f"{'=' * 60}")

    # 步骤 1: 扫描文件
    print(f"\n  [1/4] 扫描 .md 文件...")
    md_files = scan_md_files(dir_path)
    print(f"        找到 {len(md_files)} 个 .md 文件")

    if not md_files:
        print("  无文件需要索引")
        return {
            "total_files": 0, "indexed_files": 0,
            "skipped_files": 0, "total_chunks": 0, "errors": 0,
        }

    # 步骤 2: 初始化索引器（加载 embedding 模型）
    print(f"\n  [2/4] 初始化索引器（加载 embedding 模型）...")
    collection = indexer.get_collection()
    embedder = indexer._get_embedder()
    splitter = indexer._splitter
    print(f"        模型: {EMBEDDING_MODEL} | 设备: {DEVICE}")

    # 步骤 3: 检查已索引文件
    print(f"\n  [3/4] 检查已索引文件（增量更新）...")
    indexed_paths = get_indexed_filepaths(collection)
    print(f"        数据库中已有: {len(indexed_paths)} 个文件")

    stats = {
        "total_files": len(md_files),
        "indexed_files": 0,
        "skipped_files": 0,
        "total_chunks": 0,
        "errors": 0,
    }

    # 步骤 4: 开始索引
    print(f"\n  [4/4] 开始索引...")
    start_time = time.time()

    for i, filepath in enumerate(md_files, 1):
        file_path_str = str(filepath)
        filename = filepath.name

        # 进度显示
        elapsed = time.time() - start_time
        speed = i / elapsed if elapsed > 0 else 0
        print(f"\n  [{i}/{len(md_files)}] ({elapsed:.0f}s, {speed:.1f}文件/秒) {filename}")

        # 增量更新：跳过已索引文件
        if file_path_str in indexed_paths:
            print(f"      -> 跳过（已索引）")
            stats["skipped_files"] += 1
            continue

        try:
            # 读取文件内容（UTF-8 编码，忽略无法解码的字符）
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if len(content) < 20:
                print(f"      -> 跳过（内容过短, {len(content)} 字符）")
                stats["skipped_files"] += 1
                continue

            # 使用 RecursiveCharacterTextSplitter 分片
            chunks = splitter.split_text(content)
            print(f"      -> 分片: {len(chunks)} 块")

            # 超大文件跳过（避免字典类文件耗时过长）
            if len(chunks) > MAX_CHUNKS_PER_FILE:
                print(f"      -> 跳过（分片数 {len(chunks)} 超过上限 {MAX_CHUNKS_PER_FILE}）")
                stats["skipped_files"] += 1
                continue

            # 生成 embedding 并写入（分批处理，避免内存溢出）
            batch_size = 32
            for j in range(0, len(chunks), batch_size):
                batch_texts = chunks[j:j + batch_size]
                batch_embeddings = embedder.encode(batch_texts).tolist()

                # 准备本批数据
                batch_ids = []
                batch_docs = []
                batch_metas = []
                for k, chunk in enumerate(batch_texts):
                    doc_id = hashlib.md5(f"{filepath}:{j + k}".encode()).hexdigest()[:12]
                    batch_ids.append(doc_id)
                    batch_docs.append(chunk)
                    batch_metas.append({
                        "source": filename,
                        "page": j + k + 1,
                        "file_path": file_path_str,
                    })

                # 使用 upsert 避免重复 ID 报错
                collection.upsert(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas,
                    embeddings=batch_embeddings,
                )

            stats["indexed_files"] += 1
            stats["total_chunks"] += len(chunks)
            print(f"      -> 完成: {len(chunks)} 块已写入")

            # 释放内存
            del chunks, content
            gc.collect()

        except Exception as e:
            print(f"      -> 错误: {e}")
            traceback.print_exc()
            stats["errors"] += 1
            gc.collect()

    # 统计信息
    elapsed = time.time() - start_time
    print(f"\n  {'─' * 50}")
    print(f"  完成! 耗时: {elapsed:.1f}s")
    print(f"  总文件数:     {stats['total_files']}")
    print(f"  新索引文件:   {stats['indexed_files']}")
    print(f"  跳过文件:     {stats['skipped_files']}")
    print(f"  总分片数:     {stats['total_chunks']}")
    print(f"  错误数:       {stats['errors']}")

    # 显示 collection 当前总量
    try:
        total_in_db = collection.count()
        print(f"  数据库总量:   {total_in_db} 块")
    except Exception:
        pass

    return stats


def main():
    """主函数：索引两个知识库目录"""
    print("=" * 60)
    print("  知识库索引工具 (Knowledge Base Indexer)")
    print("=" * 60)
    print(f"  Embedding 模型: {EMBEDDING_MODEL}")
    print(f"  设备:           {DEVICE}")
    print(f"  分片配置:       chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  工作目录:       {os.getcwd()}")
    print(f"  ChromaDB 路径:  {os.path.join(str(PROJECT_ROOT), 'chroma_db')}")

    total_start = time.time()

    # ========== 1. 索引心理人际目录 -> psychology_kb ==========
    print(f"\n{'#' * 60}")
    print("#  [1/2] 心理人际知识库 (psychology_kb)")
    print(f"{'#' * 60}")

    psych_indexer = Indexer(collection_name="psychology_kb")
    psych_stats = index_directory(
        psych_indexer, PSYCHOLOGY_DIR, "psychology_kb"
    )

    # ========== 2. 索引哲思灵智目录 -> general_kb ==========
    print(f"\n{'#' * 60}")
    print("#  [2/2] 哲思灵智知识库 (general_kb)")
    print(f"{'#' * 60}")

    general_indexer = Indexer(collection_name="general_kb")
    general_stats = index_directory(
        general_indexer, GENERAL_DIR, "general_kb"
    )

    # ========== 汇总 ==========
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("  索引完成汇总")
    print(f"{'=' * 60}")
    print(f"  心理人际 (psychology_kb):")
    print(f"    新索引文件:   {psych_stats['indexed_files']}")
    print(f"    跳过文件:     {psych_stats['skipped_files']}")
    print(f"    总分片数:     {psych_stats['total_chunks']}")
    print(f"    错误数:       {psych_stats['errors']}")
    print(f"  哲思灵智 (general_kb):")
    print(f"    新索引文件:   {general_stats['indexed_files']}")
    print(f"    跳过文件:     {general_stats['skipped_files']}")
    print(f"    总分片数:     {general_stats['total_chunks']}")
    print(f"    错误数:       {general_stats['errors']}")
    print(f"  {'─' * 40}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    print(f"  总分片: {psych_stats['total_chunks'] + general_stats['total_chunks']}")


if __name__ == "__main__":
    main()
