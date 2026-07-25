"""
多进程预加载索引器 v6
- 多进程读取文件（绕过GIL）
- 大缓冲区预加载
- GPU持续工作
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
import time
import hashlib
import gc
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from multiprocessing import Process, Queue, Event
import psutil

# 配置
ROOT_DIR = "D:/文档/ai提问相关"
MAX_DB_SIZE_MB = 2000
BATCH_SIZE = 512  # 更大的GPU批次
MAX_FILE_SIZE_MB = 2
NUM_READERS = 8  # 8个进程
PRELOAD_BATCHES = 20  # 预加载20个批次
MAX_MEMORY_GB = 28.8  # 32GB * 90%
MAX_VRAM_GB = 7.2  # 8GB * 90%

SUPPORTED_EXTS = {
    ".md", ".txt", ".py", ".js", ".ts", ".java",
    ".sh", ".bat", ".json", ".yaml", ".yml",
    ".xml", ".html", ".css", ".log", ".csv",
}

BLACKLIST_DIRS = {
    "node_modules", "__pycache__", ".git", ".svn",
    "venv", ".venv", "env", ".cache", ".npm",
    "dist", "build", "target", "obj", "bin",
    "Steam", "Epic Games", "WeGame",
    "Temp", "tmp", "temp", "Cache", "BaiduNetdiskTmp",
    ".pytest_cache", ".benchmarks",
}


def should_skip_dir(dir_path: Path) -> bool:
    name = dir_path.name
    if name in BLACKLIST_DIRS:
        return True
    if name.startswith(".") and name not in {".config", ".ssh"}:
        return True
    return False


def get_db_size_mb() -> float:
    db_path = Path("./chroma_db/chroma.sqlite3")
    if db_path.exists():
        return db_path.stat().st_size / 1024 / 1024
    return 0


def reader_process(file_queue, batch_queue, root_str, stop_event):
    """多进程读取器"""
    root = Path(root_str)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "]
    )

    batch_texts = []
    batch_ids = []
    batch_metadata = []

    while not stop_event.is_set():
        try:
            filepath = file_queue.get(timeout=1)
            if filepath is None:
                break

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if len(content) < 20:
                continue

            chunks = splitter.split_text(content)

            try:
                rel_path = str(Path(filepath).relative_to(root))
            except ValueError:
                rel_path = str(filepath)

            for j, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{filepath}:{j}".encode()).hexdigest()[:16]
                batch_texts.append(chunk)
                batch_ids.append(doc_id)
                batch_metadata.append({
                    'source': rel_path,
                    'page': j + 1,
                    'type': Path(filepath).suffix,
                })

                # 达到批次大小
                if len(batch_texts) >= BATCH_SIZE:
                    batch_queue.put({
                        'texts': batch_texts,
                        'ids': batch_ids,
                        'metadatas': batch_metadata
                    })
                    batch_texts = []
                    batch_ids = []
                    batch_metadata = []

        except Exception:
            continue

    # 处理剩余
    if batch_texts:
        batch_queue.put({
            'texts': batch_texts,
            'ids': batch_ids,
            'metadatas': batch_metadata
        })


def main():
    print("=" * 70)
    print("🚀 多进程预加载索引器 v6")
    print("=" * 70)
    print(f"根目录: {ROOT_DIR}")
    print(f"GPU批次: {BATCH_SIZE}")
    print(f"读取进程: {NUM_READERS}")
    print(f"预加载批次: {PRELOAD_BATCHES}")
    print(f"内存限制: {MAX_MEMORY_GB:.1f} GB")
    print(f"显存限制: {MAX_VRAM_GB:.1f} GB")
    print()

    # 加载模型
    print("📦 加载GPU模型...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")
    print("✅ 模型已加载")
    print()

    # 连接数据库
    print("📦 连接数据库...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("full_docs")
    initial_count = collection.count()
    print(f"✅ 数据库已连接（当前 {initial_count:,} 个文档块）")
    print()

    # 收集文件
    print(f"📁 扫描文件...")
    root = Path(ROOT_DIR)
    all_files = []

    for ext in SUPPORTED_EXTS:
        for item in root.rglob(f"*{ext}"):
            try:
                should_skip = False
                for parent in item.parents:
                    if parent == root:
                        break
                    if should_skip_dir(parent):
                        should_skip = True
                        break

                if should_skip:
                    continue

                size_mb = item.stat().st_size / 1024 / 1024
                if size_mb > MAX_FILE_SIZE_MB:
                    continue

                all_files.append(str(item))
            except Exception:
                continue

    print(f"找到 {len(all_files)} 个文件")
    print()

    # 创建队列
    file_queue = Queue(maxsize=1000)
    batch_queue = Queue(maxsize=PRELOAD_BATCHES)  # 预加载缓冲
    stop_event = Event()

    # 启动读取进程
    print("🚀 启动读取进程...")
    readers = []
    for i in range(NUM_READERS):
        p = Process(
            target=reader_process,
            args=(file_queue, batch_queue, str(root), stop_event)
        )
        p.start()
        readers.append(p)

    # 填充文件队列
    for filepath in all_files:
        file_queue.put(filepath)

    for _ in range(NUM_READERS):
        file_queue.put(None)

    # GPU处理循环
    print("🚀 GPU开始处理...")
    print()

    total_chunks = 0
    batches_processed = 0
    start_time = time.time()

    while True:
        # 检查内存
        mem_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        vram_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024

        if mem_gb > MAX_MEMORY_GB:
            print(f"⚠️ 内存超限 ({mem_gb:.1f}/{MAX_MEMORY_GB:.1f} GB)，清理中...")
            gc.collect()
            time.sleep(2)
            continue

        if vram_gb > MAX_VRAM_GB:
            print(f"⚠️ 显存超限 ({vram_gb:.1f}/{MAX_VRAM_GB:.1f} GB)，清理中...")
            torch.cuda.empty_cache()
            time.sleep(2)
            continue

        # 检查数据库大小
        if batches_processed % 10 == 0:
            current_size = get_db_size_mb()
            if current_size >= MAX_DB_SIZE_MB:
                print(f"⚠️ 达到大小限制 ({current_size:.1f} MB)")
                break

        # 获取批次
        try:
            batch = batch_queue.get(timeout=5)
        except:
            # 检查是否所有进程都结束
            if all(not p.is_alive() for p in readers) and batch_queue.empty():
                break
            continue

        # GPU向量化
        embeddings = model.encode(
            batch['texts'],
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 写入数据库
        collection.add(
            documents=batch['texts'],
            ids=batch['ids'],
            metadatas=batch['metadatas'],
            embeddings=embeddings.tolist()
        )

        total_chunks += len(batch['texts'])
        batches_processed += 1

        # 进度报告
        if batches_processed % 10 == 0:
            elapsed = time.time() - start_time
            speed = total_chunks / elapsed if elapsed > 0 else 0
            print(f"[批次 {batches_processed}] 块: {total_chunks:,} | 速度: {speed:.0f} 块/秒 | 内存: {mem_gb:.1f}GB | 显存: {vram_gb:.1f}GB")

        # 清理
        torch.cuda.empty_cache()
        if batches_processed % 50 == 0:
            gc.collect()

    # 清理
    stop_event.set()
    for p in readers:
        p.terminate()
        p.join()

    # 统计
    elapsed = time.time() - start_time
    final_size = get_db_size_mb()
    final_count = collection.count()
    new_chunks = final_count - initial_count

    print()
    print("=" * 70)
    print("✅ 索引完成")
    print("=" * 70)
    print(f"处理文件: {len(all_files)}")
    print(f"新增文档块: {new_chunks:,}")
    print(f"数据库总块数: {final_count:,}")
    print(f"数据库大小: {final_size:.1f} MB ({final_size/1024:.2f} GB)")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"平均速度: {total_chunks/elapsed:.0f} 块/秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
