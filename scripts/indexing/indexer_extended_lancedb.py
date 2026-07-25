"""
外围数据库索引器 - LanceDB 版本
4线程：2读取 + 1GPU + 1写入（LanceDB 快30x）
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
import time
import hashlib
import gc
import torch
from sentence_transformers import SentenceTransformer
import lancedb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psutil
from threading import Thread
from queue import Queue
import threading
import shutil

# 配置
ROOT_DIR = "D:/文档"
EXCLUDE_DIR = "ai提问相关"
DB_PATH = "./lancedb_extended"
MAX_DB_SIZE_MB = 3000
BATCH_SIZE = 256
MAX_FILE_SIZE_MB = 2
NUM_READERS = 2
MAX_MEMORY_GB = 28.8
MAX_VRAM_GB = 7.2
QUEUE_SIZE = 10

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


def should_skip_dir(dir_path: Path, root: Path) -> bool:
    """是否跳过目录"""
    try:
        rel_path = dir_path.relative_to(root)
        if str(rel_path).startswith(EXCLUDE_DIR):
            return True
    except ValueError:
        pass

    name = dir_path.name
    if name in BLACKLIST_DIRS:
        return True
    if name.startswith(".") and name not in {".config", ".ssh"}:
        return True
    return False


def get_db_size_mb(db_path: str) -> float:
    """获取数据库大小"""
    if Path(db_path).exists():
        return sum(f.stat().st_size for f in Path(db_path).rglob('*') if f.is_file()) / 1024 / 1024
    return 0


def reader_thread(file_queue, batch_queue, root, splitter, stop_event, stats):
    """读取线程：从文件队列读取并分块"""
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
                rel_path = str(filepath.relative_to(root))
            except ValueError:
                rel_path = str(filepath)

            for j, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{filepath}:{j}".encode()).hexdigest()[:16]
                batch_texts.append(chunk)
                batch_ids.append(doc_id)
                batch_metadata.append({
                    'source': rel_path,
                    'page': j + 1,
                    'type': filepath.suffix,
                })

                if len(batch_texts) >= BATCH_SIZE:
                    batch_queue.put({
                        'texts': batch_texts,
                        'ids': batch_ids,
                        'metadatas': batch_metadata
                    })
                    batch_texts = []
                    batch_ids = []
                    batch_metadata = []

            stats['files_processed'] += 1

        except Exception:
            continue

    if batch_texts:
        batch_queue.put({
            'texts': batch_texts,
            'ids': batch_ids,
            'metadatas': batch_metadata
        })


def gpu_thread(batch_queue, embedding_queue, model, stop_event, stats):
    """GPU线程：向量化"""
    batches_processed = 0
    total_chunks = 0

    while not stop_event.is_set():
        try:
            vram_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024

            if vram_gb > MAX_VRAM_GB:
                print(f"⚠️ 显存超限 ({vram_gb:.1f}/{MAX_VRAM_GB:.1f} GB)，清理中...")
                torch.cuda.empty_cache()
                time.sleep(2)
                continue

            batch = batch_queue.get(timeout=5)
            if batch is None:
                embedding_queue.put(None)
                break

            try:
                embeddings = model.encode(
                    batch['texts'],
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"❌ GPU显存不足: {e}")
                    torch.cuda.empty_cache()
                    stop_event.set()
                    break
                else:
                    raise

            embedding_queue.put({
                'texts': batch['texts'],
                'ids': batch['ids'],
                'metadatas': batch['metadatas'],
                'embeddings': embeddings
            })

            total_chunks += len(batch['texts'])
            batches_processed += 1

            torch.cuda.empty_cache()

        except Exception as e:
            if not stop_event.is_set():
                continue

    stats['total_chunks'] = total_chunks
    stats['batches_processed'] = batches_processed


def writer_thread(embedding_queue, table, stop_event, stats):
    """写入线程：LanceDB 写入"""
    batches_written = 0
    start_time = time.time()

    while not stop_event.is_set():
        try:
            mem_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024

            if mem_gb > MAX_MEMORY_GB:
                print(f"⚠️ 内存超限 ({mem_gb:.1f}/{MAX_MEMORY_GB:.1f} GB)，清理中...")
                gc.collect()
                time.sleep(2)
                continue

            if batches_written % 10 == 0:
                current_size = get_db_size_mb(DB_PATH)
                if current_size >= MAX_DB_SIZE_MB:
                    print(f"⚠️ 达到大小限制 ({current_size:.1f} MB)")
                    stop_event.set()
                    break

            batch = embedding_queue.get(timeout=5)
            if batch is None:
                break

            # 转换为 LanceDB 格式
            data = []
            for i in range(len(batch['texts'])):
                data.append({
                    'id': batch['ids'][i],
                    'text': batch['texts'][i],
                    'vector': batch['embeddings'][i].tolist(),
                    'source': batch['metadatas'][i].get('source', ''),
                    'page': batch['metadatas'][i].get('page', 0),
                    'type': batch['metadatas'][i].get('type', ''),
                })

            # 写入 LanceDB（快30x）
            table.add(data)

            batches_written += 1

            if batches_written % 10 == 0:
                elapsed = time.time() - start_time
                total_chunks = stats.get('total_chunks', 0)
                speed = total_chunks / elapsed if elapsed > 0 else 0
                current_size = get_db_size_mb(DB_PATH)
                files_done = stats.get('files_processed', 0)
                print(f"[文件 {files_done}] 批次 {batches_written} | 块: {total_chunks:,} | 速度: {speed:.0f} 块/秒 | DB: {current_size:.0f}MB | 内存: {mem_gb:.1f}GB")

            if batches_written % 50 == 0:
                gc.collect()

        except Exception as e:
            if not stop_event.is_set():
                continue


def main():
    print("=" * 70)
    print("🚀 外围数据库索引器（LanceDB 4线程版）")
    print("=" * 70)
    print(f"根目录: {ROOT_DIR}")
    print(f"排除目录: {EXCLUDE_DIR}")
    print(f"数据库路径: {DB_PATH}")
    print(f"大小限制: {MAX_DB_SIZE_MB} MB ({MAX_DB_SIZE_MB/1024:.1f} GB)")
    print()

    try:
        # 检查初始内存
        mem_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        print(f"当前内存: {mem_gb:.1f} GB / {MAX_MEMORY_GB:.1f} GB")

        if mem_gb > MAX_MEMORY_GB * 0.7:
            print(f"❌ 内存不足，无法启动（需要至少 {MAX_MEMORY_GB * 0.3:.1f} GB 可用）")
            return

        # 加载模型
        print("📦 加载GPU模型...")
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")

        vram_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        print(f"✅ 模型已加载（显存: {vram_gb:.1f} GB / {MAX_VRAM_GB:.1f} GB）")

        if vram_gb > MAX_VRAM_GB * 0.5:
            print(f"⚠️ 显存占用过高，可能不足以处理批次")
        print()

        # 创建/加载 LanceDB
        print("📦 创建 LanceDB...")
        db = lancedb.connect(DB_PATH)

        # 检查是否已存在
        try:
            table = db.open_table("extended_docs")
            initial_count = len(table)
            print(f"✅ 数据库已存在（当前 {initial_count:,} 个文档块）")
        except:
            table = None
            initial_count = 0
            print(f"✅ 数据库已创建（新建）")
        print()

        # 文本分块器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "]
        )

        # 收集文件
        print(f"📁 扫描文件（排除 {EXCLUDE_DIR}）...")
        root = Path(ROOT_DIR)
        all_files = []

        for ext in SUPPORTED_EXTS:
            for item in root.rglob(f"*{ext}"):
                try:
                    should_skip = False
                    for parent in item.parents:
                        if parent == root:
                            break
                        if should_skip_dir(parent, root):
                            should_skip = True
                            break

                    if should_skip:
                        continue

                    size_mb = item.stat().st_size / 1024 / 1024
                    if size_mb > MAX_FILE_SIZE_MB:
                        continue

                    all_files.append(item)
                except Exception:
                    continue

        print(f"找到 {len(all_files)} 个文件")
        print()

        # 创建队列和事件
        file_queue = Queue(maxsize=1000)
        batch_queue = Queue(maxsize=QUEUE_SIZE)
        embedding_queue = Queue(maxsize=QUEUE_SIZE)
        stop_event = threading.Event()
        stats = {'files_processed': 0, 'total_chunks': 0, 'batches_processed': 0}

        # 启动线程
        print(f"🚀 启动 4 个线程（{NUM_READERS} 读取 + 1 GPU + 1 写入）...")
        print()

        readers = []
        for i in range(NUM_READERS):
            t = Thread(
                target=reader_thread,
                args=(file_queue, batch_queue, root, splitter, stop_event, stats)
            )
            t.start()
            readers.append(t)

        gpu = Thread(
            target=gpu_thread,
            args=(batch_queue, embedding_queue, model, stop_event, stats)
        )
        gpu.start()

        # 创建临时表用于首次写入
        if table is None:
            # 等待第一个批次
            first_batch = embedding_queue.get(timeout=60)
            if first_batch is not None:
                data = []
                for i in range(len(first_batch['texts'])):
                    data.append({
                        'id': first_batch['ids'][i],
                        'text': first_batch['texts'][i],
                        'vector': first_batch['embeddings'][i].tolist(),
                        'source': first_batch['metadatas'][i].get('source', ''),
                        'page': first_batch['metadatas'][i].get('page', 0),
                        'type': first_batch['metadatas'][i].get('type', ''),
                    })
                table = db.create_table("extended_docs", data=data)
                print(f"✅ 表已创建，写入首批 {len(data)} 个文档")

        writer = Thread(
            target=writer_thread,
            args=(embedding_queue, table, stop_event, stats)
        )
        writer.start()

        # 填充文件队列
        start_time = time.time()
        for filepath in all_files:
            file_queue.put(filepath)

        for _ in range(NUM_READERS):
            file_queue.put(None)

        for t in readers:
            t.join()

        batch_queue.put(None)
        gpu.join()
        writer.join()

        # 统计
        elapsed = time.time() - start_time
        final_size = get_db_size_mb(DB_PATH)
        final_count = len(table)
        new_chunks = final_count - initial_count

        print()
        print("=" * 70)
        print("✅ 外围数据库索引完成")
        print("=" * 70)
        print(f"处理文件: {len(all_files)}")
        print(f"新增文档块: {new_chunks:,}")
        print(f"数据库总块数: {final_count:,}")
        print(f"数据库大小: {final_size:.1f} MB ({final_size/1024:.2f} GB)")
        print(f"总耗时: {elapsed/60:.1f} 分钟")
        if stats['total_chunks'] > 0:
            print(f"平均速度: {stats['total_chunks']/elapsed:.0f} 块/秒")
        print("=" * 70)

    except KeyboardInterrupt:
        print()
        print("⚠️ 用户中断")
        stop_event.set()
    except Exception as e:
        print()
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
