"""
统一迁移到 LanceDB
将所有 ChromaDB 数据迁移到 LanceDB
"""
import sys
sys.path.insert(0, '.')

import chromadb
import lancedb
import pyarrow as pa
from pathlib import Path
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("🔄 统一迁移到 LanceDB")
print("=" * 70)
print()

# 配置
MIGRATIONS = [
    {
        "name": "核心数据库",
        "chroma_path": "./deep-rag/chroma_db",
        "chroma_collection": "full_docs",
        "lance_path": "./deep-rag/lancedb_core",
        "lance_table": "core_docs",
    },
    {
        "name": "外围数据库",
        "chroma_path": "./deep-rag/chroma_db_extended",
        "chroma_collection": "extended_docs",
        "lance_path": "./deep-rag/lancedb_extended",
        "lance_table": "extended_docs",
    },
]

BATCH_SIZE = 1000

# 加载 Embedding 模型
print("📦 加载 Embedding 模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("✅ 模型已加载")
print()

# 定义 LanceDB Schema
schema = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("source", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), 512)),
])

# 迁移每个数据库
for migration in MIGRATIONS:
    print(f"📊 迁移: {migration['name']}")
    print("-" * 70)

    chroma_path = Path(migration["chroma_path"])
    lance_path = Path(migration["lance_path"])

    # 检查 ChromaDB 是否存在
    if not chroma_path.exists():
        print(f"⚠️  ChromaDB 不存在: {chroma_path}")
        print()
        continue

    # 检查是否为空
    if not any(chroma_path.iterdir()):
        print(f"⚠️  ChromaDB 为空: {chroma_path}")
        print()
        continue

    try:
        # 加载 ChromaDB
        print(f"📦 加载 ChromaDB: {chroma_path}")
        chroma_client = chromadb.PersistentClient(path=str(chroma_path))

        try:
            chroma_collection = chroma_client.get_collection(migration["chroma_collection"])
            total_docs = chroma_collection.count()
            print(f"✅ ChromaDB: {total_docs:,} 个文档块")
        except Exception as e:
            print(f"⚠️  集合不存在: {migration['chroma_collection']}")
            print()
            continue

        if total_docs == 0:
            print(f"⚠️  ChromaDB 为空")
            print()
            continue

        # 创建 LanceDB
        print(f"📦 创建 LanceDB: {lance_path}")
        lance_db = lancedb.connect(str(lance_path))

        # 删除旧表（如果存在）
        try:
            lance_db.drop_table(migration["lance_table"])
            print(f"🗑️  删除旧表: {migration['lance_table']}")
        except Exception:
            pass

        # 创建新表
        lance_table = lance_db.create_table(migration["lance_table"], schema=schema)
        print(f"✅ LanceDB 表已创建")
        print()

        # 分批迁移
        print(f"🔄 开始迁移（批次大小: {BATCH_SIZE}）...")
        num_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

        migrated_count = 0
        for batch_idx in tqdm(range(num_batches), desc="迁移进度"):
            offset = batch_idx * BATCH_SIZE
            limit = min(BATCH_SIZE, total_docs - offset)

            # 从 ChromaDB 读取
            result = chroma_collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )

            if not result["ids"]:
                continue

            # 准备数据
            data = []
            for i, doc_id in enumerate(result["ids"]):
                content = result["documents"][i]
                metadata = result["metadatas"][i] if result["metadatas"] else {}

                # 生成向量（如果 ChromaDB 没有存储向量）
                vector = model.encode([content])[0]

                data.append({
                    "doc_id": doc_id,
                    "content": content,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 1),
                    "vector": vector.tolist(),
                })

            # 写入 LanceDB
            if data:
                lance_table.add(data)
                migrated_count += len(data)

        print()
        print(f"✅ 迁移完成: {migrated_count:,} 个文档块")

        # 验证
        lance_count = len(lance_table.to_pandas())
        print(f"✅ 验证: LanceDB 有 {lance_count:,} 个文档块")

        if lance_count == total_docs:
            print(f"✅ 数据完整性验证通过")
        else:
            print(f"⚠️  数据不一致: ChromaDB={total_docs}, LanceDB={lance_count}")

        print()

    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        print()
        continue

print("=" * 70)
print("🎉 迁移完成")
print("=" * 70)
print()

# 统计
print("📊 迁移统计:")
print()
for migration in MIGRATIONS:
    lance_path = Path(migration["lance_path"])
    if lance_path.exists():
        try:
            lance_db = lancedb.connect(str(lance_path))
            lance_table = lance_db.open_table(migration["lance_table"])
            count = len(lance_table.to_pandas())
            size = sum(f.stat().st_size for f in lance_path.rglob("*") if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"✅ {migration['name']}: {count:,} 文档, {size_mb:.1f} MB")
        except Exception:
            print(f"⚠️  {migration['name']}: 无法读取")
    else:
        print(f"⚠️  {migration['name']}: 不存在")

print()
print("🗑️  下一步: 清理 ChromaDB 目录")
print("   运行: python cleanup_chromadb.py")
