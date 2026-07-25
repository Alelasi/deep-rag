"""
清理所有 ChromaDB 目录
迁移到 LanceDB 后执行
"""
import shutil
from pathlib import Path

print("=" * 70)
print("🗑️  清理 ChromaDB 目录")
print("=" * 70)
print()

# 要清理的目录
CHROMADB_DIRS = [
    "./deep-rag/chroma_db",
    "./deep-rag/chroma_db_extended",
    "./deep-rag/chroma_db_backup_1021mb_20260527_190236",
    "./deep-rag/chroma_db_corrupted_20260527_190427",
    "./deep-rag/chroma_db_corrupted_215105",
    "./deep-rag/chroma_db_failed_200527",
    "./deep-rag/benchmark_chroma",
    "./chroma_db",
    "./chroma_db_extended",
]

total_size = 0
deleted_count = 0

for dir_path in CHROMADB_DIRS:
    path = Path(dir_path)
    if path.exists():
        # 计算大小
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        size_mb = size / (1024 * 1024)
        total_size += size

        print(f"🗑️  删除: {dir_path} ({size_mb:.1f} MB)")

        # 删除
        try:
            shutil.rmtree(path)
            deleted_count += 1
            print(f"   ✅ 已删除")
        except Exception as e:
            print(f"   ❌ 删除失败: {e}")
    else:
        print(f"⚠️  不存在: {dir_path}")

    print()

print("=" * 70)
print(f"✅ 清理完成")
print(f"   删除目录: {deleted_count} 个")
print(f"   释放空间: {total_size / (1024 * 1024):.1f} MB")
print("=" * 70)
