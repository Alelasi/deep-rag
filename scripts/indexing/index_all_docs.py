"""索引 D:\文档 下的所有文档"""
import os
import sys
from pathlib import Path

# 禁用 OpenMP 线程亲和性
os.environ['KMP_AFFINITY'] = 'disabled'
os.environ['OMP_NUM_THREADS'] = '4'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "deep-rag"))

from src.local_document_rag import LocalDocumentRAG

def main():
    print("=" * 60)
    print("开始索引 D:\文档 下的所有文档")
    print("=" * 60)
    
    # 创建 RAG 实例
    rag = LocalDocumentRAG(
        db_path="data/full_docs_rag",
        hot_capacity=30000,  # 热数据容量
        use_gpu=True,
    )
    
    # 扫描文件
    print("\n📂 扫描文件...")
    files = rag.scan_files(
        root_dir="D:/文档",
        extensions=[".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".html", ".js", ".ts", ".jsx", ".tsx", ".css", ".xml"]
    )
    print(f"✅ 找到 {len(files)} 个文件")
    
    # 索引文件
    print("\n🔨 开始索引...")
    rag.index_files(files, batch_size=100)
    
    print("\n✅ 索引完成！")
    print(f"📊 统计信息：")
    print(f"  - 文件数：{len(rag.metadata['files'])}")
    print(f"  - 文本块数：{len(rag.metadata['chunks'])}")
    print(f"  - 热数据：{rag.store.hot_index.ntotal if rag.store.hot_index else 0}")
    print(f"  - 冷数据：{rag.store.cold_db.count_rows('vectors') if rag.store.cold_db else 0}")

if __name__ == "__main__":
    main()
