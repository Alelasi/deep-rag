"""
从核心 ChromaDB 筛选常用文档到 FAISS
根据文件类型和路径筛选最常用的文档
"""
import sys
sys.path.insert(0, '.')

import chromadb
import faiss
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

# 配置
CHROMA_DB_PATH = "./chroma_db"
FAISS_INDEX_PATH = "./faiss_index"
FAISS_METADATA_PATH = "./faiss_metadata.pkl"

# 常用文件类型（优先级从高到低）
PRIORITY_TYPES = {
    '.md': 10,      # Markdown 文档（最常用）
    '.py': 8,       # Python 代码
    '.txt': 6,      # 文本文件
    '.json': 5,     # 配置文件
    '.yaml': 5,
    '.yml': 5,
    '.sh': 4,       # 脚本
    '.bat': 4,
    '.js': 3,       # 前端代码
    '.ts': 3,
    '.html': 2,
    '.css': 2,
}

# 常用目录（优先级）
PRIORITY_DIRS = {
    'docs': 10,             # 文档目录
    '面试资料': 10,
    '简历': 9,
    'deep-rag': 8,          # 项目代码
    'job-agent': 8,
    'self-healing-pipeline': 8,
    '工作日志': 7,
    '任务文档': 6,
}

print("=" * 70)
print("🔍 从核心数据库筛选常用文档到 FAISS")
print("=" * 70)
print()

# 加载 ChromaDB
print("📦 加载核心数据库...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection("full_docs")
total_docs = collection.count()
print(f"✅ 核心数据库: {total_docs:,} 个文档块")
print()

# 获取所有文档
print("📊 分析文档分布...")
results = collection.get(
    include=['embeddings', 'metadatas', 'documents']
)

# 计算每个文档的优先级分数
doc_scores = []
for i, metadata in enumerate(results['metadatas']):
    source = metadata.get('source', '')
    doc_type = metadata.get('type', '')

    # 基础分数
    score = 0

    # 文件类型分数
    score += PRIORITY_TYPES.get(doc_type, 0)

    # 目录分数
    for dir_name, dir_score in PRIORITY_DIRS.items():
        if dir_name in source:
            score += dir_score
            break

    # 特殊加分
    if 'CLAUDE.md' in source or 'README.md' in source:
        score += 20
    if '工作日志' in source and '2026-05' in source:
        score += 15  # 最近的工作日志
    if 'AI_Agent' in source:
        score += 10

    doc_scores.append((i, score, source, doc_type))

# 排序
doc_scores.sort(key=lambda x: x[1], reverse=True)

# 统计
print("📈 文档类型分布:")
type_counts = defaultdict(int)
for _, _, _, doc_type in doc_scores:
    type_counts[doc_type] += 1

for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {doc_type:10s}: {count:>6,} 个")
print()

# 选择 Top 文档
print("🎯 筛选策略:")
print("  1. 优先选择 .md 文档（文档和笔记）")
print("  2. 优先选择项目代码和面试资料")
print("  3. 优先选择最近的工作日志")
print()

# 动态选择数量（目标：10-30K 文档，约 0.05-0.15 GB）
target_docs = min(30000, total_docs // 5)
selected_indices = [idx for idx, score, _, _ in doc_scores[:target_docs] if score > 0]

print(f"✅ 筛选出 {len(selected_indices):,} 个高优先级文档（占比 {len(selected_indices)/total_docs*100:.1f}%）")
print()

# 创建 FAISS 索引
print("📦 创建 FAISS 索引...")
embeddings = np.array([results['embeddings'][i] for i in selected_indices], dtype='float32')
embedding_dim = embeddings.shape[1]

index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

print(f"✅ FAISS 索引已创建: {index.ntotal:,} 个向量")
print(f"📦 索引大小: {index.ntotal * embedding_dim * 4 / 1024 / 1024:.1f} MB")
print()

# 保存元数据
print("💾 保存元数据...")
metadata_list = []
for i in selected_indices:
    metadata_list.append({
        'id': results['ids'][i],
        'document': results['documents'][i],
        'metadata': results['metadatas'][i]
    })

Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, FAISS_INDEX_PATH)

with open(FAISS_METADATA_PATH, 'wb') as f:
    pickle.dump(metadata_list, f)

print(f"✅ 索引已保存: {FAISS_INDEX_PATH}")
print(f"✅ 元数据已保存: {FAISS_METADATA_PATH}")
print()

# 统计筛选结果
print("=" * 70)
print("📊 筛选结果统计")
print("=" * 70)

selected_types = defaultdict(int)
selected_dirs = defaultdict(int)

for i in selected_indices:
    metadata = results['metadatas'][i]
    source = metadata.get('source', '')
    doc_type = metadata.get('type', '')

    selected_types[doc_type] += 1

    # 提取顶级目录
    parts = source.split('/')
    if len(parts) > 0:
        selected_dirs[parts[0]] += 1

print("文件类型分布:")
for doc_type, count in sorted(selected_types.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {doc_type:10s}: {count:>6,} 个 ({count/len(selected_indices)*100:.1f}%)")

print()
print("目录分布:")
for dir_name, count in sorted(selected_dirs.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {dir_name:30s}: {count:>6,} 个 ({count/len(selected_indices)*100:.1f}%)")

print()
print("=" * 70)
print("✅ 完成！")
print("=" * 70)
print(f"核心数据库: {total_docs:,} 个文档块 (1.01 GB)")
print(f"FAISS 索引: {len(selected_indices):,} 个文档块 (~{index.ntotal * embedding_dim * 4 / 1024 / 1024:.0f} MB)")
print(f"压缩比: {total_docs / len(selected_indices):.1f}x")
print()
print("💡 使用方法:")
print("  1. 优先查询 FAISS 索引（快速、低内存）")
print("  2. 如果结果不满意，再查询完整 ChromaDB")
print("=" * 70)
