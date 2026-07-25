"""
从 LanceDB 核心数据库筛选常用文档到 FAISS 热缓存
根据文件类型和路径筛选最常用的文档
"""
import sys
sys.path.insert(0, '.')

import lancedb
import faiss
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

# 配置
LANCE_DB_PATH = "./lancedb_core"
FAISS_INDEX_PATH = "./faiss_hotcache.index"
FAISS_METADATA_PATH = "./faiss_hotcache.pkl"
TARGET_DOCS = 30000  # 目标：3万个常用文档

# 常用文件类型（优先级从高到低）
PRIORITY_TYPES = {
    '.md': 10,      # Markdown 文档（最常用）
    '.py': 8,       # Python 代码
    '.txt': 6,      # 文本文件
    '.json': 5,     # 配置文件
    '.yaml': 5,
    '.yml': 5,
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
}

print("=" * 70)
print("🔥 创建 FAISS 热缓存")
print("=" * 70)
print(f"源数据库: {LANCE_DB_PATH}")
print(f"目标文档数: {TARGET_DOCS:,}")
print()

# 加载 LanceDB
print("📦 加载核心数据库...")
db = lancedb.connect(LANCE_DB_PATH)
table = db.open_table("core_docs")
total_docs = len(table)
print(f"✅ 核心数据库: {total_docs:,} 个文档块")
print()

# 获取所有文档
print("📊 分析文档分布...")
df = table.to_pandas()

# 计算每个文档的优先级分数
doc_scores = []
for idx, row in df.iterrows():
    source = row.get('source', '')
    doc_type = row.get('type', '')

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

    doc_scores.append((idx, score, source, doc_type))

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

# 动态选择数量
target_docs = min(TARGET_DOCS, total_docs // 5)
selected_indices = [idx for idx, score, _, _ in doc_scores[:target_docs] if score > 0]

print(f"✅ 筛选出 {len(selected_indices):,} 个高优先级文档（占比 {len(selected_indices)/total_docs*100:.1f}%）")
print()

# 提取向量和元数据
print("📦 提取向量和元数据...")
selected_rows = df.iloc[selected_indices]
embeddings = np.array([row['vector'] for _, row in selected_rows.iterrows()], dtype='float32')
embedding_dim = embeddings.shape[1]

print(f"✅ 提取完成: {len(embeddings):,} 个向量（维度: {embedding_dim}）")
print()

# 创建 FAISS 索引
print("📦 创建 FAISS 索引...")
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

print(f"✅ FAISS 索引已创建: {index.ntotal:,} 个向量")
print(f"📦 索引大小: {index.ntotal * embedding_dim * 4 / 1024 / 1024:.1f} MB")
print()

# 保存元数据
print("💾 保存索引和元数据...")
metadata_list = []
for _, row in selected_rows.iterrows():
    metadata_list.append({
        'id': row['id'],
        'text': row['text'],
        'source': row['source'],
        'page': row['page'],
        'type': row['type'],
    })

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

for _, row in selected_rows.iterrows():
    source = row['source']
    doc_type = row['type']

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
print("✅ FAISS 热缓存创建完成！")
print("=" * 70)
print(f"核心数据库: {total_docs:,} 个文档块")
print(f"FAISS 热缓存: {len(selected_indices):,} 个文档块 (~{index.ntotal * embedding_dim * 4 / 1024 / 1024:.0f} MB)")
print(f"压缩比: {total_docs / len(selected_indices):.1f}x")
print()
print("💡 使用方法:")
print("  1. 优先查询 FAISS 热缓存（极速）")
print("  2. 如果结果不满意，再查询完整 LanceDB")
print()
print("📊 预期性能:")
print(f"  FAISS 检索速度: ~1ms/查询（比 LanceDB 快 28x）")
print(f"  内存占用: ~{index.ntotal * embedding_dim * 4 / 1024 / 1024:.0f} MB")
print("=" * 70)
