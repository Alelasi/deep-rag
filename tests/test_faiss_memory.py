"""
FAISS 大规模索引内存占用测试
模拟索引 150K 文档的内存使用情况
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import psutil
import gc

EMBEDDING_DIM = 512
TARGET_DOCS = 153378  # 核心数据库的文档数
BATCH_SIZE = 256

print("=" * 70)
print("🧪 FAISS 大规模索引内存占用测试")
print("=" * 70)
print(f"目标文档数: {TARGET_DOCS:,}")
print(f"批次大小: {BATCH_SIZE}")
print()

# 初始内存
process = psutil.Process()
initial_mem = process.memory_info().rss / 1024 / 1024 / 1024
print(f"初始内存: {initial_mem:.2f} GB")
print()

# 加载模型
print("📦 加载GPU模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")
after_model_mem = process.memory_info().rss / 1024 / 1024 / 1024
print(f"✅ 模型已加载")
print(f"模型内存: {after_model_mem - initial_mem:.2f} GB")
print()

# 创建 FAISS 索引
print("📦 创建 FAISS 索引...")
import faiss

index = faiss.IndexFlatIP(EMBEDDING_DIM)
after_index_mem = process.memory_info().rss / 1024 / 1024 / 1024
print(f"✅ 索引已创建")
print(f"索引内存: {after_index_mem - after_model_mem:.2f} GB")
print()

# 模拟批量写入
print("🔥 模拟批量写入...")
print()

num_batches = (TARGET_DOCS + BATCH_SIZE - 1) // BATCH_SIZE
mem_samples = []

for i in range(num_batches):
    # 生成测试向量
    batch_vectors = np.random.rand(BATCH_SIZE, EMBEDDING_DIM).astype('float32')

    # 写入索引
    index.add(batch_vectors)

    # 清理
    del batch_vectors

    # 每10个批次采样一次内存
    if i % 10 == 0:
        current_mem = process.memory_info().rss / 1024 / 1024 / 1024
        mem_samples.append((i * BATCH_SIZE, current_mem))

        vram_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        index_size_gb = index.ntotal * EMBEDDING_DIM * 4 / 1024 / 1024 / 1024

        print(f"[{i * BATCH_SIZE:>6,} 文档] 内存: {current_mem:.2f} GB | 显存: {vram_gb:.2f} GB | 索引: {index_size_gb:.2f} GB")

    # 定期清理
    if i % 50 == 0:
        gc.collect()

# 最终统计
final_mem = process.memory_info().rss / 1024 / 1024 / 1024
final_vram = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
index_size_gb = index.ntotal * EMBEDDING_DIM * 4 / 1024 / 1024 / 1024

print()
print("=" * 70)
print("📊 内存占用总结")
print("=" * 70)
print(f"初始内存:     {initial_mem:.2f} GB")
print(f"模型加载后:   {after_model_mem:.2f} GB (+{after_model_mem - initial_mem:.2f} GB)")
print(f"最终内存:     {final_mem:.2f} GB (+{final_mem - initial_mem:.2f} GB)")
print(f"显存占用:     {final_vram:.2f} GB")
print()
print(f"FAISS 索引大小: {index_size_gb:.2f} GB ({index.ntotal:,} 文档)")
print(f"理论大小:       {TARGET_DOCS * EMBEDDING_DIM * 4 / 1024 / 1024 / 1024:.2f} GB")
print()

# 内存增长分析
if len(mem_samples) > 1:
    mem_growth = mem_samples[-1][1] - mem_samples[0][1]
    docs_added = mem_samples[-1][0] - mem_samples[0][0]
    mem_per_10k = (mem_growth / docs_added) * 10000 if docs_added > 0 else 0

    print(f"内存增长:       {mem_growth:.2f} GB")
    print(f"每1万文档:      {mem_per_10k:.3f} GB")
    print(f"预估150K文档:   {mem_per_10k * 15:.2f} GB")

print("=" * 70)

# 测试检索性能
print()
print("🔍 测试检索性能...")
query_vector = np.random.rand(1, EMBEDDING_DIM).astype('float32')

start = time.time()
for _ in range(100):
    distances, indices = index.search(query_vector, k=5)
elapsed = time.time() - start

print(f"✅ 100次检索耗时: {elapsed*1000:.0f}ms")
print(f"📊 平均检索时间: {elapsed*10:.1f}ms")
print(f"📊 QPS: {100/elapsed:.0f}")
print()

print("=" * 70)
print("💡 结论")
print("=" * 70)
print(f"✅ FAISS 索引 {TARGET_DOCS:,} 文档需要约 {index_size_gb:.2f} GB 内存")
print(f"✅ 写入速度极快（几乎瞬间）")
print(f"✅ 检索速度: ~{elapsed*10:.1f}ms/查询")
print()
print("⚠️ 注意: FAISS 是纯内存索引，需要常驻内存")
print("⚠️ 重启后需要重新加载索引")
print("=" * 70)
