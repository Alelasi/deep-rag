"""测试按需加载方案的性能"""

import sys
sys.path.insert(0, 'src')

from retrieval.chunked_metadata import ChunkedMetadata
import json
import time
import random
from pathlib import Path

print('=' * 80)
print('按需加载方案性能测试')
print('=' * 80)

# 1. 转换现有元数据
print('\n=== 步骤1：转换为分块格式 ===')
original_file = Path('../data/matryoshka_rq_rag/metadata.json')
chunked_dir = Path('../data/matryoshka_rq_rag_chunked')

print(f'原始文件: {original_file}')
print(f'目标目录: {chunked_dir}')

# 加载原始文件
start = time.time()
with open(original_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
load_time = time.time() - start

print(f'✓ 加载完成: {load_time:.1f}s')
print(f'  Chunks数: {len(metadata["chunks"])}')

# 创建分块存储
chunked = ChunkedMetadata(
    base_dir=str(chunked_dir),
    chunk_size=10000,  # 每块1万chunks
    cache_size=10,     # 缓存10块
    compress=True
)

# 保存
start = time.time()
chunked.save(metadata)
save_time = time.time() - start
print(f'✓ 保存完成: {save_time:.1f}s')

# 2. 测试查询性能
print('\n=== 步骤2：查询性能测试 ===')

# 加载索引
chunked.load_index()

chunk_ids = list(metadata['chunks'].keys())
test_ids = random.sample(chunk_ids, min(100, len(chunk_ids)))

# 单个查询（10次）
times = []
for cid in test_ids[:10]:
    start = time.time()
    result = chunked.get_chunk(cid)
    times.append(time.time() - start)

avg_single = sum(times) / len(times)
print(f'单个查询: {avg_single*1000:.1f}ms/次')

# 批量查询（100个）
start = time.time()
results = chunked.get_chunks(test_ids)
batch_time = time.time() - start
print(f'批量查询: {batch_time*1000:.1f}ms (100个)')
print(f'  平均: {batch_time*1000/100:.2f}ms/个')

# 缓存统计
stats = chunked.get_cache_stats()
print(f'\n缓存统计:')
print(f'  缓存块数: {stats["cache_size"]}/{stats["cache_limit"]}')
print(f'  索引大小: {stats["index_size"]} chunks')

# 3. 内存占用估算
print('\n=== 步骤3：内存占用估算 ===')
index_mb = len(chunked._index) * 50 / 1024 / 1024
cache_mb = chunked.cache_size * chunked.chunk_size * 500 / 1024 / 1024
total_mb = index_mb + cache_mb

print(f'索引: ~{index_mb:.1f} MB')
print(f'缓存: ~{cache_mb:.1f} MB (10块 × 1万chunks)')
print(f'总计: ~{total_mb:.1f} MB')

# 4. 磁盘占用
print('\n=== 步骤4：磁盘占用 ===')
total_size = sum(f.stat().st_size for f in chunked_dir.glob('*'))
print(f'分块文件: {total_size / 1024 / 1024:.1f} MB')
print(f'原始文件: {original_file.stat().st_size / 1024 / 1024:.1f} MB')
print(f'节省: {(1 - total_size / original_file.stat().st_size) * 100:.0f}%')

# 5. 对比总结
print('\n' + '=' * 80)
print('方案对比')
print('=' * 80)
print(f'{"方案":<20} {"内存占用":<15} {"查询速度":<15} {"磁盘占用":<15}')
print('-' * 80)
print(f'{"全量加载":<20} {"~845 MB":<15} {"0.003ms":<15} {"845 MB":<15}')
print(f'{"gzip压缩":<20} {"~845 MB":<15} {"0.003ms":<15} {"208 MB":<15}')
print(f'{"按需加载(本方案)":<20} {f"~{total_mb:.0f} MB":<15} {f"{avg_single*1000:.1f}ms":<15} {f"{total_size/1024/1024:.0f} MB":<15}')

print('\n推荐：')
print('  ✓ 内存充足（>8GB）：gzip压缩 + 全量加载')
print('  ✓ 内存有限（<8GB）：按需加载（本方案）')
