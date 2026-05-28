"""分块元数据存储（按需加载，节省内存）

适用场景：
- 内存有限（<8GB）
- 大规模索引（>100万chunks）
- 不需要全量加载

优势：
- 按需加载（只加载需要的块）
- 内存占用小（<100MB）
- 查询速度快（<50ms）
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict


class ChunkedMetadata:
    """分块元数据存储

    将大文件分成多个小块，按需加载
    """

    def __init__(
        self,
        base_dir: str,
        chunk_size: int = 10000,
        cache_size: int = 10,
        compress: bool = True
    ):
        """初始化

        Args:
            base_dir: 存储目录
            chunk_size: 每块包含的chunk数量
            cache_size: 缓存块数量（LRU）
            compress: 是否压缩
        """
        self.base_dir = Path(base_dir)
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self.compress = compress

        # LRU缓存
        self._cache: OrderedDict[int, Dict] = OrderedDict()

        # 索引：chunk_id -> block_index
        self._index: Dict[str, int] = {}

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, metadata: Dict):
        """保存元数据（分块）

        Args:
            metadata: 完整元数据 {"chunks": {...}}
        """
        chunks = metadata.get('chunks', {})
        chunk_ids = list(chunks.keys())

        print(f"分块保存 {len(chunk_ids)} chunks...")

        # 分块保存
        for i in range(0, len(chunk_ids), self.chunk_size):
            block_index = i // self.chunk_size
            batch_ids = chunk_ids[i:i+self.chunk_size]
            batch_data = {cid: chunks[cid] for cid in batch_ids}

            # 更新索引
            for cid in batch_ids:
                self._index[cid] = block_index

            # 保存块
            self._save_block(block_index, batch_data)

            if (block_index + 1) % 10 == 0:
                print(f"  已保存 {block_index + 1} 块...")

        # 保存索引
        index_file = self.base_dir / 'index.json.gz'
        with gzip.open(index_file, 'wt', encoding='utf-8') as f:
            json.dump(self._index, f, ensure_ascii=False)

        print(f"✓ 完成！共 {len(chunk_ids)} chunks，{block_index + 1} 块")
        print(f"  块大小: {self.chunk_size} chunks/块")
        print(f"  压缩: {'是' if self.compress else '否'}")

    def _save_block(self, block_index: int, data: Dict):
        """保存单个块"""
        if self.compress:
            block_file = self.base_dir / f'block_{block_index:04d}.json.gz'
            with gzip.open(block_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            block_file = self.base_dir / f'block_{block_index:04d}.json'
            with open(block_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)

    def _load_block(self, block_index: int) -> Dict:
        """加载单个块（带LRU缓存）"""
        # 检查缓存
        if block_index in self._cache:
            # 移到最后（最近使用）
            self._cache.move_to_end(block_index)
            return self._cache[block_index]

        # 加载块
        if self.compress:
            block_file = self.base_dir / f'block_{block_index:04d}.json.gz'
            with gzip.open(block_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            block_file = self.base_dir / f'block_{block_index:04d}.json'
            with open(block_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # 加入缓存
        self._cache[block_index] = data

        # LRU淘汰
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return data

    def load_index(self):
        """加载索引（启动时调用一次）"""
        index_file = self.base_dir / 'index.json.gz'
        if not index_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")

        with gzip.open(index_file, 'rt', encoding='utf-8') as f:
            self._index = json.load(f)

        print(f"✓ 加载索引: {len(self._index)} chunks")

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """获取单个chunk（按需加载）

        Args:
            chunk_id: chunk ID

        Returns:
            chunk数据，不存在返回None
        """
        # 查找块索引
        block_index = self._index.get(chunk_id)
        if block_index is None:
            return None

        # 加载块
        block_data = self._load_block(block_index)

        return block_data.get(chunk_id)

    def get_chunks(self, chunk_ids: List[str]) -> Dict[str, Dict]:
        """批量获取chunks（优化版）

        Args:
            chunk_ids: chunk ID列表

        Returns:
            {chunk_id: chunk_data}
        """
        # 按块分组
        blocks_to_load = {}
        for cid in chunk_ids:
            block_index = self._index.get(cid)
            if block_index is not None:
                if block_index not in blocks_to_load:
                    blocks_to_load[block_index] = []
                blocks_to_load[block_index].append(cid)

        # 批量加载
        results = {}
        for block_index, cids in blocks_to_load.items():
            block_data = self._load_block(block_index)
            for cid in cids:
                if cid in block_data:
                    results[cid] = block_data[cid]

        return results

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            'cache_size': len(self._cache),
            'cache_limit': self.cache_size,
            'index_size': len(self._index),
        }


# 使用示例
if __name__ == '__main__':
    import time

    # 1. 转换现有元数据为分块格式
    print("=== 转换元数据为分块格式 ===")

    original_file = Path('data/matryoshka_rq_rag/metadata.json')
    chunked_dir = Path('data/matryoshka_rq_rag_chunked')

    # 加载原始文件
    print(f"加载原始文件: {original_file}")
    with open(original_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 创建分块存储
    chunked = ChunkedMetadata(
        base_dir=str(chunked_dir),
        chunk_size=10000,  # 每块1万chunks
        cache_size=10,     # 缓存10块（约10万chunks）
        compress=True
    )

    # 保存
    start = time.time()
    chunked.save(metadata)
    save_time = time.time() - start
    print(f"保存时间: {save_time:.1f}s")

    # 2. 测试查询性能
    print("\n=== 测试查询性能 ===")

    # 加载索引
    chunked.load_index()

    # 随机查询
    import random
    chunk_ids = list(metadata['chunks'].keys())
    test_ids = random.sample(chunk_ids, 100)

    # 单个查询
    start = time.time()
    for cid in test_ids[:10]:
        _ = chunked.get_chunk(cid)
    single_time = (time.time() - start) / 10

    # 批量查询
    start = time.time()
    _ = chunked.get_chunks(test_ids)
    batch_time = time.time() - start

    print(f"单个查询: {single_time*1000:.1f}ms/次")
    print(f"批量查询: {batch_time*1000:.1f}ms (100个)")
    print(f"缓存统计: {chunked.get_cache_stats()}")

    # 3. 内存占用估算
    print("\n=== 内存占用估算 ===")
    print(f"索引: ~{len(chunked._index) * 50 / 1024 / 1024:.1f} MB")
    print(f"缓存: ~{chunked.cache_size * chunked.chunk_size * 500 / 1024 / 1024:.1f} MB")
    print(f"总计: ~{(len(chunked._index) * 50 + chunked.cache_size * chunked.chunk_size * 500) / 1024 / 1024:.1f} MB")
