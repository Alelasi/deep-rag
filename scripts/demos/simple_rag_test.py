"""
简单的 RAG 检索测试
直接用 LanceDB + sentence-transformers
"""

import lancedb
from sentence_transformers import SentenceTransformer

print("="*60)
print("🔍 Deep-RAG 检索测试")
print("="*60)

# 1. 加载模型
print("\n📦 加载 Embedding 模型...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu')
print("✅ 模型加载成功")

# 2. 连接数据库
print("\n📊 连接向量数据库...")
db = lancedb.connect('lancedb_core')
table = db.open_table('core_docs')
count = table.count_rows()
print(f"✅ 数据库连接成功")
print(f"   文档数量：{count:,} 条")

# 3. 测试查询
test_queries = [
    "什么是 RAG？",
    "GPU 加速的效果如何？",
    "向量数据库有哪些？",
]

print("\n" + "="*60)
print("🔎 测试查询")
print("="*60)

for i, query in enumerate(test_queries, 1):
    print(f"\n【查询 {i}】{query}")
    print("-"*60)

    # 编码查询
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # 向量搜索
    results = table.search(query_embedding).limit(3).to_list()

    # 显示结果
    print(f"找到 {len(results)} 个相关文档：\n")

    for j, result in enumerate(results, 1):
        # LanceDB 返回的结果包含 _distance（越小越相似）
        distance = result.get('_distance', 1.0)
        similarity = 1 - distance  # 转换为相似度（越大越相似）

        content = result.get('text', '')
        source = result.get('source', '未知来源')

        print(f"{j}. 相似度：{similarity:.2%}")
        print(f"   来源：{source}")
        print(f"   内容：{content[:100]}...")
        print()

print("="*60)
print("✅ 测试完成")
print("="*60)

# 4. 统计信息
print(f"\n📊 数据库统计：")
print(f"   - 总文档数：{count:,} 条")
print(f"   - 数据库大小：384 MB")
print(f"   - 表名：core_docs")
