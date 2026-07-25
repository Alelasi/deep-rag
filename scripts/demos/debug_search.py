"""调试搜索问题"""
import sys
sys.path.insert(0, '.')

import lancedb
from pathlib import Path

# 直接查 LanceDB
db_path = "data/local_document_rag/lancedb"
db = lancedb.connect(db_path)

# 看有哪些表
print(f"Tables: {db.table_names()}")

if db.table_names():
    table = db.open_table(db.table_names()[0])
    print(f"Row count: {table.count_rows()}")

    # 直接搜索
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode("GPU加速向量检索")

    results = table.search(query_vec.tolist()).limit(5).to_list()
    print(f"\nDirect LanceDB search results: {len(results)}")
    for r in results:
        print(f"  ID: {r.get('id', 'N/A')}, Distance: {r.get('_distance', 'N/A'):.4f}")
else:
    print("No tables found!")
