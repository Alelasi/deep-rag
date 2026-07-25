import sys
sys.path.insert(0, '.')
from src.local_document_rag import LocalDocumentRAG

rag = LocalDocumentRAG()
print(f'Metadata chunks: {len(rag.metadata["chunks"])}')

results = rag.search('GPU加速向量检索', top_k=5)
print(f'Results: {len(results)}')

for i, r in enumerate(results, 1):
    print(f'{i}. Score: {r["score"]:.4f} | File: {r["metadata"]["file_name"]}')
    print(f'   Text: {r["text"][:80]}...')
