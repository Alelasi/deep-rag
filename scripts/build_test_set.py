"""从 xlsx 构建测试集 — 提取非闲聊问题，80/20划分训练/验证集"""
import json
import random
from pathlib import Path
import pandas as pd


def build_test_set(xlsx_path: str, output_path: str = "data/test_set.json",
                   sample_size: int = 1000, train_ratio: float = 0.8):
    """从 xlsx 提取测试集

    Args:
        xlsx_path: xlsx 文件路径
        output_path: 输出 JSON 路径
        sample_size: 采样数量
        train_ratio: 训练集比例

    Returns:
        {train_count, val_count, total}
    """
    df = pd.read_excel(xlsx_path)

    # 筛选非闲聊问题
    intent_col = None
    for col in df.columns:
        if "意图" in str(col) or "intent" in str(col).lower():
            intent_col = col
            break

    if intent_col:
        # 排除闲聊类
        non_chitchat = df[~df[intent_col].astype(str).str.contains("闲聊|chitchat|greeting", case=False, na=False)]
    else:
        non_chitchat = df

    # 筛选有有效问题和回答的
    query_col = None
    response_col = None
    for col in df.columns:
        if "query" in str(col).lower() or "问题" in str(col):
            query_col = col
        if "response" in str(col).lower() or "回复" in str(col) or "answer" in str(col).lower():
            response_col = col

    if not query_col:
        query_col = df.columns[0]
    if not response_col:
        response_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    non_chitchat = non_chitchat.dropna(subset=[query_col])
    non_chitchat = non_chitchat[non_chitchat[query_col].astype(str).str.len() > 5]

    # 随机采样
    if len(non_chitchat) > sample_size:
        sampled = non_chitchat.sample(n=sample_size, random_state=42)
    else:
        sampled = non_chitchat

    # 划分训练/验证集
    train_count = int(len(sampled) * train_ratio)
    train_df = sampled.iloc[:train_count]
    val_df = sampled.iloc[train_count:]

    test_set = {
        "train": [
            {
                "question": str(row[query_col]),
                "expected_answer": str(row[response_col]) if response_col else "",
                "intent": str(row.get(intent_col, "")) if intent_col else "",
            }
            for _, row in train_df.iterrows()
        ],
        "validation": [
            {
                "question": str(row[query_col]),
                "expected_answer": str(row[response_col]) if response_col else "",
                "intent": str(row.get(intent_col, "")) if intent_col else "",
            }
            for _, row in val_df.iterrows()
        ],
    }

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    return {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "total": len(sampled),
    }


def import_xlsx_to_chromadb(xlsx_path: str, collection_name: str = "knowledge_base"):
    """将 xlsx 问答对导入 ChromaDB 作为知识源

    Args:
        xlsx_path: xlsx 文件路径
        collection_name: ChromaDB collection 名称

    Returns:
        {indexed_count, total}
    """
    import chromadb
    from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, DEVICE
    from src.ui.model_cache import get_embedding_model

    df = pd.read_excel(xlsx_path)

    # 找到问题和回答列
    query_col = None
    response_col = None
    for col in df.columns:
        if "query" in str(col).lower() or "问题" in str(col):
            query_col = col
        if "response" in str(col).lower() or "回复" in str(col) or "answer" in str(col).lower():
            response_col = col

    if not query_col:
        return {"error": "找不到问题列"}

    # 筛选有效问答
    valid = df.dropna(subset=[query_col])
    valid = valid[valid[query_col].astype(str).str.len() > 5]

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_or_create_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    embedder = get_embedding_model(EMBEDDING_MODEL, DEVICE)

    indexed = 0
    batch_size = 100
    for i in range(0, len(valid), batch_size):
        batch = valid.iloc[i:i + batch_size]
        texts = []
        metadatas = []
        ids = []
        for j, (_, row) in enumerate(batch.iterrows()):
            q = str(row[query_col])
            a = str(row[response_col]) if response_col and pd.notna(row.get(response_col)) else ""
            text = f"问题：{q}\n回答：{a}" if a else q
            texts.append(text)
            metadatas.append({"source": "xlsx", "row": i + j})
            ids.append(f"xlsx_{i + j}")

        if texts:
            embeddings = embedder.encode(texts).tolist()
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            indexed += len(texts)

    return {"indexed_count": indexed, "total": len(valid)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        xlsx = sys.argv[1]
    else:
        xlsx = input("请输入 xlsx 文件路径: ")

    result = build_test_set(xlsx)
    print(f"测试集构建完成: 训练集 {result['train_count']} 条, 验证集 {result['val_count']} 条, 总计 {result['total']} 条")

    imp = input("是否导入到 ChromaDB? (y/n): ")
    if imp.lower() == 'y':
        result = import_xlsx_to_chromadb(xlsx)
        print(f"导入完成: {result}")
