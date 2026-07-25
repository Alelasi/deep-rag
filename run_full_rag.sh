#!/bin/bash
# 本地文档 RAG 快速启动脚本

set -e

echo "🚀 本地文档 RAG 系统"
echo "=================="
echo ""

# 检查环境
echo "检查环境..."
conda run -n gpu_env python -c "import torch; import sentence_transformers; import faiss; import lancedb" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ 环境检查通过"
else
    echo "❌ 环境检查失败，请先安装依赖："
    echo "   conda run -n gpu_env pip install sentence-transformers lancedb faiss-cpu"
    exit 1
fi

echo ""
echo "选择操作："
echo "1. 索引文档（小规模测试 - docs 目录）"
echo "2. 索引文档（全量 - ai提问相关 目录）"
echo "3. 搜索查询"
echo ""
read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "📁 索引 docs 目录..."
        conda run -n gpu_env python src/local_document_rag.py index \
            --dir "D:\文档\ai提问相关\工作\docs" \
            --chunk-size 500 \
            --hot-capacity 1000
        ;;
    2)
        echo ""
        echo "📁 索引全量目录（预计 10-20 分钟）..."
        conda run -n gpu_env python src/local_document_rag.py index \
            --dir "D:\文档\ai提问相关" \
            --chunk-size 500 \
            --hot-capacity 10000
        ;;
    3)
        echo ""
        read -p "请输入查询: " query
        echo ""
        echo "🔍 搜索中..."
        conda run -n gpu_env python src/local_document_rag.py search "$query" --top-k 10
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 完成！"
