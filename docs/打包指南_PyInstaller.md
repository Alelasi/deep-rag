# DeepRAG PyInstaller 打包指南

## 前置条件

1. Python 3.11+ 
2. 已安装项目依赖：`pip install -e ".[llm,api,ui,qdrant,reranker]"`
3. 安装打包工具：`pip install pyinstaller`
4. 预下载 embedding 模型（如需离线使用）：
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-zh-v1.5')"
   ```

## 打包步骤

### 1. 使用 spec 文件打包（推荐）

```bash
cd deep-rag
pyinstaller deep-rag.spec
```

### 2. 打包结果

打包完成后，`dist/deep-rag/` 目录包含：
- `deep-rag.exe` — 主程序入口
- `_internal/` — 依赖库和数据文件
- `src/` — 项目源码
- `.streamlit/` — Streamlit 配置
- `app.py` — Streamlit 应用

### 3. 运行

```bash
cd dist/deep-rag
deep-rag.exe
```

浏览器访问 http://localhost:8501

## 注意事项

1. **模型缓存**：打包不包含 embedding 模型。首次运行时会自动下载，或手动复制 `~/.cache/huggingface` 到打包目录的 `hf_cache/`
2. **向量数据库**：需要单独配置 Qdrant 服务器或使用 Qdrant Cloud
3. **文件大小**：onedir 模式约 300-500MB
4. **杀毒软件**：PyInstaller 打包的 exe 可能被误报，需添加排除项
5. **excludes 列表**：当前排除了 torch/transformers 等大包以减小体积，如需 Reranker 功能需移除排除

## 故障排查

### ImportError: No module named 'xxx'
在 `deep-rag.spec` 的 `hiddenimports` 列表中添加缺失的模块名。

### Streamlit 页面空白
检查 `_internal/streamlit/static/` 目录是否存在，如缺失重新打包。

### 模型加载失败
设置环境变量 `HF_HOME` 指向模型缓存目录，或预下载模型到打包目录。
