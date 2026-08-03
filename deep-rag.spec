# -*- mode: python ; coding: utf-8 -*-
"""DeepRAG PyInstaller Spec — onedir 模式打包"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

# === 收集数据文件 ===
datas = []
hiddenimports = []

# Streamlit
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")
hiddenimports += collect_submodules("streamlit")

# LangChain 相关
for pkg in ["langchain", "langchain_core", "langchain_community", "langchain_text_splitters",
            "langchain_openai", "langchain_ollama", "langchain_anthropic"]:
    try:
        datas += collect_data_files(pkg)
        datas += copy_metadata(pkg)
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# 其他关键包
for pkg in ["qdrant_client", "chromadb", "sentence_transformers",
            "tiktoken", "jinja2", "yaml", "pydantic", "pydantic_core"]:
    try:
        datas += collect_data_files(pkg)
        datas += copy_metadata(pkg)
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# === 隐藏导入 ===
hiddenimports += [
    "sklearn",
    "sklearn.metrics.pairwise",
    "scipy",
    "scipy.spatial",
    "numpy",
    "pandas",
    "rank_bm25",
    "jieba",
    "graphviz",
    "PIL",
    "psutil",
]

# === 收集应用自身代码 ===
datas += [("src", "src")]
datas += [("app.py", ".")]
datas += [("api.py", ".")]
datas += [(".streamlit", ".streamlit")]

a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "torch", "transformers", "tokenizers"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="deep-rag",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="deep-rag",
)
