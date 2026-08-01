"""DeepRAG — PyInstaller 打包入口

用法：
    1. 先安装 PyInstaller: pip install pyinstaller
    2. 打包: pyinstaller deep-rag.spec
    3. 运行: dist/deep-rag/deep-rag.exe
"""
import os
import sys

# 环境变量设置（必须在 import streamlit 之前）
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '2')

if getattr(sys, 'frozen', False):
    # PyInstaller 打包后的路径
    BASE_DIR = os.path.dirname(sys.executable)
    os.environ['HF_HOME'] = os.path.join(BASE_DIR, 'hf_cache')
    # 将 _internal 目录加入 path
    internal_dir = os.path.join(BASE_DIR, '_internal')
    if os.path.isdir(internal_dir):
        sys.path.insert(0, internal_dir)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

# 启动 Streamlit
from streamlit.web import cli as stcli

sys.argv = [
    "streamlit", "run", "app.py",
    "--global.developmentMode=false",
    "--server.port=8501",
    "--server.address=0.0.0.0",
    "--browser.gatherUsageStats=false",
]

sys.exit(stcli.main())
