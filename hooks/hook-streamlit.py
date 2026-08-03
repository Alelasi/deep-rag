"""PyInstaller hook for Streamlit — 收集数据文件和隐藏导入"""
from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_submodules

# 收集 streamlit 的数据文件（HTML 模板、静态资源等）
datas = collect_data_files("streamlit")

# 收集元数据（版本信息）
datas += copy_metadata("streamlit")

# 收集所有子模块作为隐藏导入
hiddenimports = collect_submodules("streamlit")
