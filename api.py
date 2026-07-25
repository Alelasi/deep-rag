"""根入口：uvicorn api:app --host 0.0.0.0 --port 8000

转发至 scripts.api，避免文档与测试 import 路径分裂。
"""
from scripts.api import app

__all__ = ["app"]
