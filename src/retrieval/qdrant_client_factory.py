"""Qdrant 客户端工厂 — local 磁盘 / server / cloud 三模式

local：数据在 哲思灵智/qdrant_data（构建脚本默认；无需 Docker）
server：127.0.0.1:6333（Docker compose / 本地服务器）
cloud：Qdrant Cloud 托管服务（Railway 部署用）

注意：local 模式同一时刻只能有一个进程打开 path（构建时勿同时起检索）。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# 全局缓存，避免重复打开 local path
_client = None
_client_mode: Optional[str] = None


def get_qdrant_mode() -> str:
    """读取 QDRANT_MODE：server | local | cloud，默认 server（支持多客户端）。"""
    return os.getenv("QDRANT_MODE", "server").strip().lower()


def get_qdrant_path() -> Path:
    """本地模式数据目录。"""
    return Path(
        os.getenv(
            "QDRANT_PATH",
            r"D:\文档\ai提问相关\哲思灵智\qdrant_data",
        )
    )


def get_qdrant_client(force_new: bool = False):
    """获取共享 QdrantClient。

    force_new=True 时关闭缓存并重建（切换 mode 后使用）。
    """
    global _client, _client_mode
    from qdrant_client import QdrantClient

    mode = get_qdrant_mode()
    if _client is not None and _client_mode == mode and not force_new:
        return _client

    if mode == "server":
        host = os.getenv("QDRANT_HOST", "127.0.0.1")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY")
        kwargs = {"host": host, "port": port, "timeout": 30}
        if api_key:
            kwargs["api_key"] = api_key
            kwargs["https"] = True
        log.info("[Qdrant] server client %s:%s (api_key=%s)", host, port, bool(api_key))
        _client = QdrantClient(**kwargs)
    elif mode == "cloud":
        cloud_url = os.getenv("QDRANT_CLOUD_URL", "")
        cloud_key = os.getenv("QDRANT_CLOUD_KEY", "")
        if not cloud_url:
            raise ValueError("QDRANT_MODE=cloud 但未设置 QDRANT_CLOUD_URL")
        kwargs = {"url": cloud_url, "timeout": 60}
        if cloud_key:
            kwargs["api_key"] = cloud_key
        log.info("[Qdrant] cloud client %s (api_key=%s)", cloud_url, bool(cloud_key))
        _client = QdrantClient(**kwargs)
    else:
        path = get_qdrant_path()
        path.mkdir(parents=True, exist_ok=True)
        log.info("[Qdrant] local path client %s", path)
        _client = QdrantClient(path=str(path))
    _client_mode = mode
    return _client
