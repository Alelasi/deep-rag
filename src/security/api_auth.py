"""API Key 鉴权 — 生产开关与校验（常量时间比较防时序攻击）

设计：
- 未配置 API_KEY / DEEPRAG_API_KEY → 开发模式，一律放行（便于本地）
- 已配置 → 必须提供 X-API-Key 或 Authorization: Bearer <key>
"""
from __future__ import annotations

import os
import secrets
from typing import Optional


def get_configured_api_key() -> str:
    """读取环境中的 API Key（去空白）；空串表示未启用鉴权。"""
    # 兼容两套变量名，避免部署脚本命名不一致
    return (os.getenv("API_KEY") or os.getenv("DEEPRAG_API_KEY") or "").strip()


def is_auth_enabled() -> bool:
    """是否强制鉴权：有配置即启用。生产环境应始终返回 True。"""
    return bool(get_configured_api_key())


def verify_api_key(provided: Optional[str]) -> bool:
    """校验客户端提供的 Key 是否与配置一致。

    Args:
        provided: 原始头字段，可为裸 Key 或 ``Bearer xxx``

    Returns:
        True 表示允许访问；开发模式（无配置）恒为 True
    """
    expected = get_configured_api_key()
    # 开发模式：不强制 Key，方便 pytest / 本地 Streamlit
    if not expected:
        return True
    if not provided:
        return False

    # 统一成纯 token，再做常量时间比较
    token = provided.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    # secrets.compare_digest 要求同类型 str，且长度可不同时安全返回 False
    try:
        return secrets.compare_digest(token, expected)
    except (TypeError, ValueError):
        return False
