"""API Key / JWT 鉴权 — 生产开关与校验（常量时间比较防时序攻击）

设计：
- 未配置 API Key（DEEP_RAG_API_KEY）→ 开发模式，一律放行（便于本地），但启动告警
- 已配置 → 必须提供 X-API-Key 或 Authorization: Bearer <key>
- 可选 JWT（HS256，纯标准库实现）：设置 DEEP_RAG_JWT_ENABLED=1 且
  DEEP_RAG_JWT_SECRET 后，Bearer <jwt> 也可接受
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 鉴权相关环境变量名
ENV_API_KEY = "DEEP_RAG_API_KEY"
# 兼容旧变量名（保留以避免破坏既有部署脚本）
_ENV_API_KEY_LEGACY = ("API_KEY", "DEEPRAG_API_KEY")
ENV_JWT_ENABLED = "DEEP_RAG_JWT_ENABLED"
ENV_JWT_SECRET = "DEEP_RAG_JWT_SECRET"


def get_configured_api_key() -> str:
    """读取环境中的 API Key（去空白）；空串表示未启用鉴权。

    优先 DEEP_RAG_API_KEY，回退 API_KEY / DEEPRAG_API_KEY（兼容旧脚本）。
    """
    key = (os.getenv(ENV_API_KEY) or "").strip()
    if key:
        return key
    for legacy in _ENV_API_KEY_LEGACY:
        key = (os.getenv(legacy) or "").strip()
        if key:
            logger.warning(
                "使用旧环境变量 %s 配置 API Key；建议迁移到 %s。",
                legacy,
                ENV_API_KEY,
            )
            return key
    return ""


def is_auth_enabled() -> bool:
    """是否强制鉴权：有配置即启用。生产环境应始终返回 True。"""
    return bool(get_configured_api_key())


def _normalize_token(provided: Optional[str]) -> Optional[str]:
    """统一成纯 token：剥离 'Bearer ' 前缀与空白；空输入返回 None。"""
    if not provided:
        return None
    token = provided.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def verify_api_key(provided: Optional[str]) -> bool:
    """校验客户端提供的 Key 是否与配置一致。

    Args:
        provided: 原始头字段，可为裸 Key 或 ``Bearer xxx``

    Returns:
        True 表示允许访问；开发模式（无配置）恒为 True
    """
    token = _normalize_token(provided)
    expected = get_configured_api_key()
    # 开发模式：不强制 Key，方便 pytest / 本地 Streamlit
    if not expected:
        return True
    if not token:
        return False

    # secrets.compare_digest 要求同类型 str，且长度可不同时安全返回 False
    try:
        ok = secrets.compare_digest(token, expected)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        logger.warning("API Key 校验失败（疑似非法访问尝试）")
    return ok


def is_jwt_enabled() -> bool:
    """可选 JWT 鉴权是否开启：需同时设置启用开关与密钥。"""
    enabled = os.getenv(ENV_JWT_ENABLED, "").strip().lower() in ("1", "true", "yes", "on")
    secret = (os.getenv(ENV_JWT_SECRET) or "").strip()
    if enabled and not secret:
        logger.warning(
            "%s=1 但缺少 %s，JWT 鉴权不可用。", ENV_JWT_ENABLED, ENV_JWT_SECRET
        )
        return False
    return enabled


def _b64url_decode(data: str) -> bytes:
    """解码 base64url（补齐 '=' 使长度为 4 的倍数）。"""
    rem = len(data) % 4
    if rem:
        data += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data)


def verify_jwt(provided: Optional[str]) -> bool:
    """校验 HS256 JWT（仅依赖标准库，无第三方依赖）。

    前置条件：``is_jwt_enabled()`` 为 True 且 ``DEEP_RAG_JWT_SECRET`` 已配置。
    校验签名与 ``exp`` 时效；任何失败都会记录告警并返回 False。

    Args:
        provided: 原始头字段，可为 ``Bearer <jwt>``

    Returns:
        True 表示 JWT 有效
    """
    if not is_jwt_enabled():
        return False
    token = _normalize_token(provided)
    if not token:
        return False
    secret = (os.getenv(ENV_JWT_SECRET) or "").strip()
    try:
        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("JWT 格式非法（段数 != 3）")
            return False
        signing_input = (parts[0] + "." + parts[1]).encode("ascii")
        sig = _b64url_decode(parts[2])
        expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            logger.warning("JWT 签名校验失败")
            return False
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        exp = payload.get("exp")
        if exp is not None and time.time() > float(exp):
            logger.warning("JWT 已过期")
            return False
        return True
    except (ValueError, binascii.Error, UnicodeDecodeError, KeyError) as e:
        logger.warning("JWT 解析失败: %s", e)
        return False
