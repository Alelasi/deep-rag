"""生产安全模块 — API 鉴权 / 限流 / 输入防护 / 审计

真实可运行实现（非文档占位）。默认：
- 未配置 API_KEY 时开发模式放行（打 warning）
- 配置 API_KEY 后强制校验
"""

from src.security.api_auth import (
    verify_api_key,
    is_auth_enabled,
    verify_jwt,
    is_jwt_enabled,
    get_configured_api_key,
)
from src.security.rate_limiter import RateLimiter, get_rate_limiter
from src.security.input_guard import sanitize_question, validate_index_path
from src.security.audit import audit_log

__all__ = [
    "verify_api_key",
    "is_auth_enabled",
    "verify_jwt",
    "is_jwt_enabled",
    "get_configured_api_key",
    "RateLimiter",
    "get_rate_limiter",
    "sanitize_question",
    "validate_index_path",
    "audit_log",
]
