"""审计日志 — 追加写 JSONL，供合规与故障追溯

每条记录包含 UTC 时间、事件名、request_id、client、detail。
路径默认 ``logs/audit.jsonl``，可用 AUDIT_LOG_PATH 覆盖。

典型 event：
- auth_denied / rate_limited
- query_ok / query_error / index_ok / index_denied
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# 写文件互斥，避免多线程交错行导致 JSONL 破损
_lock = threading.Lock()


def _log_path() -> Path:
    """解析审计文件路径并确保父目录存在。

    优先环境变量 AUDIT_LOG_PATH；否则落在项目 logs/ 下。
    """
    # parents[2] = deep-rag 项目根（src/security/audit.py → 上两级）
    root = Path(__file__).resolve().parents[2]
    default = root / "logs" / "audit.jsonl"
    p = Path(os.getenv("AUDIT_LOG_PATH", str(default)))
    # 目录不存在时自动创建，避免首次写入失败
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def audit_log(
    event: str,
    *,
    request_id: str = "",
    client: str = "",
    detail: Optional[Dict[str, Any]] = None,
    level: str = "info",
) -> None:
    """追加一条审计事件到 JSONL。

    Args:
        event: 事件名（短横线/下划线风格）
        request_id: 请求关联 ID，便于串联 API 日志
        client: 客户端标识（通常 IP）
        detail: 附加结构化字段（勿写密钥明文）
        level: info / warning / error

    Note:
        磁盘 IO 失败会向上抛 OSError；API 层可选择捕获以免拖垮主流程。
    """
    # 统一 UTC，避免多时区部署对账困难
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        "request_id": request_id,
        "client": client,
        "detail": detail or {},
    }
    # ensure_ascii=False：中文 detail 可读
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        # 追加模式；每条一行，便于 tail -f 与日志采集
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(line + "\n")
