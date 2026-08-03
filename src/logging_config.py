"""统一日志配置。

仅依赖 Python 标准库（logging），零第三方依赖。
可被任意模块安全 import，不会产生循环依赖，也不会重复添加 handler。

用法：
    from src.logging_config import get_logger, configure_logging

    # 在程序入口调用一次即可（幂等，可重复调用）
    configure_logging(logging.INFO)

    logger = get_logger(__name__)
    logger.info("hello")
"""

from __future__ import annotations

import logging

__all__ = ["get_logger", "configure_logging"]

# 标记到 handler 上的私有属性，用于幂等判断（避免重复添加 console handler）
_CONSOLE_MARK = "_deep_rag_console"

# 模块级标记，记录是否已经配置过（双重保护）
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """返回一个以 ``name`` 命名的 logger。

    该函数无副作用、无导入依赖，可以被任意模块安全导入而不触发
    循环依赖或副作用。
    """
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """幂等地配置根 logger 的 console 输出。

    多次调用只会添加一次 console handler，并刷新日志级别。
    通过模块级标记 + handler 上的私有标记双重保护，确保不重复加 handler。
    """
    global _CONFIGURED

    root = logging.getLogger()

    # 若已存在我们注入的 console handler，则视为已配置，仅更新级别后返回
    existing = [h for h in root.handlers if getattr(h, _CONSOLE_MARK, False)]
    if not existing:
        try:
            handler = logging.StreamHandler()
            # 打上标记，便于幂等识别
            setattr(handler, _CONSOLE_MARK, True)
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            root.addHandler(handler)
        except Exception:
            # 极端情况下（如 stdout 不可用）不应影响业务模块导入
            pass

    # 设置/刷新日志级别
    try:
        root.setLevel(level)
    except Exception:
        pass

    _CONFIGURED = True
