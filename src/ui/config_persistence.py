"""API Key 持久化 — 保存到 .env 文件，重启自动加载"""
import os
from pathlib import Path
from dotenv import load_dotenv


def _get_env_path() -> Path:
    """获取 .env 文件路径"""
    # 向上查找 .env 文件
    current = Path(__file__).resolve().parent
    while current != current.parent:
        env_path = current / ".env"
        if env_path.exists():
            return env_path
        current = current.parent
    # 默认在项目根目录
    return Path(__file__).resolve().parent.parent.parent / ".env"


def load_env_dict() -> dict:
    """读取 .env 文件返回 dict"""
    env_path = _get_env_path()
    if not env_path.exists():
        return {}
    result = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
    return result


def save_to_env(updates: dict):
    """保存键值对到 .env 文件

    Args:
        updates: {KEY: VALUE} 字典
    """
    env_path = _get_env_path()
    # 读取现有内容
    existing = load_env_dict()
    # 合并更新
    existing.update(updates)
    # 写回文件
    lines = []
    for key, value in existing.items():
        lines.append(f"{key}={value}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    # 重新加载环境变量
    load_dotenv(env_path, override=True)


def get_env_value(key: str, default: str = "") -> str:
    """获取环境变量值"""
    return os.getenv(key, default)
