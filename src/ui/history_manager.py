"""Q&A 历史记录 — 保存到 JSON 文件，支持查看/搜索/清空"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


HISTORY_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "chat_history.json"


def _ensure_file():
    """确保历史记录文件存在"""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def save_qa(question: str, answer: str, mode: str = "enhanced",
            metrics: Optional[Dict] = None, citations: Optional[List] = None,
            agent_trace: Optional[List] = None):
    """保存一条问答记录

    Args:
        question: 用户问题
        answer: 系统回答
        mode: 检索模式（enhanced / agentic / naive）
        metrics: 评估指标
        citations: 引用来源
        agent_trace: Agent 决策轨迹
    """
    _ensure_file()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "mode": mode,
        "metrics": metrics or {},
        "citations": citations or [],
        "agent_trace": agent_trace or [],
    }
    history.append(record)
    # 限制最大记录数
    if len(history) > 1000:
        history = history[-1000:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_history(limit: int = 50) -> List[Dict]:
    """加载历史记录（时间倒序）

    Args:
        limit: 返回记录数

    Returns:
        历史记录列表
    """
    _ensure_file()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    return list(reversed(history))[:limit]


def search_history(keyword: str) -> List[Dict]:
    """搜索历史记录

    Args:
        keyword: 搜索关键词

    Returns:
        匹配的记录列表
    """
    _ensure_file()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    results = []
    for record in reversed(history):
        if keyword.lower() in record["question"].lower() or keyword.lower() in record["answer"].lower():
            results.append(record)
    return results


def clear_history():
    """清空所有历史记录"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


def get_history_stats() -> dict:
    """获取历史记录统计信息"""
    _ensure_file()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    total = len(history)
    modes = {}
    for record in history:
        mode = record.get("mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1
    avg_metrics = {}
    if total > 0:
        metric_keys = set()
        for record in history:
            metric_keys.update(record.get("metrics", {}).keys())
        for key in metric_keys:
            values = [r.get("metrics", {}).get(key, 0) for r in history if key in r.get("metrics", {})]
            if values:
                try:
                    avg_metrics[key] = sum(float(v) for v in values) / len(values)
                except (ValueError, TypeError):
                    avg_metrics[key] = 0
    return {"total": total, "modes": modes, "avg_metrics": avg_metrics}
