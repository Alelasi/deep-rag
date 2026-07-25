#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消息总线（Message Bus）- Agent 间通信机制

功能：
1. 事件发布/订阅（Pub/Sub）
2. 共享状态管理
3. 消息历史追踪
"""

from typing import Dict, List, Callable, Any
from datetime import datetime
import threading


class MessageBus:
    """Agent 间消息总线"""

    def __init__(self):
        # 订阅者字典：{event_name: [handler1, handler2, ...]}
        self._subscribers: Dict[str, List[Callable]] = {}

        # 共享状态字典：{key: value}
        self._shared_state: Dict[str, Any] = {}

        # 消息历史：[{event, data, timestamp}, ...]
        self._message_history: List[Dict] = []

        # 线程锁（确保线程安全）
        self._lock = threading.Lock()

    def publish(self, event: str, data: Dict) -> int:
        """
        发布事件

        Args:
            event: 事件名称（如 "query_analyzed", "docs_graded"）
            data: 事件数据

        Returns:
            通知的订阅者数量
        """
        with self._lock:
            # 记录消息历史
            self._message_history.append({
                "event": event,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })

            # 通知所有订阅者
            handlers = self._subscribers.get(event, [])
            for handler in handlers:
                try:
                    handler(data)
                except Exception as e:
                    print(f"Handler error for event '{event}': {e}")

            return len(handlers)

    def subscribe(self, event: str, handler: Callable):
        """
        订阅事件

        Args:
            event: 事件名称
            handler: 处理函数（接收 data 参数）
        """
        with self._lock:
            if event not in self._subscribers:
                self._subscribers[event] = []
            self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable):
        """取消订阅"""
        with self._lock:
            if event in self._subscribers:
                try:
                    self._subscribers[event].remove(handler)
                except ValueError:
                    pass

    def get_shared_state(self, key: str, default=None) -> Any:
        """获取共享状态"""
        with self._lock:
            return self._shared_state.get(key, default)

    def set_shared_state(self, key: str, value: Any):
        """设置共享状态"""
        with self._lock:
            self._shared_state[key] = value

            # 发布状态变更事件
            self.publish(f"state_changed:{key}", {"key": key, "value": value})

    def get_message_history(self, event: str = None) -> List[Dict]:
        """
        获取消息历史

        Args:
            event: 可选，只返回指定事件的历史

        Returns:
            消息历史列表
        """
        with self._lock:
            if event:
                return [msg for msg in self._message_history if msg["event"] == event]
            return self._message_history.copy()

    def get_metrics(self) -> Dict:
        """
        获取消息总线指标

        Returns:
            {
                "total_messages": 总消息数,
                "total_subscribers": 总订阅者数,
                "events": [事件列表],
                "message_by_event": {事件: 消息数}
            }
        """
        with self._lock:
            message_by_event = {}
            for msg in self._message_history:
                event = msg["event"]
                message_by_event[event] = message_by_event.get(event, 0) + 1

            return {
                "total_messages": len(self._message_history),
                "total_subscribers": sum(len(handlers) for handlers in self._subscribers.values()),
                "events": list(self._subscribers.keys()),
                "message_by_event": message_by_event
            }

    def clear(self):
        """清空所有数据（用于测试）"""
        with self._lock:
            self._subscribers.clear()
            self._shared_state.clear()
            self._message_history.clear()


# 全局单例
_global_bus = None

def get_message_bus() -> MessageBus:
    """获取全局消息总线单例"""
    global _global_bus
    if _global_bus is None:
        _global_bus = MessageBus()
    return _global_bus
