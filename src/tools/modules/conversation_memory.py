"""
记忆管理模块（Memory）
支持多轮对话上下文管理
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime


class ConversationMemory:
    """
    对话记忆管理器

    核心能力：
    1. 记录对话历史（查询 + 结果）
    2. 获取最近K轮上下文
    3. 支持记忆搜索（找到相关历史）
    4. 自动摘要压缩（避免上下文过长）

    示例：
    第1轮: "查询用户总数" → "用户总数是5个"
    第2轮: "那年龄最大的是谁？" → 自动带上第1轮上下文
    """

    def __init__(self, max_history: int = 10, context_window: int = 3):
        """
        初始化记忆管理器

        Args:
            max_history: 最大保存历史条数
            context_window: 默认上下文窗口大小（最近K轮）
        """
        self.max_history = max_history
        self.context_window = context_window
        self.history: List[Dict[str, Any]] = []

    def add(
        self,
        query: str,
        result: Any,
        sql: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        添加一轮对话到记忆

        Args:
            query: 用户查询
            result: 查询结果
            sql: 执行的SQL（可选）
            metadata: 额外元数据（可选）
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result,
            "sql": sql,
            "metadata": metadata or {}
        }

        self.history.append(record)

        # 超过最大历史条数时，删除最旧的
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self, k: Optional[int] = None) -> str:
        """
        获取最近K轮对话上下文（格式化为文本）

        Args:
            k: 上下文窗口大小，默认使用初始化时的值

        Returns:
            格式化的上下文文本
        """
        k = k or self.context_window

        if not self.history:
            return ""

        recent = self.history[-k:]
        context_lines = []

        for i, record in enumerate(recent, 1):
            context_lines.append(f"第{i}轮对话:")
            context_lines.append(f"  用户: {record['query']}")

            # 格式化结果
            result = record['result']
            if isinstance(result, str):
                result_str = result
            elif isinstance(result, list):
                if len(result) == 0:
                    result_str = "无结果"
                elif len(result) <= 3:
                    result_str = json.dumps(result, ensure_ascii=False)
                else:
                    result_str = f"[{len(result)}条记录]"
            else:
                result_str = str(result)

            context_lines.append(f"  结果: {result_str}")

            if record.get('sql'):
                context_lines.append(f"  SQL: {record['sql']}")

            context_lines.append("")  # 空行分隔

        return "\n".join(context_lines)

    def get_last_result(self) -> Optional[Any]:
        """
        获取最近一轮的结果

        Returns:
            最近一轮的查询结果，如果没有历史则返回None
        """
        if not self.history:
            return None
        return self.history[-1]['result']

    def get_last_query(self) -> Optional[str]:
        """
        获取最近一轮的查询

        Returns:
            最近一轮的用户查询，如果没有历史则返回None
        """
        if not self.history:
            return None
        return self.history[-1]['query']

    def search(self, keyword: str, max_results: int = 5) -> List[Dict]:
        """
        搜索历史记录（关键词匹配）

        Args:
            keyword: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            匹配的历史记录列表
        """
        matches = []

        for record in reversed(self.history):  # 从最新开始搜索
            # 在query或sql中搜索关键词
            if (keyword.lower() in record['query'].lower() or
                (record.get('sql') and keyword.lower() in record['sql'].lower())):
                matches.append(record)

                if len(matches) >= max_results:
                    break

        return matches

    def clear(self):
        """清空所有记忆"""
        self.history.clear()

    def get_summary(self) -> str:
        """
        获取对话摘要

        Returns:
            对话摘要文本
        """
        if not self.history:
            return "无对话历史"

        total = len(self.history)
        recent = self.history[-3:] if len(self.history) >= 3 else self.history

        summary_lines = [
            f"对话摘要（共{total}轮）:",
            ""
        ]

        for i, record in enumerate(recent, 1):
            summary_lines.append(f"{i}. {record['query']}")

        return "\n".join(summary_lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        导出记忆为字典（用于持久化）

        Returns:
            包含所有历史的字典
        """
        return {
            "max_history": self.max_history,
            "context_window": self.context_window,
            "history": self.history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """
        从字典恢复记忆（用于持久化）

        Args:
            data: 导出的字典

        Returns:
            ConversationMemory实例
        """
        memory = cls(
            max_history=data.get("max_history", 10),
            context_window=data.get("context_window", 3)
        )
        memory.history = data.get("history", [])
        return memory


# ============================================================================
# 使用示例
# ============================================================================

def demo_conversation_memory():
    """演示对话记忆管理器"""
    memory = ConversationMemory(max_history=10, context_window=3)

    # 模拟多轮对话
    conversations = [
        ("查询所有用户", [{"id": 1, "name": "张三", "age": 25}]),
        ("年龄大于25的有几个？", [{"count": 3}]),
        ("刚才那个用户的邮箱是什么？", "zhang@example.com"),
        ("统计订单总数", [{"count": 100}]),
    ]

    print("=" * 60)
    print("模拟多轮对话")
    print("=" * 60)

    for query, result in conversations:
        memory.add(query, result)
        print(f"\n用户: {query}")
        print(f"结果: {result}")

    # 获取上下文
    print("\n" + "=" * 60)
    print("获取最近3轮对话上下文")
    print("=" * 60)
    print(memory.get_context(k=3))

    # 搜索历史
    print("\n" + "=" * 60)
    print("搜索包含'用户'的历史记录")
    print("=" * 60)
    matches = memory.search("用户")
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match['query']} → {match['result']}")

    # 获取摘要
    print("\n" + "=" * 60)
    print("对话摘要")
    print("=" * 60)
    print(memory.get_summary())


if __name__ == "__main__":
    demo_conversation_memory()
