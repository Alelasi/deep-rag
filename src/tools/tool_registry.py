"""
工具注册中心 - 白名单机制（v2.9: 支持 OpenAI Function Calling 格式）
只允许安全的只读操作，禁止增删改

v2.9 改进：
  - get_tool_schemas() 返回标准 OpenAI Function Calling 格式
  - 新增 to_openai_function() 方法
  - execute() 支持 collection_name 上下文传递
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

log = logging.getLogger("deeprag")


class ToolCategory(Enum):
    """工具分类"""
    SEARCH = "搜索查询"
    RETRIEVAL = "知识检索"
    CALCULATION = "计算分析"
    READ_ONLY = "只读操作"
    GENERATION = "答案生成"


class ToolRisk(Enum):
    """风险等级"""
    SAFE = "安全"  # 只读操作
    DANGEROUS = "危险"  # 增删改操作


@dataclass
class Tool:
    """工具定义（v2.9: 增加 openai_schema 字段）"""
    name: str
    description: str
    category: ToolCategory
    risk: ToolRisk
    function: Callable
    parameters: Dict[str, Any]
    examples: List[str]
    openai_schema: Optional[Dict] = None  # v2.9: 可选的自定义 OpenAI schema

    def to_openai_function(self) -> Dict:
        """转换为 OpenAI Function Calling 格式（v2.9新增）

        返回格式:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        if self.openai_schema:
            return self.openai_schema
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolRegistry:
    """
    工具注册中心

    白名单机制：
    1. 只允许注册SAFE级别的工具
    2. 禁止任何写入/修改/删除操作
    3. 所有工具必须显式声明风险等级
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._blacklist_keywords = [
            # SQL危险关键词
            "insert", "update", "delete", "drop", "truncate",
            "alter", "create", "replace", "merge",
            # 文件操作危险关键词
            "write", "remove", "unlink", "rmdir", "chmod",
            # 系统操作危险关键词
            "exec", "eval", "system", "popen", "subprocess"
        ]

    def register(self, tool: Tool) -> bool:
        """
        注册工具（白名单机制）

        规则：
        1. 只允许SAFE级别工具
        2. 工具名不能包含危险关键词
        3. 描述必须明确标注"只读"
        """
        # 规则1：只允许SAFE级别
        if tool.risk != ToolRisk.SAFE:
            raise ValueError(f"❌ 工具 [{tool.name}] 风险等级为 {tool.risk.value}，禁止注册")

        # 规则2：检查危险关键词
        tool_name_lower = tool.name.lower()
        for keyword in self._blacklist_keywords:
            if keyword in tool_name_lower:
                raise ValueError(f"❌ 工具名 [{tool.name}] 包含危险关键词 '{keyword}'，禁止注册")

        # 规则3：描述必须包含"只读"或"查询"
        desc_lower = tool.description.lower()
        safe_keywords = ["只读", "查询", "搜索", "检索", "read", "query", "search", "get", "list"]
        if not any(kw in desc_lower for kw in safe_keywords):
            raise ValueError(f"❌ 工具 [{tool.name}] 描述未明确标注为只读操作")

        # 注册成功
        self._tools[tool.name] = tool
        return True

    def get(self, tool_name: str) -> Tool:
        """获取工具"""
        if tool_name not in self._tools:
            raise KeyError(f"工具 [{tool_name}] 未注册")
        return self._tools[tool_name]

    def list_tools(self, category: ToolCategory = None) -> List[Tool]:
        """列出所有工具（可按分类过滤）"""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_tool_schemas(self) -> List[Dict]:
        """
        获取所有工具的Schema（v2.9: 返回标准 OpenAI Function Calling 格式）

        返回格式:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "...",
                        "description": "...",
                        "parameters": {...}
                    }
                },
                ...
            ]
        """
        return [tool.to_openai_function() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具（带安全检查）

        二次验证：
        1. 工具必须在白名单中
        2. 执行前再次检查参数中是否有危险操作

        v2.9: 自动过滤 collection_name 参数（上下文参数，非工具参数）
        """
        tool = self.get(tool_name)

        # v2.9: collection_name 是上下文参数，不在工具的 parameters schema 中
        # 但需要传递给需要它的工具函数
        collection_name = kwargs.pop("collection_name", "default")

        # 二次验证：检查参数中是否有危险SQL关键词
        for key, value in kwargs.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for keyword in self._blacklist_keywords[:7]:  # 只检查SQL关键词
                    if keyword in value_lower:
                        raise ValueError(f"❌ 参数 [{key}] 包含危险操作 '{keyword}'，拒绝执行")

        # 执行工具
        try:
            # v2.9: 如果工具函数接受 collection_name 参数，则传递
            import inspect
            sig = inspect.signature(tool.function)
            if "collection_name" in sig.parameters:
                kwargs["collection_name"] = collection_name
            result = tool.function(**kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f"工具 [{tool_name}] 执行失败: {str(e)}")


# 全局单例
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """获取全局工具注册中心"""
    return _registry
