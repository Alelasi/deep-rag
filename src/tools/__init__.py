"""
工具模块 - 白名单沙箱 + Function Calling + Skill系统 + MCP Server（v2.9.2）
"""

from .tool_registry import ToolRegistry, Tool, ToolCategory, ToolRisk, get_registry
from .builtin_tools import search_database, register_builtin_tools
# 旧名 agent_executor.py 已升级为 agent_executor_v2.py，保留旧导出名兼容调用方
from .agent_executor_v2 import AgentExecutorV2 as AgentExecutor
from .skill_system import skill, SkillRegistry, get_skill_registry, register_builtin_skills
from .mcp_server import MCPServer, run_stdio

__all__ = [
    "ToolRegistry",
    "Tool",
    "ToolCategory",
    "ToolRisk",
    "get_registry",
    "search_database",
    "register_builtin_tools",
    "AgentExecutor",
    # v2.9.1: Skill系统
    "skill",
    "SkillRegistry",
    "get_skill_registry",
    "register_builtin_skills",
    # v2.9.2: MCP Server
    "MCPServer",
    "run_stdio",
]
