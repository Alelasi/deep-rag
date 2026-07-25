"""Skill系统抽象层 — v2.9.1新增

声明式工具注册 + 自动Schema生成 + 依赖注入 + 渐进式加载

三层渐进式加载（Progressive Disclosure）：
1. 第一层（启动时）：只加载name+description（30-50 token/skill）
2. 第二层（匹配时）：加载完整docstring和参数说明
3. 第三层（执行时）：按需加载脚本/模板/参考文档

用法：
    from src.tools.skill_system import skill, SkillRegistry

    @skill(name="search_knowledge_base", description="搜索知识库文档", risk="SAFE")
    def search_kb(query: str, collection_name: str = "default", top_k: int = 5) -> str:
        '''搜索指定知识库，返回top-k相关文档块

        Args:
            query: 搜索关键词
            collection_name: 知识库集合名
            top_k: 返回文档数量
        '''
        ...

    # 执行
    registry = SkillRegistry()
    result = registry.execute("search_knowledge_base", query="INTJ", collection_name="mbti")
"""
import inspect
import logging
from typing import Any, Callable, Optional, get_type_hints
from dataclasses import dataclass, field
from functools import wraps

log = logging.getLogger("deeprag")


# Python类型 → JSON Schema 类型映射
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class SkillMetadata:
    """Skill元数据（第一层：轻量级，30-50 token）"""
    name: str
    description: str
    risk: str = "SAFE"  # SAFE / MODERATE / DANGEROUS
    version: str = "1.0"


@dataclass
class SkillInstructions:
    """Skill指令（第二层：完整描述）"""
    metadata: SkillMetadata
    docstring: str
    params: dict  # {param_name: {type, description, default, required}}


@dataclass
class Skill:
    """完整Skill定义（内部使用）"""
    func: Callable
    metadata: SkillMetadata
    instructions: SkillInstructions
    schema: dict
    _resources_loaded: bool = False


def _python_type_to_json(py_type: type) -> str:
    """Python类型转JSON Schema类型"""
    return _TYPE_MAP.get(py_type, "string")


def _generate_schema(func: Callable, metadata: SkillMetadata) -> dict:
    """从函数签名自动生成JSON Schema"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        py_type = hints.get(param_name, str)
        json_type = _python_type_to_json(py_type)

        prop = {"type": json_type}

        # 默认值
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        # 从docstring提取参数描述
        properties[param_name] = prop

    schema = {
        "name": metadata.name,
        "description": metadata.description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }

    if required:
        schema["parameters"]["required"] = required

    return schema


def _extract_param_docs(func: Callable) -> dict:
    """从docstring提取参数说明"""
    doc = inspect.getdoc(func) or ""
    params = {}
    in_args = False
    current_param = None

    for line in doc.split("\n"):
        line = line.strip()
        if line.lower().startswith("args:"):
            in_args = True
            continue
        elif line.lower().startswith("returns:") or line.lower().startswith("raises:"):
            in_args = False
            continue

        if in_args and line:
            # 格式: param_name: description
            if ":" in line and not line.startswith(" "):
                parts = line.split(":", 1)
                current_param = parts[0].strip()
                params[current_param] = parts[1].strip() if len(parts) > 1 else ""
            elif current_param:
                params[current_param] += " " + line

    return params


def skill(name: str, description: str, risk: str = "SAFE", version: str = "1.0"):
    """@skill 装饰器 — 声明式注册工具

    Args:
        name: Skill唯一标识（英文，用于API调用）
        description: 简短描述（30-50字，第一层加载时展示）
        risk: 风险等级 SAFE/MODERATE/DANGEROUS
        version: 版本号

    用法：
        @skill(name="search_kb", description="搜索知识库")
        def search_kb(query: str, top_k: int = 5) -> str:
            '''搜索知识库文档'''
            ...
    """

    def decorator(func: Callable) -> Callable:
        metadata = SkillMetadata(
            name=name,
            description=description,
            risk=risk,
            version=version,
        )

        # 提取参数文档
        param_docs = _extract_param_docs(func)

        # 构建参数信息
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        params_info = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            params_info[param_name] = {
                "type": _python_type_to_json(hints.get(param_name, str)),
                "description": param_docs.get(param_name, ""),
                "default": param.default if param.default is not inspect.Parameter.empty else None,
                "required": param.default is inspect.Parameter.empty,
            }

        instructions = SkillInstructions(
            metadata=metadata,
            docstring=inspect.getdoc(func) or "",
            params=params_info,
        )

        schema = _generate_schema(func, metadata)

        # 创建Skill对象
        skill_obj = Skill(
            func=func,
            metadata=metadata,
            instructions=instructions,
            schema=schema,
        )

        # 在函数上附加skill对象
        func._skill = skill_obj

        # 自动注册到全局registry
        _global_registry.register(skill_obj)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._skill = skill_obj
        return wrapper

    return decorator


class SkillRegistry:
    """Skill注册中心 — 管理注册、发现、加载、执行

    三层渐进式加载：
    1. get_metadata() — 启动时调用，只返回name+description（省context）
    2. get_instructions() — 匹配到skill后调用，返回完整指令
    3. get_resources() — 执行时调用，按需加载脚本/模板
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._context: dict = {}  # 依赖注入上下文

    def register(self, skill_obj: Skill):
        """注册一个Skill"""
        name = skill_obj.metadata.name

        # 安全检查：DANGEROUS级别需要确认
        if skill_obj.metadata.risk == "DANGEROUS":
            log.warning(f"[SkillRegistry] 注册危险Skill: {name}（已跳过）")
            return

        self._skills[name] = skill_obj
        log.debug(f"[SkillRegistry] 已注册: {name} ({skill_obj.metadata.description[:30]}...)")

    def set_context(self, **kwargs):
        """设置依赖注入上下文

        常见注入参数：
        - collection_name: 当前知识库集合名
        - retriever: 检索器实例
        - chroma_client: ChromaDB客户端
        """
        self._context.update(kwargs)

    # === 第一层：轻量元数据 ===

    def get_metadata(self, name: str) -> Optional[SkillMetadata]:
        """第一层：获取Skill元数据（30-50 token）"""
        skill_obj = self._skills.get(name)
        return skill_obj.metadata if skill_obj else None

    def list_metadata(self) -> list[SkillMetadata]:
        """第一层：列出所有Skill的元数据"""
        return [s.metadata for s in self._skills.values()]

    def get_metadata_summary(self) -> str:
        """第一层：获取所有Skill的摘要文本（用于LLM context）"""
        lines = []
        for s in self._skills.values():
            m = s.metadata
            lines.append(f"- {m.name}: {m.description}")
        return "\n".join(lines)

    # === 第二层：完整指令 ===

    def get_instructions(self, name: str) -> Optional[SkillInstructions]:
        """第二层：获取完整指令（docstring + 参数说明）"""
        skill_obj = self._skills.get(name)
        return skill_obj.instructions if skill_obj else None

    def get_instructions_text(self, name: str) -> str:
        """第二层：获取格式化的指令文本"""
        skill_obj = self._skills.get(name)
        if not skill_obj:
            return ""

        inst = skill_obj.instructions
        lines = [f"## {inst.metadata.name}", inst.docstring, ""]

        if inst.params:
            lines.append("参数:")
            for param_name, info in inst.params.items():
                req = "必填" if info["required"] else f"可选(默认: {info['default']})"
                lines.append(f"  - {param_name} ({info['type']}): {info['description']} [{req}]")

        return "\n".join(lines)

    # === 第三层：资源加载 ===

    def get_resources(self, name: str) -> list:
        """第三层：按需加载资源（脚本/模板/参考文档）

        当前实现返回空列表，实际资源在具体Skill中按需加载。
        """
        skill_obj = self._skills.get(name)
        if not skill_obj:
            return []

        if not skill_obj._resources_loaded:
            # 标记为已加载（实际资源加载逻辑在具体Skill中实现）
            skill_obj._resources_loaded = True
            log.debug(f"[SkillRegistry] 资源已加载: {name}")

        return []

    # === Schema生成 ===

    def get_schema(self, name: str) -> Optional[dict]:
        """获取Skill的JSON Schema（OpenAI Function Calling兼容）"""
        skill_obj = self._skills.get(name)
        return skill_obj.schema if skill_obj else None

    def get_all_schemas(self) -> list[dict]:
        """获取所有Skill的Schema列表"""
        return [s.schema for s in self._skills.values()]

    # === 执行（带依赖注入）===

    def execute(self, name: str, **kwargs) -> Any:
        """执行Skill（带依赖注入）

        自动从context中注入缺失的参数：
        - 如果函数签名包含 collection_name 但kwargs未提供，从context注入
        - 如果函数签名包含 retriever 但kwargs未提供，从context注入
        """
        skill_obj = self._skills.get(name)
        if not skill_obj:
            raise ValueError(f"Skill未注册: {name}")

        # 安全检查
        if skill_obj.metadata.risk == "DANGEROUS":
            raise PermissionError(f"拒绝执行危险Skill: {name}")

        # 确保资源已加载（第三层）
        self.get_resources(name)

        # 依赖注入：检查函数签名，从context补充缺失参数
        sig = inspect.signature(skill_obj.func)
        injected = kwargs.copy()

        for param_name, param in sig.parameters.items():
            if param_name in injected:
                continue
            if param_name in self._context:
                injected[param_name] = self._context[param_name]
            elif param.default is not inspect.Parameter.empty:
                continue  # 有默认值，不注入
            else:
                # 必填参数缺失
                log.warning(f"[SkillRegistry] Skill '{name}' 缺少必填参数: {param_name}")

        return skill_obj.func(**injected)

    def has(self, name: str) -> bool:
        """检查Skill是否已注册"""
        return name in self._skills

    def count(self) -> int:
        """已注册Skill数量"""
        return len(self._skills)


# 全局注册中心
_global_registry = SkillRegistry()


def get_skill_registry() -> SkillRegistry:
    """获取全局SkillRegistry实例"""
    return _global_registry


def register_builtin_skills():
    """注册内置Skill（从现有工具迁移）

    将 agentic_tools.py 中的4个检索工具用 @skill 重新声明。
    在graph.py初始化时调用。
    """
    # 避免重复注册
    if _global_registry.count() > 0:
        return _global_registry

    try:
        from src.retrieval.agentic_tools import (
            ExactMatchTool,
            VectorSearchTool,
            GraphSearchTool,
            WebSearchTool,
        )

        # ExactMatch
        @skill(name="exact_match", description="精确匹配查询，从SQLite检索完全匹配的文档")
        def exact_match(query: str, collection_name: str = "default") -> str:
            """精确匹配查询

            Args:
                query: 精确匹配关键词
                collection_name: 知识库集合名
            """
            tool = ExactMatchTool()
            results = tool.search(query, collection_name=collection_name)
            import json
            return json.dumps(results, ensure_ascii=False) if results else "未找到精确匹配结果"

        # VectorSearch
        @skill(name="vector_search", description="向量语义检索，返回与问题最相关的文档块")
        def vector_search(query: str, collection_name: str = "default", top_k: int = 5) -> str:
            """向量语义检索

            Args:
                query: 搜索问题
                collection_name: 知识库集合名
                top_k: 返回文档数量
            """
            tool = VectorSearchTool()
            results = tool.search(query, collection_name=collection_name, top_k=top_k)
            import json
            return json.dumps(results, ensure_ascii=False) if results else "未找到相关文档"

        # GraphSearch
        @skill(name="graph_search", description="知识图谱查询，检索实体间关系")
        def graph_search(entity: str, relation: str = "", collection_name: str = "default") -> str:
            """知识图谱查询

            Args:
                entity: 查询实体名
                relation: 关系类型（可选）
                collection_name: 知识库集合名
            """
            tool = GraphSearchTool()
            results = tool.search(entity, relation=relation)
            import json
            return json.dumps(results, ensure_ascii=False) if results else "未找到图谱关系"

        # WebSearch
        @skill(name="web_search", description="网络搜索，从DuckDuckGo获取实时信息")
        def web_search(query: str, max_results: int = 3) -> str:
            """网络搜索

            Args:
                query: 搜索关键词
                max_results: 最大返回结果数
            """
            tool = WebSearchTool()
            results = tool.search(query, max_results=max_results)
            import json
            return json.dumps(results, ensure_ascii=False) if results else "未找到网络搜索结果"

        log.info(f"[SkillRegistry] 已注册 {_global_registry.count()} 个内置Skill")

    except ImportError as e:
        log.warning(f"[SkillRegistry] 无法导入agentic_tools，跳过内置Skill注册: {e}")
    except Exception as e:
        log.error(f"[SkillRegistry] 注册内置Skill失败: {e}")

    return _global_registry
