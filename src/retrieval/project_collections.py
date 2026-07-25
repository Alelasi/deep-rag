"""多项目 Qdrant collection 注册表

与 ``scripts/build_all_projects_qdrant.py`` 的 PROJECTS 保持一致。
构建完成后，UI / API / 检索 用本表选库，避免写死字符串。
"""
from __future__ import annotations

from typing import Dict, List, Optional

# 逻辑项目名 → Qdrant collection
PROJECT_COLLECTIONS: Dict[str, str] = {
    "work": "proj_work",
    "thesis": "proj_thesis",
    "psychology": "proj_psychology",
    "social": "proj_social",
    "ideas": "proj_ideas",
    "assistant": "proj_assistant",
    "tools": "proj_tools",
    "worklog": "proj_worklog",
    "zhesi": "proj_zhesi",
    "api_router": "proj_api_router",
}

# 展示名（中文）
PROJECT_LABELS: Dict[str, str] = {
    "work": "工作区",
    "thesis": "论文",
    "psychology": "心理人际",
    "social": "社科",
    "ideas": "奇思妙想",
    "assistant": "助理",
    "tools": "工具",
    "worklog": "工作日志",
    "zhesi": "哲思灵智",
    "api_router": "api-router-ui",
}

# 默认检索库（构建完成后 deep-rag 主问答优先工作区）
DEFAULT_PROJECT = "work"
DEFAULT_COLLECTION = PROJECT_COLLECTIONS[DEFAULT_PROJECT]


def resolve_collection(project_or_collection: Optional[str] = None) -> str:
    """把项目别名或 collection 名解析为真实 collection。

    - None → proj_work
    - work / psychology → proj_*
    - proj_work → 原样返回
    """
    if not project_or_collection:
        return DEFAULT_COLLECTION
    key = project_or_collection.strip()
    if key in PROJECT_COLLECTIONS:
        return PROJECT_COLLECTIONS[key]
    if key.startswith("proj_"):
        return key
    # 兼容旧名
    legacy = {
        "default": DEFAULT_COLLECTION,
        "knowledge_base": DEFAULT_COLLECTION,
        "general_kb": "proj_zhesi",
        "psychology_kb": "proj_psychology",
    }
    return legacy.get(key, key)


def list_projects() -> List[dict]:
    """供 UI 下拉：[{key, label, collection}, ...]"""
    return [
        {
            "key": k,
            "label": PROJECT_LABELS.get(k, k),
            "collection": c,
        }
        for k, c in PROJECT_COLLECTIONS.items()
    ]
