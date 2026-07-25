"""受约束解码 — v2.9.1新增

强制LLM输出特定格式（JSON），消除正则解析失败风险。

两种模式：
- Ollama: 使用 format="json" 参数
- API后端: 使用 LangChain with_structured_output()
"""
import json
import logging
from typing import Any, Optional

log = logging.getLogger("deeprag")


# === 预定义Schema ===

HALLUCINATION_SCHEMA = {
    "type": "object",
    "properties": {
        "hallucination_score": {
            "type": "number",
            "description": "幻觉分数 0-1, 0=无幻觉, 1=完全幻觉",
        },
        "passed": {
            "type": "boolean",
            "description": "是否通过事实校验",
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "不受文档支撑的断言列表",
        },
        "reasoning": {
            "type": "string",
            "description": "校验推理过程",
        },
    },
    "required": ["hallucination_score", "passed", "unsupported_claims", "reasoning"],
}

DOC_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "相关度评分 0-1",
        },
        "reasoning": {
            "type": "string",
            "description": "评分理由",
        },
    },
    "required": ["score", "reasoning"],
}

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["agree", "conflict", "partial"],
            "description": "对比结论: agree=一致, conflict=矛盾, partial=部分差异",
        },
        "conflict_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "矛盾点列表",
        },
        "recommendation": {
            "type": "string",
            "enum": ["merge", "prefer_a", "prefer_b", "re_search"],
            "description": "处理建议: merge=融合, prefer_a/b=选某方, re_search=需重搜",
        },
    },
    "required": ["verdict", "conflict_points", "recommendation"],
}

# 批量文档评分Schema（v2.9.2新增）
DOC_GRADE_ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "文档序号（从1开始）",
            },
            "grade": {
                "type": "string",
                "enum": ["relevant", "ambiguous", "irrelevant"],
                "description": "相关性评级",
            },
            "relevance_score": {
                "type": "number",
                "description": "相关度评分 0-1",
            },
            "reasoning": {
                "type": "string",
                "description": "评分理由",
            },
        },
        "required": ["index", "grade", "relevance_score", "reasoning"],
    },
}

# 查询分析Schema（v2.9.2新增）
QUERY_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["factual", "analytical", "creative", "code", "conversational"],
            "description": "查询意图类型",
        },
        "complexity": {
            "type": "string",
            "enum": ["simple", "medium", "complex"],
            "description": "查询复杂度",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "关键词列表",
        },
        "requires_web": {
            "type": "boolean",
            "description": "是否需要网络搜索",
        },
        "requires_graph": {
            "type": "boolean",
            "description": "是否需要知识图谱",
        },
    },
    "required": ["intent", "complexity", "keywords"],
}


class StructuredOllamaWrapper:
    """Ollama结构化输出包装器：使用format='json'参数

    Ollama原生支持 format="json" 参数强制JSON输出。
    """

    def __init__(self, llm, schema: dict):
        self.llm = llm
        self.schema = schema

    def invoke(self, messages, **kwargs) -> dict:
        """调用LLM并返回结构化dict"""
        # 方法1: 如果llm是ChatOllama，使用format参数
        try:
            response = self.llm.invoke(messages, format="json", **kwargs)
        except TypeError:
            # format参数不被支持，降级到普通调用
            response = self.llm.invoke(messages, **kwargs)

        content = response.content if hasattr(response, "content") else str(response)

        # 解析JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试从markdown代码块中提取
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
                data = json.loads(json_str.strip())
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
            else:
                log.warning(f"[ConstrainedDecoder] JSON解析失败: {content[:100]}")
                return self._default_value()

        # 验证并填充默认值
        return self._validate(data)

    def _validate(self, data: dict) -> dict:
        """验证数据符合schema，填充缺失字段"""
        properties = self.schema.get("properties", {})
        required = self.schema.get("required", [])

        result = {}
        for field_name, field_schema in properties.items():
            if field_name in data:
                result[field_name] = data[field_name]
            elif field_name in required:
                # 填充类型默认值
                field_type = field_schema.get("type", "string")
                if field_type == "number":
                    result[field_name] = 0.0
                elif field_type == "boolean":
                    result[field_name] = False
                elif field_type == "array":
                    result[field_name] = []
                elif field_type == "string":
                    enum = field_schema.get("enum")
                    result[field_name] = enum[0] if enum else ""
                else:
                    result[field_name] = None
            # 可选字段缺失时不填充

        return result

    def _default_value(self) -> dict:
        """返回schema的默认值"""
        return self._validate({})


def get_structured_llm(llm, schema: dict, backend: str = "auto"):
    """返回强制结构化输出的LLM

    Args:
        llm: 原始LLM实例
        schema: JSON Schema定义
        backend: "ollama"用format参数, "api"用with_structured_output, "auto"自动检测

    Returns:
        结构化输出LLM包装器
    """
    if backend == "auto":
        # 检测后端类型
        llm_type = getattr(llm, "_llm_type", "")
        base_url = str(getattr(llm, "base_url", ""))
        if "ollama" in llm_type or "11434" in base_url:
            backend = "ollama"
        else:
            backend = "api"

    if backend == "ollama":
        return StructuredOllamaWrapper(llm, schema)
    else:
        # API后端：尝试LangChain的with_structured_output
        try:
            return llm.with_structured_output(schema)
        except Exception as e:
            log.warning(f"[ConstrainedDecoder] with_structured_output不支持，降级到JSON包装器: {e}")
            return StructuredOllamaWrapper(llm, schema)


def safe_parse_json(content: str, schema: dict = None) -> Optional[dict]:
    """安全解析JSON内容（带fallback）

    用于在结构化输出失败时的降级解析。
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试从markdown代码块提取
        for delimiter in ["```json", "```"]:
            if delimiter in content:
                try:
                    json_str = content.split(delimiter)[1].split("```")[0]
                    return json.loads(json_str.strip())
                except (IndexError, json.JSONDecodeError):
                    continue

        # 尝试找到第一个{和最后一个}
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

        log.warning(f"[ConstrainedDecoder] JSON解析完全失败: {content[:100]}")
        return None
