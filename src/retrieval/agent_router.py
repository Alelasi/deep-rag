"""Agent决策路由器
根据问题类型动态选择AgenticRAGToolbox中的工具

理论依据（《AI Agent与RAG完整技术指南》）：
- 传统混合检索：固定流程，所有问题走相同路径
- Agentic RAG：Agent分析问题特征，动态选择最合适的工具
- 提供两种实现：
  1. RuleBasedRouter：基于关键词/正则的规则路由（无LLM依赖，零延迟）
  2. LLMRouter：基于LLM推理的智能路由（精度更高，成本/延迟略增）
"""
import re
from typing import List, Optional
from src.retrieval.agentic_tools import AgenticRAGToolbox
from src.state import Document


class BaseRouter:
    """路由器基类"""

    def route(self, question: str) -> str:
        """根据问题选择工具名

        Returns:
            工具名（应在toolbox中已注册）
        """
        raise NotImplementedError


class RuleBasedRouter(BaseRouter):
    """基于规则的路由器（无LLM依赖）

    决策规则（按优先级）：
    1. 包含明确ID/编号 → exact_match
    2. 包含关系词（"和...的关系"、"依赖"、"继承"） → graph_search
    3. 包含时效词（"最新"、"2026"、"近期"） → web_search
    4. 默认 → vector_search
    """

    # 精确匹配触发词
    EXACT_PATTERNS = [
        re.compile(r"用户ID[\s:：]?\w+"),
        re.compile(r"订单号[\s:：]?\w+"),
        re.compile(r"版本[\s:：]?[vV]?\d+"),
        re.compile(r"ID为\s*\w+"),
        re.compile(r"编号[\s:：]?\w+"),
    ]

    # 关系查询关键词
    GRAPH_KEYWORDS = [
        "之间的关系", "和.*的关系", "依赖", "继承自", "调用关系",
        "关联", "上下游", "调用链", "依赖图",
    ]

    # 时效查询关键词
    WEB_KEYWORDS = [
        "最新", "近期", "今年", "刚刚", "目前", "现在的",
        "2026", "2025年下半年", "this week", "latest",
    ]

    def __init__(self, default_tool: str = "vector_search"):
        self.default_tool = default_tool

    def route(self, question: str) -> str:
        # 1. 精确ID匹配
        for pattern in self.EXACT_PATTERNS:
            if pattern.search(question):
                return "exact_match"

        # 2. 关系查询
        for kw in self.GRAPH_KEYWORDS:
            if re.search(kw, question):
                return "graph_search"

        # 3. 时效查询
        for kw in self.WEB_KEYWORDS:
            if kw in question:
                return "web_search"

        # 4. 默认
        return self.default_tool


class LLMRouter(BaseRouter):
    """基于LLM的智能路由器

    通过Prompt让LLM根据问题特征+工具描述选择最合适的工具。
    精度高于规则路由，但带来约200-500ms LLM调用延迟。
    """

    DECISION_PROMPT = """你是一个检索工具路由器。根据用户问题，从以下工具中选择最合适的一个。

可用工具：
{tools_description}

用户问题：{question}

请只输出工具名（如：vector_search），不要输出其他内容。"""

    def __init__(self, llm, toolbox: AgenticRAGToolbox,
                 fallback_tool: str = "vector_search"):
        """
        Args:
            llm: LLM实例（需有invoke方法或可调用）
            toolbox: 工具箱（用于读取tool descriptions）
            fallback_tool: LLM调用失败时的fallback工具
        """
        self.llm = llm
        self.toolbox = toolbox
        self.fallback_tool = fallback_tool

    def _build_tools_description(self) -> str:
        """构造工具描述供LLM参考"""
        tools = self.toolbox.list_tools()
        lines = []
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(lines)

    def route(self, question: str) -> str:
        prompt = self.DECISION_PROMPT.format(
            tools_description=self._build_tools_description(),
            question=question,
        )

        try:
            # 兼容langchain LLM和原生callable
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt)
                # 提取文本：langchain返回的是AIMessage
                text = response.content if hasattr(response, "content") else str(response)
            else:
                text = self.llm(prompt)
        except Exception as e:
            print(f"[LLMRouter] LLM call failed: {e}, using fallback")
            return self.fallback_tool

        # 解析输出（取第一个匹配的工具名）
        text = text.strip()
        registered = set(self.toolbox.tools.keys())
        for tool_name in registered:
            if tool_name in text:
                return tool_name

        # 找不到匹配工具时fallback
        print(f"[LLMRouter] No tool matched in '{text}', using fallback")
        return self.fallback_tool


class AgenticRetriever:
    """Agentic检索器：组合Router + Toolbox

    使用方式：
        toolbox = create_toolbox(vector_retriever)
        toolbox.register_tool("exact_match", ExactMatchTool(db))
        retriever = AgenticRetriever(toolbox, RuleBasedRouter())
        docs = retriever.retrieve("INTJ的主导功能")  # 自动路由
    """

    def __init__(self, toolbox: AgenticRAGToolbox, router: BaseRouter):
        self.toolbox = toolbox
        self.router = router

    def retrieve(self, question: str, top_k: int = 5,
                 force_tool: Optional[str] = None,
                 **tool_kwargs) -> List[Document]:
        """根据Router决策选择工具并检索

        Args:
            question: 用户问题
            top_k: 返回文档数
            force_tool: 强制使用指定工具（跳过Router）
            **tool_kwargs: 透传给工具的额外参数

        Returns:
            (chosen_tool_name, documents) 元组
        """
        chosen = force_tool or self.router.route(question)
        # 不同工具签名略不同，VectorSearch/WebSearch接受top_k
        if "max_results" in tool_kwargs or "filters" in tool_kwargs:
            results = self.toolbox.execute_tool(chosen, question, **tool_kwargs)
        else:
            try:
                results = self.toolbox.execute_tool(chosen, question, top_k=top_k)
            except TypeError:
                # 工具不接受top_k参数（如ExactMatchTool）
                results = self.toolbox.execute_tool(chosen, question)
        return results

    def retrieve_with_decision(self, question: str, top_k: int = 5) -> dict:
        """检索并返回决策信息

        Returns:
            {"tool": str, "documents": List[Document], "question": str}
        """
        chosen = self.router.route(question)
        results = self.retrieve(question, top_k=top_k, force_tool=chosen)
        return {
            "tool": chosen,
            "documents": results,
            "question": question,
        }
