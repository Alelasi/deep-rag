"""Agentic RAG Agent - 真正的自主决策Agent

基于ReAct模式（Reasoning + Acting）实现的自主检索Agent。
不同于固定Pipeline，Agent会：
1. 分析问题（Reasoning）
2. 选择工具（Acting）
3. 观察结果（Observation）
4. 决定下一步（Reflection）
5. 循环直到满意

参考：
- nageoffer-ai-马丁.md 第三章"Agent的工作循环"
- AI_Agent与RAG完整技术指南.md "Agentic RAG"章节
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.state import Document
from src.retrieval.agentic_tools import AgenticRAGToolbox
from src.retrieval.agent_router import BaseRouter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agentic_rag_agent")


class AgentStatus(Enum):
    """Agent状态"""
    THINKING = "thinking"       # 正在思考
    ACTING = "acting"           # 正在执行工具
    OBSERVING = "observing"     # 观察结果
    DECIDING = "deciding"       # 决策下一步
    FINISHED = "finished"       # 任务完成
    FAILED = "failed"           # 任务失败


@dataclass
class AgentStep:
    """Agent单步执行记录"""
    step_num: int
    reasoning: str              # 推理过程
    action: str                 # 选择的工具
    action_input: str           # 工具输入
    observation: str            # 工具输出摘要
    reflection: str             # 对结果的反思
    status: AgentStatus


class AgenticRAGAgent:
    """自主决策的RAG Agent

    核心特性：
    1. ReAct循环：Reasoning → Acting → Observing → Reflection
    2. 自主决策：根据检索结果质量决定是否继续
    3. 多轮检索：可以多次调用不同工具
    4. 动态调整：根据反馈调整检索策略

    示例流程：
    用户问："LangChain 0.3.0有什么新特性？"

    Step 1 (Reasoning): 这是版本特性查询，需要最新信息
    Step 1 (Acting): 选择 web_search 工具
    Step 1 (Observation): 找到3篇博客，但都是0.2.x版本
    Step 1 (Reflection): 结果不够新，需要补充向量检索

    Step 2 (Reasoning): Web没找到，可能本地文档有
    Step 2 (Acting): 选择 vector_search 工具
    Step 2 (Observation): 找到changelog文档，包含0.3.0更新
    Step 2 (Reflection): 信息充分，可以生成答案

    Step 3: 完成
    """

    def __init__(
        self,
        toolbox: AgenticRAGToolbox,
        router: BaseRouter,
        llm=None,
        max_steps: int = 5,
        min_docs_threshold: int = 2,
        quality_threshold: float = 0.7
    ):
        """
        Args:
            toolbox: 工具箱（提供4种检索工具）
            router: 路由器（选择工具）
            llm: 大模型（用于推理和反思，可选）
            max_steps: 最大迭代次数
            min_docs_threshold: 最少文档数阈值
            quality_threshold: 文档质量阈值（0-1）
        """
        self.toolbox = toolbox
        self.router = router
        self.llm = llm
        self.max_steps = max_steps
        self.min_docs_threshold = min_docs_threshold
        self.quality_threshold = quality_threshold

        self.history: List[AgentStep] = []
        self.all_documents: List[Document] = []

    def run(self, question: str) -> Dict[str, Any]:
        """运行Agent主循环

        Returns:
            {
                "documents": List[Document],  # 最终检索到的文档
                "steps": List[AgentStep],     # 执行历史
                "status": AgentStatus,        # 最终状态
                "reasoning": str              # 最终推理
            }
        """
        log.info(f"Agent启动，问题: {question}")
        self.history = []
        self.all_documents = []

        for step_num in range(1, self.max_steps + 1):
            log.info(f"=== Step {step_num} ===")

            # 1. Reasoning: 分析当前状态，决定策略
            reasoning = self._reason(question, step_num)
            log.info(f"Reasoning: {reasoning}")

            # 2. Acting: 选择工具并执行
            action = self._select_tool(question, reasoning)
            log.info(f"Acting: 使用工具 {action}")

            action_input = question  # 简化版：直接用原问题
            documents = self._execute_tool(action, action_input)

            # 3. Observing: 观察结果
            observation = self._observe(documents)
            log.info(f"Observation: {observation}")

            # 4. Reflection: 反思结果，决定下一步
            reflection, should_continue = self._reflect(
                question, documents, step_num
            )
            log.info(f"Reflection: {reflection}")

            # 记录步骤
            step = AgentStep(
                step_num=step_num,
                reasoning=reasoning,
                action=action,
                action_input=action_input,
                observation=observation,
                reflection=reflection,
                status=AgentStatus.DECIDING if should_continue else AgentStatus.FINISHED
            )
            self.history.append(step)
            self.all_documents.extend(documents)

            # 5. 决策：继续还是结束
            if not should_continue:
                log.info("Agent决定：信息充分，任务完成")
                return {
                    "documents": self._deduplicate_documents(self.all_documents),
                    "steps": self.history,
                    "status": AgentStatus.FINISHED,
                    "reasoning": f"经过{step_num}步检索，找到{len(self.all_documents)}个相关文档"
                }

        # 达到最大步数
        log.warning(f"达到最大步数{self.max_steps}，强制结束")
        return {
            "documents": self._deduplicate_documents(self.all_documents),
            "steps": self.history,
            "status": AgentStatus.FAILED,
            "reasoning": f"达到最大步数{self.max_steps}，但结果仍不理想"
        }

    def _reason(self, question: str, step_num: int) -> str:
        """推理：分析问题和当前状态

        简化版实现：基于规则推理
        完整版可以用LLM生成推理过程（Few-shot Prompt）
        """
        if step_num == 1:
            # 第一步：分析问题类型
            if any(kw in question for kw in ["最新", "2026", "近期", "目前"]):
                return "这是时效性查询，应该先尝试网络搜索"
            elif any(kw in question for kw in ["关系", "依赖", "调用链"]):
                return "这是关系查询，应该使用图检索"
            elif any(kw in question for kw in ["ID", "订单号", "版本号"]):
                return "这是精确查询，应该使用精确匹配"
            else:
                return "这是语义查询，应该使用向量检索"
        else:
            # 后续步骤：分析已有结果
            doc_count = len(self.all_documents)
            if doc_count == 0:
                return f"前{step_num-1}步未找到相关文档，尝试切换检索策略"
            elif doc_count < self.min_docs_threshold:
                return f"只找到{doc_count}个文档，数量不够，需要补充检索"
            else:
                # 检查文档质量（简化版：检查是否包含问题关键词）
                keywords = self._extract_keywords(question)
                relevant_count = sum(
                    1 for doc in self.all_documents
                    if any(kw.lower() in doc.get("content", "").lower() for kw in keywords)
                )
                if relevant_count < self.min_docs_threshold:
                    return f"文档相关性不高（{relevant_count}/{doc_count}），需要调整策略"
                else:
                    return f"找到{doc_count}个文档，其中{relevant_count}个高度相关，可以结束"

    def _select_tool(self, question: str, reasoning: str) -> str:
        """选择工具

        两种模式：
        1. 使用Router自动选择（当前实现）
        2. 根据reasoning用LLM选择（高级版）
        """
        # 当前实现：使用Router
        tool_name = self.router.route(question)

        # 如果已经用过这个工具，尝试换一个
        used_tools = [step.action for step in self.history]
        if tool_name in used_tools and len(self.history) > 0:
            # 简单策略：切换到vector_search（通用）
            if tool_name != "vector_search" and "vector_search" not in used_tools:
                log.info(f"工具{tool_name}已使用过，切换到vector_search")
                return "vector_search"

        return tool_name

    def _execute_tool(self, tool_name: str, query: str) -> List[Document]:
        """执行工具"""
        try:
            tool = self.toolbox.get_tool(tool_name)
            if tool is None:
                log.warning(f"工具{tool_name}不存在，回退到vector_search")
                tool = self.toolbox.get_tool("vector_search")

            documents = tool.search(query, top_k=5)
            return documents
        except Exception as e:
            log.error(f"工具执行失败: {e}")
            return []

    def _observe(self, documents: List[Document]) -> str:
        """观察结果"""
        if not documents:
            return "未找到任何文档"

        # 统计文档信息（Document是TypedDict，用字典方式访问）
        scores = [doc.get("score", 0.5) for doc in documents]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return f"找到{len(documents)}个文档，平均相关度{avg_score:.2f}"

    def _reflect(
        self,
        question: str,
        documents: List[Document],
        step_num: int
    ) -> tuple[str, bool]:
        """反思：评估结果质量，决定是否继续

        Returns:
            (reflection: str, should_continue: bool)
        """
        total_docs = len(self.all_documents) + len(documents)

        # 条件1：文档数量充足
        if total_docs >= self.min_docs_threshold * 2:
            return f"已找到{total_docs}个文档，数量充足", False

        # 条件2：质量达标
        if documents:
            scores = [doc.get("score", 0.5) for doc in documents]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score >= self.quality_threshold and total_docs >= self.min_docs_threshold:
                return f"文档质量达标（{avg_score:.2f}），可以结束", False

        # 条件3：达到最大步数
        if step_num >= self.max_steps:
            return "达到最大步数，强制结束", False

        # 继续检索
        if not documents:
            return "本次检索无结果，尝试其他策略", True
        else:
            scores = [doc.get("score", 0.5) for doc in documents]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return f"文档质量一般（{avg_score:.2f}），继续补充", True

    def _extract_keywords(self, question: str) -> List[str]:
        """提取问题关键词（简化版）"""
        # 移除常见停用词
        stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        words = [w for w in question.split() if w not in stopwords and len(w) > 1]
        return words[:5]  # 取前5个关键词

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """去重文档"""
        seen = set()
        unique_docs = []
        for doc in documents:
            # 简单去重：基于content hash
            content = doc.get("content", "")
            content_hash = hash(content[:100])  # 用前100字符作为指纹
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        return unique_docs


# === 高级版：LLM驱动的Agent（可选实现） ===

class LLMAgenticRAGAgent(AgenticRAGAgent):
    """用LLM驱动的Agentic RAG Agent

    与基础版的区别：
    - Reasoning用LLM生成（Few-shot CoT）
    - Tool Selection用LLM选择（基于工具描述）
    - Reflection用LLM评估（LLM-as-Judge）

    优势：更智能、更灵活
    劣势：成本更高、延迟更大
    """

    def __init__(self, *args, llm, **kwargs):
        super().__init__(*args, llm=llm, **kwargs)
        if llm is None:
            raise ValueError("LLMAgenticRAGAgent必须提供llm参数")

    def _reason(self, question: str, step_num: int) -> str:
        """用LLM生成推理过程（Few-shot CoT）"""

        # 构建历史上下文
        history_text = "\n".join([
            f"Step {s.step_num}: {s.action} → {s.observation}"
            for s in self.history
        ])

        prompt = f"""你是一个检索策略专家。分析用户问题和当前检索状态，给出下一步推理。

【示例】
问题：LangChain 0.3.0有什么新特性？
当前状态：第1步，尚未检索
推理：这是版本特性查询，需要最新信息，应该先尝试网络搜索。

问题：如何使用ChatPromptTemplate？
当前状态：第1步，尚未检索
推理：这是API用法查询，应该使用向量检索查找文档。

【当前任务】
问题：{question}
当前状态：第{step_num}步
已执行操作：
{history_text if history_text else "无"}

请给出推理（一句话）："""

        # 调用LLM
        try:
            response = self.llm.invoke(prompt)
            reasoning = response.content.strip()
            return reasoning
        except Exception as e:
            log.error(f"LLM推理失败: {e}，回退到规则推理")
            return super()._reason(question, step_num)

    def _reflect(
        self,
        question: str,
        documents: List[Document],
        step_num: int
    ) -> tuple[str, bool]:
        """用LLM评估结果质量（LLM-as-Judge）"""

        # 构建文档摘要
        doc_summary = "\n".join([
            f"- [{doc.score:.2f}] {doc.content[:100]}..."
            for doc in documents[:3]
        ])

        prompt = f"""你是一个检索质量评估专家。判断当前检索结果是否充分。

【用户问题】
{question}

【检索结果】
{doc_summary if doc_summary else "无结果"}

【已累计文档】
{len(self.all_documents)}个

【判断标准】
1. 文档数量是否足够（至少2个）
2. 文档相关性是否高（>0.7）
3. 文档是否覆盖问题核心内容

只返回JSON：{{"reflection": "评估结果", "should_continue": true/false}}"""

        try:
            response = self.llm.invoke(prompt)
            import json
            import re
            content = response.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result["reflection"], result["should_continue"]
            else:
                raise ValueError("无法解析LLM返回")
        except Exception as e:
            log.error(f"LLM反思失败: {e}，回退到规则反思")
            return super()._reflect(question, documents, step_num)
