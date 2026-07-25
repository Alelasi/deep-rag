"""Prompt Engineering五要素模板系统 — v2.9.2新增

面试要点（04-16 Prompt Engineering）：
- 五要素：Role/Task/Context/Format/Examples
- Prompt是工程问题，需要测试集+持续迭代
- Few-shot比纯文字描述效果好得多
- 每次只改一处，避免多变量干扰

本模块提供：
1. 五要素Prompt模板
2. Few-shot示例管理
3. Prompt版本控制
4. A/B测试支持

用法：
    from src.llm.prompt_templates import PromptBuilder, PromptTemplates

    # 使用Builder模式构建Prompt
    prompt = (PromptBuilder()
        .role("知识库问答专家")
        .task("根据文档回答问题")
        .context("参考文档：...")
        .format("结论 → 证据 → 引用")
        .example("什么是RAG？", "RAG是...")
        .build())

    # 使用预定义模板
    prompt = PromptTemplates.rag_answer(question, context)
"""
import hashlib
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

log = logging.getLogger("deeprag")


# ============================================================
# 1. Prompt数据结构
# ============================================================

@dataclass
class FewShotExample:
    """Few-shot示例"""
    input: str           # 输入
    output: str          # 期望输出
    description: str = ""  # 示例说明

    def to_text(self) -> str:
        """转换为文本格式"""
        if self.description:
            return f"【{self.description}】\n输入：{self.input}\n输出：{self.output}"
        return f"输入：{self.input}\n输出：{self.output}"


@dataclass
class PromptTemplate:
    """Prompt模板（五要素）"""
    name: str
    version: str = "1.0"
    role: str = ""
    task: str = ""
    context: str = ""
    format_instructions: str = ""
    examples: List[FewShotExample] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, variables: Optional[Dict[str, str]] = None) -> str:
        """渲染模板

        Args:
            variables: 变量替换（如 {question}, {context}）

        Returns:
            渲染后的Prompt文本
        """
        parts = []

        # Role
        if self.role:
            parts.append(f"## 角色\n{self.role}")

        # Task
        if self.task:
            task_text = self.task
            if variables:
                for k, v in variables.items():
                    task_text = task_text.replace(f"{{{k}}}", v)
            parts.append(f"## 任务\n{task_text}")

        # Context
        if self.context:
            context_text = self.context
            if variables:
                for k, v in variables.items():
                    context_text = context_text.replace(f"{{{k}}}", v)
            parts.append(f"## 背景\n{context_text}")

        # Format
        if self.format_instructions:
            parts.append(f"## 输出格式\n{self.format_instructions}")

        # Constraints
        if self.constraints:
            constraints_text = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"## 约束\n{constraints_text}")

        # Examples
        if self.examples:
            examples_text = "\n\n".join(
                f"### 示例{i+1}\n{ex.to_text()}"
                for i, ex in enumerate(self.examples)
            )
            parts.append(f"## 示例\n{examples_text}")

        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "role": self.role,
            "task": self.task,
            "context": self.context,
            "format_instructions": self.format_instructions,
            "examples_count": len(self.examples),
            "constraints_count": len(self.constraints),
        }


# ============================================================
# 2. Prompt Builder（Builder模式）
# ============================================================

class PromptBuilder:
    """Prompt构建器 — Builder模式

    链式调用构建五要素Prompt。

    用法：
        prompt = (PromptBuilder()
            .role("知识库问答专家")
            .task("根据文档回答问题")
            .context("参考文档：...")
            .format("结论 → 证据 → 引用")
            .example("什么是RAG？", "RAG是检索增强生成...")
            .constraint("只基于文档回答")
            .build())
    """

    def __init__(self, name: str = "custom"):
        self._name = name
        self._role = ""
        self._task = ""
        self._context = ""
        self._format = ""
        self._examples: List[FewShotExample] = []
        self._constraints: List[str] = []

    def role(self, description: str) -> "PromptBuilder":
        """设定角色"""
        self._role = description
        return self

    def task(self, description: str) -> "PromptBuilder":
        """设定任务"""
        self._task = description
        return self

    def context(self, description: str) -> "PromptBuilder":
        """设定背景上下文"""
        self._context = description
        return self

    def format(self, description: str) -> "PromptBuilder":
        """设定输出格式"""
        self._format = description
        return self

    def example(self, input_text: str, output_text: str, description: str = "") -> "PromptBuilder":
        """添加Few-shot示例"""
        self._examples.append(FewShotExample(
            input=input_text,
            output=output_text,
            description=description,
        ))
        return self

    def constraint(self, text: str) -> "PromptBuilder":
        """添加约束条件"""
        self._constraints.append(text)
        return self

    def build(self) -> PromptTemplate:
        """构建PromptTemplate"""
        return PromptTemplate(
            name=self._name,
            role=self._role,
            task=self._task,
            context=self._context,
            format_instructions=self._format,
            examples=self._examples,
            constraints=self._constraints,
        )

    def render(self, variables: Optional[Dict[str, str]] = None) -> str:
        """直接渲染为文本"""
        return self.build().render(variables)


# ============================================================
# 3. 预定义Prompt模板
# ============================================================

class PromptTemplates:
    """预定义Prompt模板库"""

    @staticmethod
    def rag_answer(question: str, context: str, style: str = "detailed") -> str:
        """RAG问答模板

        五要素：
        - Role: 知识库问答专家
        - Task: 根据文档回答问题
        - Context: 检索到的文档
        - Format: 结论→证据→引用
        - Examples: 标准问答示例
        """
        style_map = {
            "concise": "用1-3句话简洁回答，只给结论。",
            "detailed": "详细回答，包含背景、分析和结论。",
            "analytical": "分析式回答，先分析问题核心，再给出证据和结论。",
        }
        style_text = style_map.get(style, style_map["detailed"])

        builder = PromptBuilder("rag_answer")
        builder.role("你是知识库问答专家，擅长根据文档内容准确回答问题。")
        builder.task(f"根据以下文档回答用户问题。\n\n问题：{question}")
        builder.context(f"参考文档：\n{context}")
        builder.format(f"{style_text}\n\n每个关键事实标注来源：[来源: 文档名, 第X块]")
        builder.constraint("只基于文档内容回答，不要编造")
        builder.constraint("如果文档中没有相关信息，明确说明")
        builder.constraint("引用格式：[来源: 文件名]")
        builder.example(
            "什么是向量检索？",
            "向量检索是将文本转换为向量表示，通过计算向量相似度来搜索相关文档的技术。[来源: 检索技术文档]"
        )
        return builder.render()

    @staticmethod
    def fact_check(claim: str, evidence: str) -> str:
        """事实核查模板

        五要素：
        - Role: 事实核查专家
        - Task: 验证声明准确性
        - Context: 参考证据
        - Format: 结构化JSON报告
        - Examples: 核查示例
        """
        builder = PromptBuilder("fact_check")
        builder.role("你是事实核查专家，擅长验证信息的准确性。")
        builder.task(f"验证以下声明是否准确。\n\n待验证声明：{claim}")
        builder.context(f"参考证据：\n{evidence}")
        builder.format('JSON格式：{"verified": true/false, "confidence": 0-100, "issues": [...]}')
        builder.constraint("逐条验证声明中的事实点")
        builder.constraint("标注哪些是正确的、哪些是错误的")
        builder.constraint("给出可信度评分（0-100分）")
        return builder.render()

    @staticmethod
    def code_review(code: str, language: str = "python") -> str:
        """代码审查模板

        五要素：
        - Role: 资深代码审查专家
        - Task: 审查代码
        - Context: 待审查代码
        - Format: 问题列表+严重程度+修改建议
        - Examples: 审查示例
        """
        builder = PromptBuilder("code_review")
        builder.role(f"你是资深{language}工程师，负责代码审查。")
        builder.task(f"审查以下{language}代码。")
        builder.context(f"```{language}\n{code}\n```")
        builder.format("按维度逐项检查，输出问题列表、严重程度和修改建议。")
        builder.constraint("检查功能正确性、安全性、性能、可读性")
        builder.constraint("每个问题给出具体的修改建议")
        builder.example(
            "def divide(a, b): return a / b",
            "问题：未处理除零异常\n严重程度：高\n建议：添加 if b == 0: return None"
        )
        return builder.render()

    @staticmethod
    def summarize(text: str, max_length: int = 200) -> str:
        """文本摘要模板"""
        builder = PromptBuilder("summarize")
        builder.role("你是技术文档编辑，负责提炼文章精华。")
        builder.task(f"对以下文本进行摘要，不超过{max_length}字。")
        builder.context(text)
        builder.format("一句话结论 → 3个要点 → 适合人群")
        builder.constraint("保留关键信息，去除冗余")
        builder.constraint("使用简洁的语言")
        return builder.render()

    @staticmethod
    def translate(text: str, target_lang: str = "英文", domain: str = "") -> str:
        """翻译模板"""
        builder = PromptBuilder("translate")
        builder.role(f"你是专业翻译，擅长{domain if domain else '通用'}领域的翻译。")
        builder.task(f"将以下文本翻译成{target_lang}。")
        builder.context(text)
        builder.format("保持原文的段落结构和专业术语")
        builder.constraint("翻译准确，符合目标语言的表达习惯")
        builder.constraint("专业术语保持一致")
        return builder.render()

    @staticmethod
    def dual_agent_compare(question: str, answer_a: str, answer_b: str) -> str:
        """双Agent对比模板"""
        builder = PromptBuilder("dual_compare")
        builder.role("你是答案对比专家，擅长判断两个回答的一致性。")
        builder.task("对比以下两个回答，判断是否存在事实矛盾。")
        builder.context(f"问题：{question}\n\n回答A：{answer_a}\n\n回答B：{answer_b}")
        builder.format('JSON：{"verdict": "agree/conflict/partial", "conflict_points": [...]}')
        builder.constraint("只关注事实性矛盾，忽略表达差异")
        builder.constraint("列出具体的矛盾点")
        return builder.render()


# ============================================================
# 4. Prompt版本控制
# ============================================================

@dataclass
class PromptVersion:
    """Prompt版本记录"""
    name: str
    version: str
    template: PromptTemplate
    hash: str
    created_at: float
    performance: Dict[str, float] = field(default_factory=dict)


class PromptVersionControl:
    """Prompt版本控制

    支持：
    - 版本历史记录
    - 性能对比
    - A/B测试
    """

    def __init__(self):
        self._versions: Dict[str, List[PromptVersion]] = {}

    def save_version(self, template: PromptTemplate) -> str:
        """保存版本

        Args:
            template: Prompt模板

        Returns:
            版本哈希
        """
        import time

        # 计算哈希
        content = template.render()
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]

        version = PromptVersion(
            name=template.name,
            version=template.version,
            template=template,
            hash=hash_val,
            created_at=time.time(),
        )

        if template.name not in self._versions:
            self._versions[template.name] = []

        self._versions[template.name].append(version)
        log.info(f"[PromptVC] 保存版本: {template.name} v{template.version} ({hash_val})")

        return hash_val

    def get_latest(self, name: str) -> Optional[PromptVersion]:
        """获取最新版本"""
        versions = self._versions.get(name, [])
        return versions[-1] if versions else None

    def get_version(self, name: str, hash_val: str) -> Optional[PromptVersion]:
        """获取指定版本"""
        for v in self._versions.get(name, []):
            if v.hash == hash_val:
                return v
        return None

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """列出所有版本"""
        return [
            {
                "version": v.version,
                "hash": v.hash,
                "created_at": v.created_at,
                "performance": v.performance,
            }
            for v in self._versions.get(name, [])
        ]

    def record_performance(self, name: str, hash_val: str, metrics: Dict[str, float]):
        """记录版本性能

        Args:
            name: 模板名称
            hash_val: 版本哈希
            metrics: 性能指标（如 {"accuracy": 0.95, "latency": 1.2}）
        """
        version = self.get_version(name, hash_val)
        if version:
            version.performance.update(metrics)
            log.info(f"[PromptVC] 记录性能: {name} ({hash_val}) → {metrics}")

    def compare_versions(self, name: str, hash_a: str, hash_b: str) -> Dict[str, Any]:
        """对比两个版本的性能"""
        v_a = self.get_version(name, hash_a)
        v_b = self.get_version(name, hash_b)

        if not v_a or not v_b:
            return {"error": "版本不存在"}

        comparison = {
            "version_a": {"hash": hash_a, "performance": v_a.performance},
            "version_b": {"hash": hash_b, "performance": v_b.performance},
            "differences": {},
        }

        # 计算差异
        all_metrics = set(v_a.performance.keys()) | set(v_b.performance.keys())
        for metric in all_metrics:
            val_a = v_a.performance.get(metric, 0)
            val_b = v_b.performance.get(metric, 0)
            diff = val_b - val_a
            comparison["differences"][metric] = {
                "a": val_a,
                "b": val_b,
                "diff": round(diff, 4),
                "improved": diff > 0 if "error" not in metric.lower() else diff < 0,
            }

        return comparison


# ============================================================
# 5. 全局实例
# ============================================================

_version_control: Optional[PromptVersionControl] = None


def get_prompt_version_control() -> PromptVersionControl:
    """获取全局Prompt版本控制实例"""
    global _version_control
    if _version_control is None:
        _version_control = PromptVersionControl()
    return _version_control
