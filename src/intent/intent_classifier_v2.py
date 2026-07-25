"""
意图识别器（增强版）- 集成本地 LLM
规则层 + LLM 兜底，目标准确率 ≥90%
"""

import re
import httpx
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class IntentL1(Enum):
    """一级意图"""
    KNOWLEDGE_QUERY = "知识查询"
    REALTIME_QUERY = "实时查询"
    MIXED_QUERY = "知识查询+实时查询"
    REFUSAL = "引导/拒答"


class IntentL2(Enum):
    """二级意图"""
    # 知识查询
    CONCEPT = "概念解释"
    API_USAGE = "API用法"
    CODE_EXAMPLE = "代码示例"
    BEST_PRACTICE = "最佳实践"
    ERROR_DEBUG = "错误排查"
    VERSION_COMPARE = "版本对比"
    ARCHITECTURE = "架构设计"
    PERFORMANCE = "性能优化"

    # 实时查询
    LATEST_VERSION = "最新版本"
    ISSUE_STATUS = "Issue状态"
    PACKAGE_DEP = "包依赖"
    COMMUNITY = "社区讨论"
    DOC_UPDATE = "文档更新"

    # 引导/拒答
    OUT_OF_SCOPE = "超出范围"
    AMBIGUOUS = "歧义澄清"


@dataclass
class IntentResult:
    """意图识别结果"""
    intent_l1: IntentL1
    intent_l2: IntentL2
    confidence: float
    route_decision: str
    reason: str


class IntentClassifier:
    """
    意图识别器（增强版）
    规则层（快速过滤） + LLM层（精准分类）
    """

    def __init__(self, use_llm: bool = True, llm_threshold: float = 0.7):
        """
        参数：
            use_llm: 是否使用 LLM 兜底
            llm_threshold: 低于此置信度时使用 LLM
        """
        self.use_llm = use_llm
        self.llm_threshold = llm_threshold
        self.llm_base_url = "http://localhost:11434/v1/chat/completions"

        # 规则：一级意图关键词（优化后）
        self.l1_keywords = {
            IntentL1.KNOWLEDGE_QUERY: [
                "是什么", "怎么用", "如何", "怎样", "区别", "对比",
                "实现", "设计", "原理", "为什么", "有什么", "怎么做",
                "怎么", "什么", "哪些", "能不能", "可以", "如何使用",
                "怎么设置", "怎么配置", "怎么实现"
            ],
            IntentL1.REALTIME_QUERY: [
                "最新", "现在", "当前", "最近", "新版本", "版本号",
                "issue", "#", "修了", "修复了", "更新了", "有没有更新"
            ],
            IntentL1.REFUSAL: [
                "训练gpt", "破解", "盗版", "写作业", "代写", "帮我写"
            ]
        }

        # 规则：二级意图关键词（优化后）
        self.l2_keywords = {
            # 知识查询
            IntentL2.CONCEPT: ["是什么", "定义", "含义", "解释", "理解", "vs", "和...的区别"],
            IntentL2.API_USAGE: [
                "怎么用", "如何使用", "用法", "调用", "接口",
                "怎么设置", "怎么配置", "怎么做", "如何集成", "如何配置"
            ],
            IntentL2.CODE_EXAMPLE: ["示例", "例子", "demo", "完整代码", "sample", "完整示例"],
            IntentL2.BEST_PRACTICE: ["最佳实践", "生产", "部署", "坑", "注意"],
            IntentL2.ERROR_DEBUG: ["报错", "错误", "失败", "不工作", "bug", "不生效"],
            IntentL2.VERSION_COMPARE: ["对比", "版本", "差异", "0.2", "0.3"],
            IntentL2.ARCHITECTURE: ["设计", "架构", "系统", "流程"],
            IntentL2.PERFORMANCE: ["优化", "提升", "加速", "慢", "性能", "速度"],

            # 实时查询
            IntentL2.LATEST_VERSION: ["最新版本", "版本号", "当前版本", "最新"],
            IntentL2.ISSUE_STATUS: ["issue", "#", "修了", "修复"],
            IntentL2.COMMUNITY: ["社区", "讨论", "有人", "大家"],

            # 引导/拒答
            IntentL2.OUT_OF_SCOPE: ["训练gpt", "破解", "盗版", "写作业"],
            IntentL2.AMBIGUOUS: []
        }

    def classify(self, query: str) -> IntentResult:
        """
        意图识别主函数

        流程：
        1. 规则层快速过滤
        2. 置信度 ≥ llm_threshold → 直接返回
        3. 置信度 < llm_threshold → LLM 分类
        """
        # Step 1: 规则层识别
        l1_intent, l1_confidence, l1_reason = self._rule_based_l1(query)
        l2_intent, l2_confidence, l2_reason = self._rule_based_l2(query, l1_intent)

        confidence = min(l1_confidence, l2_confidence)

        # Step 2: 如果置信度高，直接返回
        if confidence >= self.llm_threshold or not self.use_llm:
            route_decision = self._decide_route(l1_intent, l2_intent)
            return IntentResult(
                intent_l1=l1_intent,
                intent_l2=l2_intent,
                confidence=confidence,
                route_decision=route_decision,
                reason=f"规则匹配: {l1_reason} | {l2_reason}"
            )

        # Step 3: 低置信度 → LLM 分类
        try:
            llm_l1, llm_l2, llm_confidence = self._llm_classify(query)
            route_decision = self._decide_route(llm_l1, llm_l2)
            return IntentResult(
                intent_l1=llm_l1,
                intent_l2=llm_l2,
                confidence=llm_confidence,
                route_decision=route_decision,
                reason=f"LLM分类（规则置信度低: {confidence:.2f}）"
            )
        except Exception as e:
            # LLM 失败，回退到规则结果
            route_decision = self._decide_route(l1_intent, l2_intent)
            return IntentResult(
                intent_l1=l1_intent,
                intent_l2=l2_intent,
                confidence=confidence,
                route_decision=route_decision,
                reason=f"规则兜底（LLM失败: {str(e)}）"
            )

    def _llm_classify(self, query: str) -> Tuple[IntentL1, IntentL2, float]:
        """使用本地 LLM 进行意图分类"""

        prompt = f"""你是一个意图识别专家。请对用户问题进行分类。

用户问题：{query}

请分类为：

一级意图（4选1）：
- 知识查询：询问概念、用法、实现等知识性问题
- 实时查询：询问最新版本、Issue状态等实时信息
- 知识查询+实时查询：同时包含知识和实时查询
- 引导/拒答：超出范围或需要澄清

二级意图（根据一级意图选择）：
知识查询：概念解释、API用法、代码示例、最佳实践、错误排查、版本对比、架构设计、性能优化
实时查询：最新版本、Issue状态、包依赖、社区讨论、文档更新
引导/拒答：超出范围、歧义澄清

请只返回JSON格式：
{{"intent_l1": "知识查询", "intent_l2": "API用法", "confidence": 0.95}}"""

        try:
            client = httpx.Client(timeout=30.0)
            response = client.post(
                self.llm_base_url,
                json={
                    "model": "google/gemma-4-e2b",  # LM Studio加载的模型
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,  # 增加token让模型有足够空间输出JSON
                    "temperature": 0.1
                }
            )

            if response.status_code != 200:
                raise Exception(f"LLM API 返回错误: {response.status_code}")

            result = response.json()
            message = result["choices"][0]["message"]

            # LM Studio的gemma-4模型返回内容在reasoning_content里
            content = message.get("content", "")
            if not content or content.strip() == "":
                content = message.get("reasoning_content", "")

            if not content or content.strip() == "":
                raise Exception(f"LLM返回空内容")

            # 解析 JSON
            # 提取 JSON 部分（可能被包裹在其他文本中）
            json_match = re.search(r'\{.*?"intent_l1".*?\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise Exception(f"无法解析 LLM 输出: {content[:200]}")

            # 转换为枚举
            l1_str = data["intent_l1"]
            l2_str = data["intent_l2"]
            confidence = float(data.get("confidence", 0.8))

            # 映射到枚举
            l1_map = {
                "知识查询": IntentL1.KNOWLEDGE_QUERY,
                "实时查询": IntentL1.REALTIME_QUERY,
                "知识查询+实时查询": IntentL1.MIXED_QUERY,
                "引导/拒答": IntentL1.REFUSAL
            }

            l2_map = {
                "概念解释": IntentL2.CONCEPT,
                "API用法": IntentL2.API_USAGE,
                "代码示例": IntentL2.CODE_EXAMPLE,
                "最佳实践": IntentL2.BEST_PRACTICE,
                "错误排查": IntentL2.ERROR_DEBUG,
                "版本对比": IntentL2.VERSION_COMPARE,
                "架构设计": IntentL2.ARCHITECTURE,
                "性能优化": IntentL2.PERFORMANCE,
                "最新版本": IntentL2.LATEST_VERSION,
                "Issue状态": IntentL2.ISSUE_STATUS,
                "包依赖": IntentL2.PACKAGE_DEP,
                "社区讨论": IntentL2.COMMUNITY,
                "文档更新": IntentL2.DOC_UPDATE,
                "超出范围": IntentL2.OUT_OF_SCOPE,
                "歧义澄清": IntentL2.AMBIGUOUS
            }

            l1 = l1_map.get(l1_str, IntentL1.KNOWLEDGE_QUERY)
            l2 = l2_map.get(l2_str, IntentL2.API_USAGE)

            return l1, l2, confidence

        except Exception as e:
            raise Exception(f"LLM分类失败: {str(e)}")

    def _rule_based_l1(self, query: str) -> Tuple[IntentL1, float, str]:
        """规则层：一级意图识别（优化版）"""
        query_lower = query.lower()

        # 检查拒答（最高优先级）
        for kw in self.l1_keywords[IntentL1.REFUSAL]:
            if kw in query_lower:
                return IntentL1.REFUSAL, 0.95, f"包含拒答关键词: {kw}"

        # 检查是否同时包含知识查询和实时查询关键词
        kb_matches = [kw for kw in self.l1_keywords[IntentL1.KNOWLEDGE_QUERY] if kw in query]
        realtime_matches = [kw for kw in self.l1_keywords[IntentL1.REALTIME_QUERY] if kw in query]

        has_kb = len(kb_matches) > 0
        has_realtime = len(realtime_matches) > 0

        if has_kb and has_realtime:
            return IntentL1.MIXED_QUERY, 0.9, f"同时包含KB和实时关键词"

        # 检查实时查询（优先级高于知识查询）
        if has_realtime:
            return IntentL1.REALTIME_QUERY, 0.85, f"实时关键词: {realtime_matches}"

        # 检查知识查询
        if has_kb:
            confidence = 0.85 if len(kb_matches) >= 2 else 0.75
            return IntentL1.KNOWLEDGE_QUERY, confidence, f"KB关键词: {kb_matches}"

        # 默认知识查询（疑问句）
        if "?" in query or "？" in query:
            return IntentL1.KNOWLEDGE_QUERY, 0.6, "默认分类（疑问句）"

        return IntentL1.KNOWLEDGE_QUERY, 0.5, "默认分类（无明显关键词）"

    def _rule_based_l2(self, query: str, l1_intent: IntentL1) -> Tuple[IntentL2, float, str]:
        """规则层：二级意图识别（优化版）"""
        query_lower = query.lower()

        # 根据一级意图筛选候选
        if l1_intent == IntentL1.KNOWLEDGE_QUERY:
            candidates_priority = [
                IntentL2.ERROR_DEBUG,
                IntentL2.VERSION_COMPARE,
                IntentL2.CODE_EXAMPLE,
                IntentL2.API_USAGE,
                IntentL2.CONCEPT,
                IntentL2.ARCHITECTURE,
                IntentL2.PERFORMANCE,
                IntentL2.BEST_PRACTICE,
            ]
        elif l1_intent == IntentL1.REALTIME_QUERY:
            candidates_priority = [
                IntentL2.LATEST_VERSION,
                IntentL2.ISSUE_STATUS,
                IntentL2.COMMUNITY,
                IntentL2.PACKAGE_DEP,
                IntentL2.DOC_UPDATE
            ]
        elif l1_intent == IntentL1.REFUSAL:
            candidates_priority = [IntentL2.OUT_OF_SCOPE, IntentL2.AMBIGUOUS]
        else:  # MIXED_QUERY
            return IntentL2.CODE_EXAMPLE, 0.7, "混合查询默认"

        # 按优先级匹配
        for intent_l2 in candidates_priority:
            keywords = self.l2_keywords.get(intent_l2, [])
            matched_kws = [kw for kw in keywords if kw in query]
            if matched_kws:
                confidence = 0.9 if len(matched_kws) >= 2 else 0.8
                return intent_l2, confidence, f"匹配: {matched_kws}"

        # 默认：根据问句类型
        if "怎么" in query or "如何" in query:
            return IntentL2.API_USAGE, 0.65, "默认（如何类）"
        if "是什么" in query or "什么是" in query:
            return IntentL2.CONCEPT, 0.65, "默认（概念类）"

        return candidates_priority[0], 0.6, "默认分类"

    def _decide_route(self, l1_intent: IntentL1, l2_intent: IntentL2) -> str:
        """决策路由"""
        if l1_intent == IntentL1.REFUSAL:
            return "clarify" if l2_intent == IntentL2.AMBIGUOUS else "refuse"
        if l1_intent == IntentL1.REALTIME_QUERY:
            return "tool_only"
        if l1_intent == IntentL1.MIXED_QUERY:
            return "kb_and_tool"
        return "kb_only"


# ==================== 测试代码 ====================

if __name__ == "__main__":
    classifier = IntentClassifier(use_llm=True, llm_threshold=0.7)

    test_queries = [
        "LangChain 是什么？",
        "ChatOpenAI 怎么用？",
        "LangChain 最新版本是多少？",
        "如何使用 ChatPromptTemplate？",
        "为什么报 ImportError？",
    ]

    print("=" * 80)
    print("意图识别测试（规则 + LLM）")
    print("=" * 80)

    for query in test_queries:
        result = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"  L1: {result.intent_l1.value}")
        print(f"  L2: {result.intent_l2.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Route: {result.route_decision}")
        print(f"  Reason: {result.reason}")
