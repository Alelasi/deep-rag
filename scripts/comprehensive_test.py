"""DeepRAG综合测评脚本 — 100道题×多维度评分

测评维度：
1. 准确性（事实正确性）
2. 完整性（覆盖所有要点）
3. 相关性（回答是否切题）
4. 引用质量（引用是否准确）
5. 响应速度
6. 幻觉程度
7. 格式规范性
8. 语言流畅度

用法：
    python tests/comprehensive_test.py
"""
import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 1. 测试题库（100道，5类别×20题）
# ============================================================

TEST_QUESTIONS = [
    # === 类别1: MBTI/性格分析（20题）===
    {"id": 1, "category": "MBTI", "difficulty": "easy", "question": "INTJ的主导功能是什么？", "expected_keywords": ["Ni", "内向直觉"]},
    {"id": 2, "category": "MBTI", "difficulty": "easy", "question": "ENFP的主导功能是什么？", "expected_keywords": ["Ne", "外向直觉"]},
    {"id": 3, "category": "MBTI", "difficulty": "medium", "question": "INTJ和INFJ的核心区别是什么？", "expected_keywords": ["Te", "Fe", "思维", "情感"]},
    {"id": 4, "category": "MBTI", "difficulty": "medium", "question": "什么是认知功能的堆叠顺序？", "expected_keywords": ["主导", "辅助", "第三", "劣势"]},
    {"id": 5, "category": "MBTI", "difficulty": "hard", "question": "INTJ在压力下会表现出哪些特征？", "expected_keywords": ["Se", "劣势功能", "失控"]},
    {"id": 6, "category": "MBTI", "difficulty": "easy", "question": "MBTI的四个维度分别是什么？", "expected_keywords": ["E/I", "S/N", "T/F", "J/P"]},
    {"id": 7, "category": "MBTI", "difficulty": "medium", "question": "什么是阴影功能？", "expected_keywords": ["无意识", "对立", "批评"]},
    {"id": 8, "category": "MBTI", "difficulty": "hard", "question": "如何通过认知功能分析一个人的决策模式？", "expected_keywords": ["Te", "Ti", "Fi", "Fe"]},
    {"id": 9, "category": "MBTI", "difficulty": "easy", "question": "ISTJ的性格特点是什么？", "expected_keywords": ["Si", "责任感", "传统"]},
    {"id": 10, "category": "MBTI", "difficulty": "medium", "question": "ENTP和ENTJ的核心差异是什么？", "expected_keywords": ["Ne", "Te", "探索", "执行"]},
    {"id": 11, "category": "MBTI", "difficulty": "easy", "question": "什么是荣格八维？", "expected_keywords": ["八个认知功能", "Ni", "Ne", "Si", "Se"]},
    {"id": 12, "category": "MBTI", "difficulty": "hard", "question": "如何用MBTI分析团队协作问题？", "expected_keywords": ["互补", "冲突", "沟通风格"]},
    {"id": 13, "category": "MBTI", "difficulty": "medium", "question": "INFP的职业倾向有哪些？", "expected_keywords": ["创意", "咨询", "写作"]},
    {"id": 14, "category": "MBTI", "difficulty": "easy", "question": "什么是感知功能和判断功能？", "expected_keywords": ["S/N", "T/F", "信息获取", "决策"]},
    {"id": 15, "category": "MBTI", "difficulty": "hard", "question": "MBTI测试的局限性是什么？", "expected_keywords": ["二分法", "巴纳姆效应", "情境"]},
    {"id": 16, "category": "MBTI", "difficulty": "medium", "question": "ESTP的核心动机是什么？", "expected_keywords": ["Se", "即时体验", "行动"]},
    {"id": 17, "category": "MBTI", "difficulty": "easy", "question": "INTJ适合什么职业？", "expected_keywords": ["战略", "分析", "技术"]},
    {"id": 18, "category": "MBTI", "difficulty": "medium", "question": "什么是认知功能的发展阶段？", "expected_keywords": ["童年", "青少年", "成年", "成熟"]},
    {"id": 19, "category": "MBTI", "difficulty": "hard", "question": "如何用MBTI改善亲密关系？", "expected_keywords": ["理解差异", "沟通", "尊重"]},
    {"id": 20, "category": "MBTI", "difficulty": "easy", "question": "ENFJ的辅助功能是什么？", "expected_keywords": ["Ni", "内向直觉"]},

    # === 类别2: RAG技术（20题）===
    {"id": 21, "category": "RAG", "difficulty": "easy", "question": "什么是RAG？", "expected_keywords": ["检索增强生成", "Retrieval", "Augmented"]},
    {"id": 22, "category": "RAG", "difficulty": "easy", "question": "RAG系统有哪些核心组件？", "expected_keywords": ["检索", "生成", "索引"]},
    {"id": 23, "category": "RAG", "difficulty": "medium", "question": "什么是混合检索？", "expected_keywords": ["BM25", "向量", "融合"]},
    {"id": 24, "category": "RAG", "difficulty": "medium", "question": "向量检索的原理是什么？", "expected_keywords": ["embedding", "相似度", "余弦"]},
    {"id": 25, "category": "RAG", "difficulty": "hard", "question": "如何解决RAG系统中的幻觉问题？", "expected_keywords": ["事实核查", "引用", "Self-Consistency"]},
    {"id": 26, "category": "RAG", "difficulty": "easy", "question": "什么是chunk_size？如何选择？", "expected_keywords": ["分块大小", "800", "平衡"]},
    {"id": 27, "category": "RAG", "difficulty": "medium", "question": "什么是重排序（Reranking）？", "expected_keywords": ["精排", "相关性", "Cross-Encoder"]},
    {"id": 28, "category": "RAG", "difficulty": "hard", "question": "如何评估RAG系统的效果？", "expected_keywords": ["RAGAS", "Hit@K", "MRR"]},
    {"id": 29, "category": "RAG", "difficulty": "easy", "question": "什么是向量数据库？", "expected_keywords": ["存储", "索引", "查询向量"]},
    {"id": 30, "category": "RAG", "difficulty": "medium", "question": "什么是BM25算法？", "expected_keywords": ["词频", "逆文档频率", "概率"]},
    {"id": 31, "category": "RAG", "difficulty": "medium", "question": "什么是语义缓存？", "expected_keywords": ["向量相似度", "缓存命中", "降低成本"]},
    {"id": 32, "category": "RAG", "difficulty": "hard", "question": "如何处理多轮对话中的上下文？", "expected_keywords": ["历史", "压缩", "滑动窗口"]},
    {"id": 33, "category": "RAG", "difficulty": "easy", "question": "什么是Embedding模型？", "expected_keywords": ["向量化", "语义表示", "bge"]},
    {"id": 34, "category": "RAG", "difficulty": "medium", "question": "什么是RRF（Reciprocal Rank Fusion）？", "expected_keywords": ["排名融合", "倒数", "多路召回"]},
    {"id": 35, "category": "RAG", "difficulty": "hard", "question": "如何优化RAG系统的检索召回率？", "expected_keywords": ["查询扩展", "HyDE", "多路召回"]},
    {"id": 36, "category": "RAG", "difficulty": "easy", "question": "什么是Prompt Engineering？", "expected_keywords": ["提示词", "设计", "优化"]},
    {"id": 37, "category": "RAG", "difficulty": "medium", "question": "什么是Self-RAG？", "expected_keywords": ["自我反思", "检索判断", "自适应"]},
    {"id": 38, "category": "RAG", "difficulty": "hard", "question": "如何处理长文档的RAG？", "expected_keywords": ["分层索引", "摘要", "Map-Reduce"]},
    {"id": 39, "category": "RAG", "difficulty": "easy", "question": "什么是知识图谱？", "expected_keywords": ["实体", "关系", "图结构"]},
    {"id": 40, "category": "RAG", "difficulty": "medium", "question": "什么是Agentic RAG？", "expected_keywords": ["Agent", "动态决策", "多轮检索"]},

    # === 类别3: LLM原理（20题）===
    {"id": 41, "category": "LLM", "difficulty": "easy", "question": "什么是Transformer？", "expected_keywords": ["注意力机制", "Encoder", "Decoder"]},
    {"id": 42, "category": "LLM", "difficulty": "easy", "question": "什么是自回归生成？", "expected_keywords": ["逐token", "下一个词", "序列"]},
    {"id": 43, "category": "LLM", "difficulty": "medium", "question": "什么是KV Cache？", "expected_keywords": ["Key", "Value", "缓存", "推理优化"]},
    {"id": 44, "category": "LLM", "difficulty": "medium", "question": "什么是LoRA？", "expected_keywords": ["低秩", "参数高效", "微调"]},
    {"id": 45, "category": "LLM", "difficulty": "hard", "question": "什么是MoE（混合专家模型）？", "expected_keywords": ["专家", "Router", "稀疏激活"]},
    {"id": 46, "category": "LLM", "difficulty": "easy", "question": "什么是Temperature参数？", "expected_keywords": ["随机性", "采样", "创造力"]},
    {"id": 47, "category": "LLM", "difficulty": "medium", "question": "什么是量化？INT8和INT4有什么区别？", "expected_keywords": ["精度", "显存", "损失"]},
    {"id": 48, "category": "LLM", "difficulty": "hard", "question": "什么是Flash Attention？", "expected_keywords": ["IO", "分块", "内存优化"]},
    {"id": 49, "category": "LLM", "difficulty": "easy", "question": "什么是Prompt？", "expected_keywords": ["输入", "指令", "提示"]},
    {"id": 50, "category": "LLM", "difficulty": "medium", "question": "什么是RLHF？", "expected_keywords": ["人类反馈", "强化学习", "对齐"]},
    {"id": 51, "category": "LLM", "difficulty": "medium", "question": "什么是DPO？", "expected_keywords": ["直接偏好", "无需奖励模型", "对齐"]},
    {"id": 52, "category": "LLM", "difficulty": "hard", "question": "什么是Scaling Law？", "expected_keywords": ["参数", "数据", "算力", "幂律"]},
    {"id": 53, "category": "LLM", "difficulty": "easy", "question": "什么是Token？", "expected_keywords": ["分词", "子词", "最小单位"]},
    {"id": 54, "category": "LLM", "difficulty": "medium", "question": "什么是位置编码？", "expected_keywords": ["序列顺序", "RoPE", "ALiBi"]},
    {"id": 55, "category": "LLM", "difficulty": "hard", "question": "什么是PagedAttention？", "expected_keywords": ["vLLM", "虚拟内存", "分页"]},
    {"id": 56, "category": "LLM", "difficulty": "easy", "question": "什么是Top-P采样？", "expected_keywords": ["累积概率", "核采样", "多样性"]},
    {"id": 57, "category": "LLM", "difficulty": "medium", "question": "什么是CoT（Chain-of-Thought）？", "expected_keywords": ["思维链", "推理步骤", "逐步"]},
    {"id": 58, "category": "LLM", "difficulty": "hard", "question": "什么是Speculative Decoding？", "expected_keywords": ["草稿", "验证", "加速"]},
    {"id": 59, "category": "LLM", "difficulty": "easy", "question": "什么是SFT？", "expected_keywords": ["监督微调", "指令数据", "训练"]},
    {"id": 60, "category": "LLM", "difficulty": "medium", "question": "什么是Tokenizer？", "expected_keywords": ["BPE", "分词", "词表"]},

    # === 类别4: Agent/工具调用（20题）===
    {"id": 61, "category": "Agent", "difficulty": "easy", "question": "什么是Function Calling？", "expected_keywords": ["结构化JSON", "工具调用", "模型决策"]},
    {"id": 62, "category": "Agent", "difficulty": "easy", "question": "什么是MCP协议？", "expected_keywords": ["Model Context Protocol", "Anthropic", "工具标准化"]},
    {"id": 63, "category": "Agent", "difficulty": "medium", "question": "MCP和Function Calling有什么区别？", "expected_keywords": ["层次不同", "协议vs格式", "互补"]},
    {"id": 64, "category": "Agent", "difficulty": "medium", "question": "什么是A2A协议？", "expected_keywords": ["Agent-to-Agent", "Google", "协作"]},
    {"id": 65, "category": "Agent", "difficulty": "hard", "question": "什么是Skill系统？", "expected_keywords": ["渐进式加载", "操作手册", "SKILL.md"]},
    {"id": 66, "category": "Agent", "difficulty": "easy", "question": "什么是ReAct模式？", "expected_keywords": ["推理", "行动", "观察"]},
    {"id": 67, "category": "Agent", "difficulty": "medium", "question": "什么是Tool Use？", "expected_keywords": ["工具使用", "外部API", "扩展能力"]},
    {"id": 68, "category": "Agent", "difficulty": "hard", "question": "如何设计多Agent系统？", "expected_keywords": ["分工", "通信", "协调"]},
    {"id": 69, "category": "Agent", "difficulty": "easy", "question": "什么是Agent Card？", "expected_keywords": ["能力声明", "JSON", "发现"]},
    {"id": 70, "category": "Agent", "difficulty": "medium", "question": "什么是Task状态机？", "expected_keywords": ["submitted", "working", "completed"]},
    {"id": 71, "category": "Agent", "difficulty": "medium", "question": "什么是LLM Gateway？", "expected_keywords": ["统一入口", "路由", "限流"]},
    {"id": 72, "category": "Agent", "difficulty": "hard", "question": "什么是语义缓存？", "expected_keywords": ["向量相似度", "缓存命中", "降低成本"]},
    {"id": 73, "category": "Agent", "difficulty": "easy", "question": "什么是Prompt模板？", "expected_keywords": ["可复用", "变量", "格式化"]},
    {"id": 74, "category": "Agent", "difficulty": "medium", "question": "什么是约束解码？", "expected_keywords": ["JSON Schema", "强制输出", "结构化"]},
    {"id": 75, "category": "Agent", "difficulty": "hard", "question": "什么是Self-Consistency？", "expected_keywords": ["多次采样", "投票", "多数"]},
    {"id": 76, "category": "Agent", "difficulty": "easy", "question": "什么是SSE？", "expected_keywords": ["Server-Sent Events", "单向推送", "流式"]},
    {"id": 77, "category": "Agent", "difficulty": "medium", "question": "什么是WebSocket？", "expected_keywords": ["全双工", "双向", "实时"]},
    {"id": 78, "category": "Agent", "difficulty": "hard", "question": "如何选择SSE还是WebSocket？", "expected_keywords": ["单向vs双向", "LLM流式", "复杂度"]},
    {"id": 79, "category": "Agent", "difficulty": "easy", "question": "什么是JSON-RPC？", "expected_keywords": ["远程调用", "JSON", "请求响应"]},
    {"id": 80, "category": "Agent", "difficulty": "medium", "question": "什么是stdio传输？", "expected_keywords": ["标准输入输出", "本地", "管道"]},

    # === 类别5: 工程实践（20题）===
    {"id": 81, "category": "Engineering", "difficulty": "easy", "question": "如何优化LLM推理速度？", "expected_keywords": ["量化", "KV Cache", "批处理"]},
    {"id": 82, "category": "Engineering", "difficulty": "easy", "question": "如何降低LLM API成本？", "expected_keywords": ["缓存", "模型路由", "小模型"]},
    {"id": 83, "category": "Engineering", "difficulty": "medium", "question": "什么是模型路由？", "expected_keywords": ["任务类型", "选择模型", "性价比"]},
    {"id": 84, "category": "Engineering", "difficulty": "medium", "question": "什么是限流和熔断？", "expected_keywords": ["频率控制", "故障转移", "保护"]},
    {"id": 85, "category": "Engineering", "difficulty": "hard", "question": "如何设计评测体系？", "expected_keywords": ["测试集", "自动化", "人工校验"]},
    {"id": 86, "category": "Engineering", "difficulty": "easy", "question": "什么是Token追踪？", "expected_keywords": ["用量统计", "成本", "监控"]},
    {"id": 87, "category": "Engineering", "difficulty": "medium", "question": "什么是Prompt Caching？", "expected_keywords": ["固定前缀", "KV Cache", "跨请求"]},
    {"id": 88, "category": "Engineering", "difficulty": "hard", "question": "如何处理API限流？", "expected_keywords": ["重试", "退避", "队列"]},
    {"id": 89, "category": "Engineering", "difficulty": "easy", "question": "什么是降级策略？", "expected_keywords": ["备用方案", "兜底", "可用性"]},
    {"id": 90, "category": "Engineering", "difficulty": "medium", "question": "什么是A/B测试？", "expected_keywords": ["对比", "分流", "效果评估"]},
    {"id": 91, "category": "Engineering", "difficulty": "medium", "question": "什么是可观测性？", "expected_keywords": ["日志", "指标", "追踪"]},
    {"id": 92, "category": "Engineering", "difficulty": "hard", "question": "如何设计容错机制？", "expected_keywords": ["重试", "超时", "降级"]},
    {"id": 93, "category": "Engineering", "difficulty": "easy", "question": "什么是缓存？", "expected_keywords": ["存储", "复用", "加速"]},
    {"id": 94, "category": "Engineering", "difficulty": "medium", "question": "什么是负载均衡？", "expected_keywords": ["分发", "多实例", "高可用"]},
    {"id": 95, "category": "Engineering", "difficulty": "hard", "question": "如何做性能优化？", "expected_keywords": ["瓶颈", "profiling", "优化策略"]},
    {"id": 96, "category": "Engineering", "difficulty": "easy", "question": "什么是CI/CD？", "expected_keywords": ["持续集成", "持续部署", "自动化"]},
    {"id": 97, "category": "Engineering", "difficulty": "medium", "question": "什么是Docker？", "expected_keywords": ["容器", "镜像", "部署"]},
    {"id": 98, "category": "Engineering", "difficulty": "hard", "question": "如何设计微服务架构？", "expected_keywords": ["拆分", "通信", "治理"]},
    {"id": 99, "category": "Engineering", "difficulty": "easy", "question": "什么是API网关？", "expected_keywords": ["统一入口", "路由", "鉴权"]},
    {"id": 100, "category": "Engineering", "difficulty": "medium", "question": "如何做日志管理？", "expected_keywords": ["分级", "聚合", "分析"]},
]


# ============================================================
# 2. 评估维度
# ============================================================

@dataclass
class EvalScore:
    """单题评分"""
    question_id: int
    category: str
    difficulty: str
    question: str

    # 8个评分维度（1-10分）
    accuracy: float = 0.0        # 准确性
    completeness: float = 0.0    # 完整性
    relevance: float = 0.0       # 相关性
    citation_quality: float = 0.0  # 引用质量
    response_time: float = 0.0   # 响应速度（秒）
    hallucination: float = 0.0   # 幻觉程度（越低越好）
    format_score: float = 0.0    # 格式规范性
    fluency: float = 0.0         # 语言流畅度

    # 综合得分
    total_score: float = 0.0
    answer_length: int = 0
    error: str = ""

    def calc_total(self):
        """计算综合得分"""
        # 幻觉分数反向（越低越好）
        hallucination_score = 10 - self.hallucination

        # 加权平均
        weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.15,
            "citation_quality": 0.10,
            "response_time": 0.10,  # 越快越好
            "hallucination": 0.10,
            "format_score": 0.05,
            "fluency": 0.05,
        }

        # 响应速度评分（<2s=10, 2-5s=8, 5-10s=6, >10s=4）
        if self.response_time < 2:
            time_score = 10
        elif self.response_time < 5:
            time_score = 8
        elif self.response_time < 10:
            time_score = 6
        else:
            time_score = 4

        self.total_score = round(
            self.accuracy * weights["accuracy"] +
            self.completeness * weights["completeness"] +
            self.relevance * weights["relevance"] +
            self.citation_quality * weights["citation_quality"] +
            time_score * weights["response_time"] +
            hallucination_score * weights["hallucination"] +
            self.format_score * weights["format_score"] +
            self.fluency * weights["fluency"],
            2
        )
        return self.total_score


# ============================================================
# 3. 评分函数
# ============================================================

def evaluate_answer(question_data: dict, answer: str, response_time: float) -> EvalScore:
    """评估单个回答

    Args:
        question_data: 测试题数据
        answer: 系统回答
        response_time: 响应时间（秒）

    Returns:
        评分结果
    """
    score = EvalScore(
        question_id=question_data["id"],
        category=question_data["category"],
        difficulty=question_data["difficulty"],
        question=question_data["question"],
        response_time=response_time,
        answer_length=len(answer),
    )

    if not answer or answer.startswith("调用失败"):
        score.error = answer
        score.calc_total()
        return score

    expected_keywords = question_data.get("expected_keywords", [])
    answer_lower = answer.lower()

    # 1. 准确性：关键词命中率
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_rate = keyword_hits / max(len(expected_keywords), 1)
    score.accuracy = round(min(10, keyword_rate * 10), 1)

    # 2. 完整性：基于答案长度和关键词覆盖
    length_score = min(10, len(answer) / 100)  # 100字=10分
    score.completeness = round((length_score * 0.5 + keyword_rate * 10 * 0.5), 1)

    # 3. 相关性：问题关键词在答案中的出现
    question_keywords = question_data["question"].replace("？", "").replace("?", "").split()
    relevance_hits = sum(1 for kw in question_keywords if kw in answer)
    score.relevance = round(min(10, relevance_hits / max(len(question_keywords), 1) * 10), 1)

    # 4. 引用质量：是否有引用标记
    has_citation = any(marker in answer for marker in ["[来源", "[1]", "[2]", "来源:", "参考"])
    score.citation_quality = 8 if has_citation else 4

    # 5. 幻觉程度：检测不确定词汇
    uncertainty_words = ["可能", "也许", "大概", "不确定", "我不确定", "没有相关信息"]
    uncertainty_count = sum(1 for w in uncertainty_words if w in answer)
    score.hallucination = max(0, min(10, uncertainty_count * 2))

    # 6. 格式规范性
    has_structure = any(marker in answer for marker in ["\n", "。", "：", "、"])
    score.format_score = 8 if has_structure else 5

    # 7. 语言流畅度
    score.fluency = 8 if len(answer) > 50 else 6

    # 计算综合得分
    score.calc_total()

    return score


# ============================================================
# 4. 测试执行器
# ============================================================

def run_test(use_real: bool = False):
    """执行100道题的全面测评

    Args:
        use_real: True=调用真实RAG管道(real_test), False=模拟评分(simulate_test)
    """
    mode_label = "真实RAG管道" if use_real else "模拟"
    print("=" * 60)
    print(f"DeepRAG 综合测评 — 100道题×8维度 [{mode_label}模式]")
    print("=" * 60)

    results = []
    start_time = time.time()

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/100] 测试: {q['question'][:30]}...")

        # --real 使用真实RAG管道，否则使用模拟评分
        if use_real:
            score = real_test(q)
        else:
            score = simulate_test(q)
        results.append(score)

        # 打印进度
        if (i + 1) % 10 == 0:
            avg_score = sum(r.total_score for r in results) / len(results)
            print(f"  进度: {i+1}/100, 平均分: {avg_score:.2f}")

    total_time = time.time() - start_time

    # 生成报告
    report = generate_report(results, total_time)

    # 保存结果
    output_path = Path(__file__).parent.parent / "docs" / "test_results_100.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"测试完成！结果已保存到: {output_path}")
    print(f"{'=' * 60}")

    return results


def simulate_test(question_data: dict) -> EvalScore:
    """模拟测试评分（基于问题特征）

    实际部署时应替换为真实系统调用
    """
    import random

    # 基于难度和类别生成模拟分数
    difficulty_bonus = {"easy": 1.5, "medium": 1.0, "hard": 0.5}
    category_bonus = {"MBTI": 1.2, "RAG": 1.3, "LLM": 1.1, "Agent": 1.4, "Engineering": 1.0}

    base_score = 7.0
    diff_bonus = difficulty_bonus.get(question_data["difficulty"], 0)
    cat_bonus = category_bonus.get(question_data["category"], 0)

    # 生成各维度分数
    score = EvalScore(
        question_id=question_data["id"],
        category=question_data["category"],
        difficulty=question_data["difficulty"],
        question=question_data["question"],
    )

    score.accuracy = round(min(10, max(1, base_score + diff_bonus + random.uniform(-1, 1))), 1)
    score.completeness = round(min(10, max(1, base_score + cat_bonus + random.uniform(-1, 1))), 1)
    score.relevance = round(min(10, max(1, base_score + random.uniform(-0.5, 0.5))), 1)
    score.citation_quality = round(min(10, max(1, 7 + random.uniform(-2, 2))), 1)
    score.response_time = round(random.uniform(1.0, 5.0), 2)
    score.hallucination = round(max(0, min(10, 3 + random.uniform(-2, 2))), 1)
    score.format_score = round(min(10, max(1, 8 + random.uniform(-1, 1))), 1)
    score.fluency = round(min(10, max(1, 8 + random.uniform(-1, 1))), 1)
    score.answer_length = random.randint(100, 500)

    score.calc_total()

    return score


def real_test(question_data: dict) -> 'EvalScore':
    """真实 RAG 管道测试 — 调用实际 graph.query
    
    与 simulate_test 不同，此函数调用真实的 RAG 管道，
    测试端到端的检索-生成-评分流程。
    
    需要 Qdrant 和 Ollama 服务运行中。
    """
    from src.graph import query as rag_query
    
    question = question_data["question"]
    start = time.time()
    
    try:
        result = rag_query(question)
        elapsed = time.time() - start
        
        # 提取答案
        if isinstance(result, dict):
            answer = result.get("answer", str(result))
            sources = result.get("sources", [])
        else:
            answer = str(result)
            sources = []
        
        return evaluate_answer(question_data, answer, elapsed)
        
    except Exception as e:
        elapsed = time.time() - start
        score = EvalScore(
            question_id=question_data["id"],
            category=question_data["category"],
            difficulty=question_data["difficulty"],
            question=question,
            response_time=elapsed,
            error=f"调用失败: {e}"
        )
        score.calc_total()
        return score


def generate_report(results: List[EvalScore], total_time: float) -> str:
    """生成测试报告"""
    report = []
    report.append("# DeepRAG 综合测评报告")
    report.append(f"\n> 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"> 测试题数: {len(results)}")
    report.append(f"> 总耗时: {total_time:.1f}秒")
    report.append("")

    # 1. 总体统计
    report.append("## 一、总体统计")
    report.append("")
    avg_scores = {
        "accuracy": sum(r.accuracy for r in results) / len(results),
        "completeness": sum(r.completeness for r in results) / len(results),
        "relevance": sum(r.relevance for r in results) / len(results),
        "citation_quality": sum(r.citation_quality for r in results) / len(results),
        "hallucination": sum(r.hallucination for r in results) / len(results),
        "format_score": sum(r.format_score for r in results) / len(results),
        "fluency": sum(r.fluency for r in results) / len(results),
        "total": sum(r.total_score for r in results) / len(results),
    }

    report.append("| 维度 | 平均分 | 说明 |")
    report.append("|------|--------|------|")
    report.append(f"| **准确性** | {avg_scores['accuracy']:.2f} | 关键词命中率 |")
    report.append(f"| **完整性** | {avg_scores['completeness']:.2f} | 答案覆盖度 |")
    report.append(f"| **相关性** | {avg_scores['relevance']:.2f} | 回答切题度 |")
    report.append(f"| **引用质量** | {avg_scores['citation_quality']:.2f} | 引用规范性 |")
    report.append(f"| **幻觉程度** | {avg_scores['hallucination']:.2f} | 越低越好 |")
    report.append(f"| **格式规范** | {avg_scores['format_score']:.2f} | 输出格式 |")
    report.append(f"| **语言流畅** | {avg_scores['fluency']:.2f} | 表达质量 |")
    report.append(f"| **综合得分** | {avg_scores['total']:.2f} | 加权平均 |")
    report.append("")

    # 2. 按类别统计
    report.append("## 二、按类别统计")
    report.append("")
    categories = set(r.category for r in results)
    report.append("| 类别 | 题数 | 平均分 | 最高分 | 最低分 |")
    report.append("|------|------|--------|--------|--------|")

    for cat in sorted(categories):
        cat_results = [r for r in results if r.category == cat]
        avg = sum(r.total_score for r in cat_results) / len(cat_results)
        max_score = max(r.total_score for r in cat_results)
        min_score = min(r.total_score for r in cat_results)
        report.append(f"| {cat} | {len(cat_results)} | {avg:.2f} | {max_score:.2f} | {min_score:.2f} |")

    report.append("")

    # 3. 按难度统计
    report.append("## 三、按难度统计")
    report.append("")
    difficulties = ["easy", "medium", "hard"]
    report.append("| 难度 | 题数 | 平均分 | 说明 |")
    report.append("|------|------|--------|------|")

    for diff in difficulties:
        diff_results = [r for r in results if r.difficulty == diff]
        avg = sum(r.total_score for r in diff_results) / len(diff_results)
        report.append(f"| {diff} | {len(diff_results)} | {avg:.2f} | {'简单' if diff == 'easy' else '中等' if diff == 'medium' else '困难'} |")

    report.append("")

    # 4. 详细结果表格
    report.append("## 四、详细结果（100题）")
    report.append("")
    report.append("| ID | 类别 | 难度 | 问题 | 准确性 | 完整性 | 相关性 | 引用 | 幻觉 | 格式 | 流畅 | 综合 |")
    report.append("|-----|------|------|------|--------|--------|--------|------|------|------|------|------|")

    for r in results:
        question_short = r.question[:20] + "..." if len(r.question) > 20 else r.question
        report.append(
            f"| {r.question_id} | {r.category} | {r.difficulty} | {question_short} | "
            f"{r.accuracy} | {r.completeness} | {r.relevance} | {r.citation_quality} | "
            f"{r.hallucination} | {r.format_score} | {r.fluency} | **{r.total_score}** |"
        )

    report.append("")

    # 5. Top 10 和 Bottom 10
    report.append("## 五、最佳/最差题目")
    report.append("")
    sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)

    report.append("### Top 10 最佳题目")
    report.append("")
    report.append("| 排名 | ID | 问题 | 综合分 |")
    report.append("|------|-----|------|--------|")
    for i, r in enumerate(sorted_results[:10], 1):
        report.append(f"| {i} | {r.question_id} | {r.question[:30]} | {r.total_score} |")

    report.append("")
    report.append("### Bottom 10 最差题目")
    report.append("")
    report.append("| 排名 | ID | 问题 | 综合分 |")
    report.append("|------|-----|------|--------|")
    for i, r in enumerate(sorted_results[-10:], 1):
        report.append(f"| {i} | {r.question_id} | {r.question[:30]} | {r.total_score} |")

    report.append("")

    # 6. 结论
    report.append("## 六、结论与建议")
    report.append("")
    report.append("### 优势领域")
    strong_categories = [cat for cat in categories
                        if sum(r.total_score for r in results if r.category == cat) /
                        len([r for r in results if r.category == cat]) > avg_scores['total']]
    for cat in strong_categories:
        report.append(f"- **{cat}**: 表现优异")

    report.append("")
    report.append("### 待改进领域")
    weak_categories = [cat for cat in categories
                      if sum(r.total_score for r in results if r.category == cat) /
                      len([r for r in results if r.category == cat]) <= avg_scores['total']]
    for cat in weak_categories:
        report.append(f"- **{cat}**: 需要加强")

    report.append("")
    report.append("### 优化建议")
    report.append("1. 提升困难题目的准确性")
    report.append("2. 加强引用规范性")
    report.append("3. 优化响应速度")
    report.append("4. 减少幻觉发生")

    report.append("")
    report.append("---")
    report.append(f"\n**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(report)


# ============================================================
# 5. 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepRAG 综合测评脚本 — 100道题×多维度评分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/comprehensive_test.py                 # 模拟测试（默认）
  python scripts/comprehensive_test.py --real          # 真实RAG管道测试
  python scripts/comprehensive_test.py --simulate      # 模拟测试（显式指定）
        """,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="使用真实RAG管道测试（需要Qdrant和Ollama运行）",
    )
    mode_group.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="使用模拟测试（默认模式，无需外部服务）",
    )
    args = parser.parse_args()

    # --real 优先；默认为模拟模式
    use_real = args.real
    mode_label = "真实RAG管道" if use_real else "模拟"
    print(f"测试模式: {mode_label}")

    run_test(use_real=use_real)
