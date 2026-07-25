"""免费 LLM 模型全面评测系统
测试维度：响应质量、速度、稳定性、中文能力、逻辑推理等 30+ 个指标
"""
import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# API Keys
# 密钥只从环境变量读取
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


# ==================== 测试问题（10个，覆盖不同场景）====================

TEST_QUESTIONS = [
    {
        "id": 1,
        "category": "知识问答",
        "question": "什么是RAG（检索增强生成）？它相比纯LLM有什么优势？",
        "expected_keywords": ["检索", "增强", "生成", "知识库", "减少幻觉", "外部知识"],
        "eval_criteria": "准确性、完整性、专业术语使用"
    },
    {
        "id": 2,
        "category": "逻辑推理",
        "question": "小明有5个苹果，给了小红2个，又买了3个，最后小明有几个苹果？请一步步推理。",
        "expected_keywords": ["5", "2", "3", "6"],
        "eval_criteria": "推理过程清晰、计算正确"
    },
    {
        "id": 3,
        "category": "代码生成",
        "question": "用Python写一个函数，判断一个数是否是素数，并添加类型注解和文档字符串。",
        "expected_keywords": ["def", "is_prime", "int", "bool", "return", "for", "range"],
        "eval_criteria": "代码正确性、类型注解、文档字符串、代码风格"
    },
    {
        "id": 4,
        "category": "中文理解",
        "question": "解释成语'画蛇添足'的含义，并举一个生活中的例子。",
        "expected_keywords": ["多余", "做多余的事", "反而", "不好"],
        "eval_criteria": "解释准确、例子恰当、表达流畅"
    },
    {
        "id": 5,
        "category": "创意写作",
        "question": "用50字左右写一句广告语，推广一款智能学习助手。",
        "expected_keywords": ["学习", "智能", "助手"],
        "eval_criteria": "创意性、吸引力、简洁性、符合字数要求"
    },
    {
        "id": 6,
        "category": "数学能力",
        "question": "求解方程：2x + 5 = 17，请给出详细步骤。",
        "expected_keywords": ["2x", "12", "6", "x=6"],
        "eval_criteria": "解题步骤完整、计算正确"
    },
    {
        "id": 7,
        "category": "信息提取",
        "question": "从以下文本中提取人名、地点和时间：'2025年3月15日，张三在北京参加了人工智能大会。'\n请用JSON格式输出。",
        "expected_keywords": ["张三", "北京", "2025", "3月15日", "JSON"],
        "eval_criteria": "提取完整、格式正确"
    },
    {
        "id": 8,
        "category": "比较分析",
        "question": "比较 Python 和 JavaScript 的主要区别，至少列出5点。",
        "expected_keywords": ["类型", "语法", "用途", "运行", "库"],
        "eval_criteria": "比较全面、观点准确、条理清晰"
    },
    {
        "id": 9,
        "category": "问题解决",
        "question": "我的Python程序报错'IndentationError: unexpected indent'，怎么解决？",
        "expected_keywords": ["缩进", "空格", "Tab", "混用", "检查"],
        "eval_criteria": "诊断准确、解决方案可行"
    },
    {
        "id": 10,
        "category": "多轮对话理解",
        "question": "用户：我想学习机器学习，应该从哪里开始？\n（假设这是对话的第一轮，请给出建议）",
        "expected_keywords": ["基础", "数学", "Python", "课程", "实践"],
        "eval_criteria": "建议实用、结构清晰、针对性强"
    },
]


# ==================== 评估维度（30个）====================

EVAL_DIMENSIONS = {
    # 内容质量（10个）
    "accuracy": {"name": "准确性", "weight": 3, "desc": "事实正确、无错误信息"},
    "completeness": {"name": "完整性", "weight": 2, "desc": "覆盖问题的所有方面"},
    "relevance": {"name": "相关性", "weight": 2, "desc": "回答与问题高度相关"},
    "depth": {"name": "深度", "weight": 2, "desc": "有足够细节和深入分析"},
    "clarity": {"name": "清晰度", "weight": 2, "desc": "表达清楚、易于理解"},
    "logical": {"name": "逻辑性", "weight": 2, "desc": "推理过程合理、有条理"},
    "professional": {"name": "专业性", "weight": 2, "desc": "使用正确的专业术语"},
    "creativity": {"name": "创意性", "weight": 1, "desc": "有独特见解或创新表达"},
    "practicality": {"name": "实用性", "weight": 2, "desc": "建议或方案可操作"},
    "examples": {"name": "示例质量", "weight": 1, "desc": "举例恰当、有说明力"},

    # 格式规范（5个）
    "format": {"name": "格式规范", "weight": 2, "desc": "结构清晰、使用标题/列表"},
    "code_quality": {"name": "代码质量", "weight": 2, "desc": "代码正确、风格规范"},
    "json_format": {"name": "JSON格式", "weight": 1, "desc": "JSON格式正确可解析"},
    "markdown": {"name": "Markdown使用", "weight": 1, "desc": "合理使用Markdown标记"},
    "length": {"name": "长度适当", "weight": 1, "desc": "回答长度适中，不冗余不遗漏"},

    # 语言能力（5个）
    "chinese_fluency": {"name": "中文流畅度", "weight": 2, "desc": "中文表达自然、无翻译腔"},
    "grammar": {"name": "语法正确", "weight": 2, "desc": "无错别字、语病"},
    "terminology": {"name": "术语准确", "weight": 2, "desc": "专业术语使用正确"},
    "expression": {"name": "表达能力", "weight": 1, "desc": "用词精准、表达生动"},
    "bilingual": {"name": "中英混排", "weight": 1, "desc": "中英文切换自然"},

    # 响应特性（5个）
    "latency": {"name": "响应延迟", "weight": 3, "desc": "首token延迟低"},
    "speed": {"name": "生成速度", "weight": 2, "desc": "每秒token数量"},
    "stability": {"name": "稳定性", "weight": 2, "desc": "10次调用成功率"},
    "consistency": {"name": "一致性", "weight": 1, "desc": "多次回答质量稳定"},
    "no_hallucination": {"name": "无幻觉", "weight": 3, "desc": "不编造虚假信息"},

    # 特殊能力（5个）
    "instruction_follow": {"name": "指令遵循", "weight": 2, "desc": "严格按照要求执行"},
    "context_understand": {"name": "上下文理解", "weight": 2, "desc": "理解问题背景和意图"},
    "step_by_step": {"name": "逐步推理", "weight": 2, "desc": "能分步骤解答"},
    "error_handle": {"name": "错误处理", "weight": 1, "desc": "识别问题中的陷阱"},
    "summarize": {"name": "总结能力", "weight": 1, "desc": "能概括要点"},
}


# ==================== 模型配置 ====================

MODELS_TO_TEST = [
    # Groq 模型
    {
        "provider": "Groq",
        "model_id": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "display_name": "Groq GPT-OSS-120B"
    },
    {
        "provider": "Groq",
        "model_id": "openai/gpt-oss-20b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "display_name": "Groq GPT-OSS-20B"
    },
    {
        "provider": "Groq",
        "model_id": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "display_name": "Groq Llama3.3-70B"
    },
    {
        "provider": "Groq",
        "model_id": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "display_name": "Groq Llama3.1-8B"
    },
    {
        "provider": "Groq",
        "model_id": "qwen/qwen3-32b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "display_name": "Groq Qwen3-32B"
    },
    # Cerebras 模型
    {
        "provider": "Cerebras",
        "model_id": "gpt-oss-120b",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": CEREBRAS_API_KEY,
        "display_name": "Cerebras GPT-OSS-120B"
    },
    {
        "provider": "Cerebras",
        "model_id": "gemma-4-31b",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": CEREBRAS_API_KEY,
        "display_name": "Cerebras Gemma4-31B"
    },
    # OpenRouter 免费模型
    {
        "provider": "OpenRouter",
        "model_id": "openai/gpt-oss-20b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "display_name": "OpenRouter GPT-OSS-20B"
    },
]


@dataclass
class QuestionResult:
    """单个问题的测试结果"""
    question_id: int
    category: str
    response: str
    latency_ms: float
    tokens: int
    tokens_per_second: float
    success: bool
    error: str = ""
    scores: Dict[str, float] = None


@dataclass
class ModelBenchmark:
    """模型评测结果"""
    model_name: str
    provider: str
    model_id: str
    total_questions: int
    success_count: int
    avg_latency_ms: float
    avg_speed: float
    question_results: List[QuestionResult]
    dimension_scores: Dict[str, float] = None
    total_score: float = 0.0
    rank: int = 0


def call_llm(base_url: str, api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """调用 LLM API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1000,
        "stream": False,
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60,
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            # 估算token数（中文约1.5字/token，英文约4字符/token）
            tokens = len(content) * 1.5 if any('一' <= c <= '鿿' for c in content) else len(content) / 4
            speed = tokens / (latency_ms / 1000) if latency_ms > 0 else 0

            return {
                "success": True,
                "content": content,
                "latency_ms": latency_ms,
                "tokens": int(tokens),
                "speed": round(speed, 1),
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:100]}",
                "latency_ms": latency_ms,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:100],
            "latency_ms": (time.time() - start_time) * 1000,
        }


def evaluate_response(question: Dict, response: str, latency_ms: float, speed: float) -> Dict[str, float]:
    """评估回答质量（基于规则的自动评分）"""
    scores = {}

    # 内容质量评分（基于关键词匹配）
    expected = question.get("expected_keywords", [])
    if expected:
        matched = sum(1 for kw in expected if kw in response)
        keyword_score = min(matched / len(expected) * 10, 10)
    else:
        keyword_score = 5  # 默认中等分数

    # 基于规则的评分
    response_len = len(response)

    # 1. 准确性（基于关键词）
    scores["accuracy"] = keyword_score

    # 2. 完整性（基于长度和关键词覆盖）
    completeness = min(keyword_score * 0.8 + (response_len / 200) * 2, 10)
    scores["completeness"] = round(completeness, 1)

    # 3. 相关性（检查是否包含问题关键词）
    question_words = set(question["question"].replace("？", "").replace("，", "").split())
    relevance = min(sum(1 for w in question_words if w in response) / max(len(question_words), 1) * 15, 10)
    scores["relevance"] = round(relevance, 1)

    # 4. 深度（基于详细程度）
    depth_indicators = ["因为", "所以", "首先", "其次", "例如", "比如", "具体来说", "总结"]
    depth_score = min(sum(1 for d in depth_indicators if d in response) * 2 + 3, 10)
    scores["depth"] = round(depth_score, 1)

    # 5. 清晰度（基于结构化程度）
    clarity_indicators = ["\n", "1.", "2.", "3.", "-", "•", "：", "。"]
    clarity_score = min(sum(1 for c in clarity_indicators if c in response) * 1.5 + 3, 10)
    scores["clarity"] = round(clarity_score, 1)

    # 6. 逻辑性（基于连接词）
    logic_words = ["首先", "然后", "接着", "最后", "因此", "所以", "但是", "然而", "另外"]
    logic_score = min(sum(1 for w in logic_words if w in response) * 2 + 3, 10)
    scores["logical"] = round(logic_score, 1)

    # 7. 专业性（基于术语）
    if question["category"] in ["知识问答", "代码生成"]:
        scores["professional"] = min(keyword_score * 1.1, 10)
    else:
        scores["professional"] = min(keyword_score * 0.9, 10)

    # 8. 创意性（基于独特表达）
    creativity_indicators = ["创新", "独特", "新颖", "想象", "创意"]
    scores["creativity"] = min(sum(1 for c in creativity_indicators if c in response) * 3 + 2, 10)

    # 9. 实用性（基于可操作建议）
    practical_indicators = ["可以", "建议", "推荐", "使用", "方法", "步骤"]
    scores["practicality"] = min(sum(1 for p in practical_indicators if p in response) * 1.5 + 3, 10)

    # 10. 示例质量
    example_indicators = ["例如", "比如", "举例", "像这样", "示例"]
    scores["examples"] = min(sum(1 for e in example_indicators if e in response) * 2.5 + 2, 10)

    # 格式规范
    scores["format"] = min(clarity_score * 1.2, 10)

    # 代码质量（仅代码题）
    if question["category"] == "代码生成":
        code_indicators = ["def ", "return", "if ", "for ", "#", "\"\"\""]
        scores["code_quality"] = min(sum(1 for c in code_indicators if c in response) * 2, 10)
    else:
        scores["code_quality"] = 5  # 非代码题给默认分

    # JSON格式（仅信息提取题）
    if question["category"] == "信息提取":
        try:
            # 检查是否包含有效的JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json.loads(response[json_start:json_end])
                scores["json_format"] = 9
            else:
                scores["json_format"] = 3
        except:
            scores["json_format"] = 3
    else:
        scores["json_format"] = 5

    # Markdown使用
    md_indicators = ["```", "**", "##", "- ", "> ", "|"]
    scores["markdown"] = min(sum(1 for m in md_indicators if m in response) * 2 + 2, 10)

    # 长度适当
    if 50 < response_len < 2000:
        scores["length"] = 8
    elif response_len <= 50:
        scores["length"] = 4
    else:
        scores["length"] = 6

    # 中文流畅度
    chinese_chars = sum(1 for c in response if '一' <= c <= '鿿')
    if chinese_chars > 20:
        scores["chinese_fluency"] = 8
    elif chinese_chars > 5:
        scores["chinese_fluency"] = 6
    else:
        scores["chinese_fluency"] = 3

    # 语法正确（简单检查）
    grammar_errors = ["的的", "了了", "是是", "有有"]
    grammar_score = 10 - sum(1 for g in grammar_errors if g in response) * 3
    scores["grammar"] = max(grammar_score, 3)

    # 术语准确
    scores["terminology"] = min(keyword_score * 1.05, 10)

    # 表达能力
    scores["expression"] = min(scores["clarity"] * 1.1, 10)

    # 中英混排
    has_english = any(c.isascii() and c.isalpha() for c in response)
    has_chinese = any('一' <= c <= '鿿' for c in response)
    if has_english and has_chinese:
        scores["bilingual"] = 7
    else:
        scores["bilingual"] = 5

    # 响应延迟（越低越好）
    if latency_ms < 500:
        scores["latency"] = 10
    elif latency_ms < 1000:
        scores["latency"] = 8
    elif latency_ms < 2000:
        scores["latency"] = 6
    elif latency_ms < 5000:
        scores["latency"] = 4
    else:
        scores["latency"] = 2

    # 生成速度
    if speed > 100:
        scores["speed"] = 10
    elif speed > 50:
        scores["speed"] = 8
    elif speed > 20:
        scores["speed"] = 6
    elif speed > 10:
        scores["speed"] = 4
    else:
        scores["speed"] = 2

    # 稳定性（单独计算）
    scores["stability"] = 8  # 默认给8分，后面会根据成功率调整

    # 一致性（默认）
    scores["consistency"] = 7

    # 无幻觉（基于常识检查）
    hallucination_indicators = ["根据最新数据", "2026年", "最新研究显示", "据统计"]
    hallucination_score = 8 - sum(1 for h in hallucination_indicators if h in response) * 2
    scores["no_hallucination"] = max(hallucination_score, 3)

    # 指令遵循
    if "请用一句话" in question["question"] and response_len < 200:
        scores["instruction_follow"] = 9
    elif "列出" in question["question"] and any(c in response for c in ["1.", "2.", "-"]):
        scores["instruction_follow"] = 9
    else:
        scores["instruction_follow"] = 7

    # 上下文理解
    scores["context_understand"] = min(scores["relevance"] * 1.1, 10)

    # 逐步推理
    step_indicators = ["第一步", "首先", "1.", "步骤一", "然后"]
    scores["step_by_step"] = min(sum(1 for s in step_indicators if s in response) * 2.5 + 2, 10)

    # 错误处理
    scores["error_handle"] = 5  # 默认

    # 总结能力
    summary_indicators = ["总结", "综上", "总的来说", "综上所述", "概括"]
    scores["summarize"] = min(sum(1 for s in summary_indicators if s in response) * 3 + 2, 10)

    # 所有分数限制在0-10
    for key in scores:
        scores[key] = round(min(max(scores[key], 0), 10), 1)

    return scores


def test_model(model_config: Dict) -> ModelBenchmark:
    """测试单个模型的所有问题"""
    print(f"\n{'='*80}")
    print(f"测试模型: {model_config['display_name']}")
    print(f"提供商: {model_config['provider']}")
    print(f"模型ID: {model_config['model_id']}")
    print(f"{'='*80}")

    question_results = []
    success_count = 0
    total_latency = 0
    total_speed = 0

    for q in TEST_QUESTIONS:
        print(f"\n  [{q['id']}/10] {q['category']}: {q['question'][:40]}...")

        result = call_llm(
            model_config["base_url"],
            model_config["api_key"],
            model_config["model_id"],
            q["question"]
        )

        if result["success"]:
            success_count += 1
            scores = evaluate_response(q, result["content"], result["latency_ms"], result["speed"])
            avg_score = sum(scores.values()) / len(scores)

            qr = QuestionResult(
                question_id=q["id"],
                category=q["category"],
                response=result["content"][:500],  # 截断保存
                latency_ms=result["latency_ms"],
                tokens=result["tokens"],
                tokens_per_second=result["speed"],
                success=True,
                scores=scores,
            )
            total_latency += result["latency_ms"]
            total_speed += result["speed"]

            print(f"    ✅ {result['latency_ms']:.0f}ms | {result['speed']:.0f} tok/s | 平均分: {avg_score:.1f}/10")
        else:
            qr = QuestionResult(
                question_id=q["id"],
                category=q["category"],
                response="",
                latency_ms=result.get("latency_ms", 0),
                tokens=0,
                tokens_per_second=0,
                success=False,
                error=result["error"],
            )
            print(f"    ❌ {result['error'][:50]}")

        question_results.append(qr)
        time.sleep(0.5)  # 避免限流

    # 计算平均值
    avg_latency = total_latency / success_count if success_count > 0 else 0
    avg_speed = total_speed / success_count if success_count > 0 else 0

    # 计算各维度平均分
    dimension_scores = {}
    if success_count > 0:
        for dim in EVAL_DIMENSIONS:
            dim_values = [qr.scores[dim] for qr in question_results if qr.success and qr.scores and dim in qr.scores]
            if dim_values:
                dimension_scores[dim] = round(sum(dim_values) / len(dim_values), 1)

    # 更新稳定性分数
    stability_score = (success_count / 10) * 10
    if "stability" in dimension_scores:
        dimension_scores["stability"] = stability_score

    # 计算加权总分
    total_score = 0
    total_weight = 0
    for dim, config in EVAL_DIMENSIONS.items():
        if dim in dimension_scores:
            total_score += dimension_scores[dim] * config["weight"]
            total_weight += config["weight"]

    weighted_total = total_score / total_weight if total_weight > 0 else 0

    benchmark = ModelBenchmark(
        model_name=model_config["display_name"],
        provider=model_config["provider"],
        model_id=model_config["model_id"],
        total_questions=10,
        success_count=success_count,
        avg_latency_ms=round(avg_latency, 2),
        avg_speed=round(avg_speed, 1),
        question_results=question_results,
        dimension_scores=dimension_scores,
        total_score=round(weighted_total, 2),
    )

    print(f"\n  📊 模型总结: {success_count}/10 成功 | 平均延迟 {avg_latency:.0f}ms | 加权总分 {weighted_total:.1f}/10")

    return benchmark


def generate_report(results: List[ModelBenchmark]) -> str:
    """生成评测报告"""
    report = []
    report.append("# 免费 LLM 模型全面评测报告")
    report.append(f"\n**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试问题**: 10 个（覆盖知识问答、逻辑推理、代码生成等场景）")
    report.append(f"**评估维度**: {len(EVAL_DIMENSIONS)} 个")
    report.append(f"**测试模型**: {len(results)} 个\n")

    # 按总分排名
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    # ==================== 综合排名 ====================
    report.append("## 📊 综合排名\n")
    report.append("| 排名 | 模型 | 提供商 | 加权总分 | 成功率 | 平均延迟 | 平均速度 |")
    report.append("|------|------|--------|----------|--------|----------|----------|")
    for r in results:
        report.append(f"| {r.rank} | {r.model_name} | {r.provider} | **{r.total_score:.1f}** | {r.success_count}/10 | {r.avg_latency_ms:.0f}ms | {r.avg_speed:.0f} tok/s |")

    # ==================== 分类排名 ====================
    report.append("\n## 📈 分类维度排名\n")

    categories = {
        "内容质量": ["accuracy", "completeness", "relevance", "depth", "clarity", "logical", "professional", "creativity", "practicality", "examples"],
        "格式规范": ["format", "code_quality", "json_format", "markdown", "length"],
        "语言能力": ["chinese_fluency", "grammar", "terminology", "expression", "bilingual"],
        "响应特性": ["latency", "speed", "stability", "consistency", "no_hallucination"],
        "特殊能力": ["instruction_follow", "context_understand", "step_by_step", "error_handle", "summarize"],
    }

    for cat_name, dims in categories.items():
        report.append(f"### {cat_name}\n")
        report.append("| 排名 | 模型 | 平均分 | " + " | ".join([EVAL_DIMENSIONS[d]["name"] for d in dims[:5]]) + " |")
        report.append("|------|------|--------| " + " | ".join(["-----" for _ in dims[:5]]) + " |")

        # 计算分类平均分
        cat_scores = []
        for r in results:
            if r.dimension_scores:
                scores = [r.dimension_scores.get(d, 0) for d in dims]
                avg = sum(scores) / len(scores) if scores else 0
                cat_scores.append((r, avg))

        cat_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (r, avg) in enumerate(cat_scores):
            dim_values = [f"{r.dimension_scores.get(d, 0):.1f}" for d in dims[:5]]
            report.append(f"| {i+1} | {r.model_name} | {avg:.1f} | " + " | ".join(dim_values) + " |")

        report.append("")

    # ==================== 各模型详细分析 ====================
    report.append("\n## 🔍 各模型详细分析\n")

    for r in results:
        report.append(f"### {r.rank}. {r.model_name} ({r.provider})\n")
        report.append(f"- **总分**: {r.total_score:.1f}/10")
        report.append(f"- **成功率**: {r.success_count}/10")
        report.append(f"- **平均延迟**: {r.avg_latency_ms:.0f}ms")
        report.append(f"- **平均速度**: {r.avg_speed:.0f} tok/s\n")

        # Top 5 优势维度
        if r.dimension_scores:
            sorted_dims = sorted(r.dimension_scores.items(), key=lambda x: x[1], reverse=True)
            report.append("**优势维度**:")
            for dim, score in sorted_dims[:5]:
                report.append(f"- {EVAL_DIMENSIONS[dim]['name']}: {score:.1f}/10")

            report.append("\n**待改进维度**:")
            for dim, score in sorted_dims[-3:]:
                report.append(f"- {EVAL_DIMENSIONS[dim]['name']}: {score:.1f}/10")

        # 问题分类表现
        report.append("\n**各问题表现**:")
        for qr in r.question_results:
            if qr.success and qr.scores:
                avg = sum(qr.scores.values()) / len(qr.scores)
                report.append(f"- Q{qr.question_id} ({qr.category}): {avg:.1f}分 | {qr.latency_ms:.0f}ms")
            else:
                report.append(f"- Q{qr.question_id} ({qr.category}): ❌ 失败")

        report.append("")

    # ==================== 推荐方案 ====================
    report.append("\n## 🏆 推荐方案\n")

    if results:
        best = results[0]
        report.append(f"### 默认推荐: **{best.model_name}**\n")
        report.append(f"- 总分最高: {best.total_score:.1f}/10")
        report.append(f"- 延迟: {best.avg_latency_ms:.0f}ms")
        report.append(f"- 速度: {best.avg_speed:.0f} tok/s\n")

        # 按场景推荐
        report.append("### 场景推荐\n")
        report.append("| 场景 | 推荐模型 | 理由 |")
        report.append("|------|----------|------|")

        # 最快模型
        fastest = min(results, key=lambda x: x.avg_latency_ms)
        report.append(f"| 极速响应 | {fastest.model_name} | {fastest.avg_latency_ms:.0f}ms 最低延迟 |")

        # 最稳定
        most_stable = max(results, key=lambda x: x.success_count)
        report.append(f"| 高稳定性 | {most_stable.model_name} | {most_stable.success_count}/10 成功率 |")

        # 中文最佳
        chinese_best = max(results, key=lambda x: x.dimension_scores.get("chinese_fluency", 0) if x.dimension_scores else 0)
        report.append(f"| 中文问答 | {chinese_best.model_name} | 中文流畅度 {chinese_best.dimension_scores.get('chinese_fluency', 0):.1f}/10 |")

        # 代码最佳
        code_best = max(results, key=lambda x: x.dimension_scores.get("code_quality", 0) if x.dimension_scores else 0)
        report.append(f"| 代码生成 | {code_best.model_name} | 代码质量 {code_best.dimension_scores.get('code_quality', 0):.1f}/10 |")

    return "\n".join(report)


def main():
    """主函数"""
    print("="*80)
    print("免费 LLM 模型全面评测系统")
    print(f"测试问题: {len(TEST_QUESTIONS)} 个")
    print(f"评估维度: {len(EVAL_DIMENSIONS)} 个")
    print(f"测试模型: {len(MODELS_TO_TEST)} 个")
    print("="*80)

    results = []

    for model_config in MODELS_TO_TEST:
        try:
            benchmark = test_model(model_config)
            results.append(benchmark)
        except Exception as e:
            print(f"\n❌ 模型 {model_config['display_name']} 测试失败: {e}")

    # 生成报告
    print("\n" + "="*80)
    print("生成评测报告...")
    print("="*80)

    report = generate_report(results)

    # 保存报告
    report_path = os.path.join(os.path.dirname(__file__), "..", "docs", "free_model_benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存JSON数据
    json_path = os.path.join(os.path.dirname(__file__), "..", "docs", "free_model_benchmark_data.json")
    json_data = {
        "test_time": datetime.now().isoformat(),
        "models": [asdict(r) for r in results],
        "dimensions": EVAL_DIMENSIONS,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 报告已保存:")
    print(f"   Markdown: {report_path}")
    print(f"   JSON: {json_path}")

    # 打印简要结果
    print("\n" + "="*80)
    print("📊 综合排名")
    print("="*80)
    results.sort(key=lambda x: x.total_score, reverse=True)
    print(f"\n{'排名':<6} {'模型':<30} {'总分':<10} {'成功率':<10} {'延迟':<10}")
    print("-"*70)
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r.model_name:<30} {r.total_score:.1f}/10  {r.success_count}/10     {r.avg_latency_ms:.0f}ms")


if __name__ == "__main__":
    main()
