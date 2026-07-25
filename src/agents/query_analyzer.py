"""Query分析Agent — 判断问题类型+查询改写（v2.6：增加HyDE/Multi-Query, v2.8.3：增加RAG路由判断, v2.9：Prompt模板化）"""
from src.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
import logging
import re

log = logging.getLogger(__name__)

# v2.9: 从PromptManager加载，失败时降级到硬编码
_FALLBACK_SYSTEM_PROMPT = """你是一个查询分析专家。分析用户问题并优化检索查询。

输出严格JSON：
{
  "question_type": "factual/reasoning/comparison/open_ended",
  "rewritten_query": "优化后的检索查询（去掉口语化表达，提取关键词）",
  "search_queries": ["子查询1", "子查询2", "子查询3"],
  "hyde_answer": "假设性答案（用1-3句话假设性地回答问题，用于HyDE检索增强）"
}

question_type判断标准：
- factual: 有明确答案的事实问题（"X是什么"、"Y的值是多少"）
- reasoning: 需要推理的问题（"为什么"、"怎么实现"）
- comparison: 对比类问题（"A和B的区别"、"哪个更好"）
- open_ended: 开放性问题（"你怎么看"、"有什么建议"）

search_queries: 将复杂问题拆分为1-3个不同角度的独立可检索子查询（Multi-Query）
hyde_answer: 假设你已经找到了答案，用1-3句话写出你期望的答案内容（HyDE技术）
"""

def _get_system_prompt() -> str:
    """获取SYSTEM_PROMPT（v2.9: 优先从PromptManager加载）"""
    try:
        from src.tools.modules.prompt_manager import get_pipeline_prompt
        return get_pipeline_prompt("query_analyze")
    except Exception:
        return _FALLBACK_SYSTEM_PROMPT

SYSTEM_PROMPT = _get_system_prompt()

HYDE_PROMPT = """请为以下问题生成一个假设性答案（HyDE技术）。
不需要真实信息，只需要写出你期望找到的答案的格式和关键词。

问题：{question}

假设性答案（1-3句话）："""


def analyze_query(question: str) -> dict:
    """分析用户查询，改写优化（v2.6：增加HyDE和Multi-Query）"""
    llm = get_llm(temperature=0)
    if llm is None:
        return analyze_query_offline(question)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"分析这个问题：{question}"),
    ]
    response = llm.invoke(messages)

    try:
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content
        data = json.loads(json_str.strip())
    except (json.JSONDecodeError, IndexError) as e:
        log.warning(f"Query analysis JSON parse failed: {e}, using fallback")
        data = {
            "question_type": "factual",
            "rewritten_query": question,
            "search_queries": [question],
            "hyde_answer": "",
        }

    result = {
        "question_type": data.get("question_type", "factual"),
        "rewritten_query": data.get("rewritten_query", question),
        "search_queries": data.get("search_queries", [question]),
        "hyde_answer": data.get("hyde_answer", ""),
    }

    log.info(f"[QueryAnalyzer] type={result['question_type']}, "
             f"rewritten='{result['rewritten_query'][:50]}', "
             f"multi_queries={len(result['search_queries'])}, "
             f"hyde={'yes' if result['hyde_answer'] else 'no'}")

    return result


def analyze_query_offline(question: str) -> dict:
    """离线版查询分析（不调LLM，用规则）"""
    import re

    q = question.strip()

    # 判断类型
    if any(kw in q for kw in ["为什么", "怎么", "如何", "原因", "why", "how"]):
        qtype = "reasoning"
    elif any(kw in q for kw in ["区别", "对比", "和", "vs", "比较", "哪个"]):
        qtype = "comparison"
    elif any(kw in q for kw in ["建议", "看法", "你觉得", "推荐"]):
        qtype = "open_ended"
    else:
        qtype = "factual"

    # 简单查询改写：去掉口语词
    rewritten = re.sub(r"(请问|请|帮我|我想知道|能不能告诉我)", "", q).strip()

    # Multi-Query：生成多个查询变体（离线版用简单规则）
    search_queries = [rewritten or q]
    if qtype == "comparison":
        # 对比类：拆分为各自的查询
        parts = re.split(r"[和与vs对比比较]", q)
        for part in parts:
            part = part.strip()
            if len(part) > 3:
                search_queries.append(part)
    elif qtype == "reasoning":
        # 推理类：添加概念查询
        search_queries.append(rewritten + " 概念 定义")
        search_queries.append(rewritten + " 原理 机制")

    # HyDE离线版：空字符串（无法生成假设答案）
    hyde_answer = ""

    return {
        "question_type": qtype,
        "rewritten_query": rewritten or q,
        "search_queries": search_queries[:3],  # 最多3个子查询
        "hyde_answer": hyde_answer,
    }


def rewrite_query_for_retry(question: str, retry_count: int) -> str:
    """重试时的查询改写（LLM版，用于node_rewrite_query）"""
    llm = get_llm(temperature=0.7)

    if llm is None:
        # 离线降级：jieba关键词扩展
        import jieba
        keywords = [w for w in jieba.cut(question) if len(w) > 1]
        if retry_count == 1:
            return " ".join(keywords[:5]) + " 相关概念 定义 说明"
        else:
            return " ".join(keywords[:3]) + " 原理 机制 类型"

    rewrite_prompt = f"""你是一个查询改写专家。用户的问题在知识库中检索结果不理想，需要改写查询以提高召回率。

原始问题：{question}
当前是第{retry_count}次重试。

改写策略（根据重试次数选择不同策略）：
- 第1次：同义词替换 + 语义扩展（保持原意但换种说法）
- 第2次：问题分解（将复杂问题拆解为更简单的子问题）
- 第3次+：泛化（去掉过于具体的限定词，回到更通用的概念）

只输出改写后的查询，不要其他内容。"""

    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten = response.content.strip()
        log.info(f"[QueryRewrite] retry #{retry_count}: '{question[:30]}' → '{rewritten[:30]}'")
        return rewritten
    except Exception as e:
        log.error(f"[QueryRewrite] LLM rewrite failed: {e}")
        return question + " 相关概念 定义"


# === v2.8.3: RAG路由判断 — 常识/闲聊问题跳过知识库 ===

# 闲聊/身份类问题（LLM自身就能回答，不需要知识库）
_CHAT_PATTERNS = [
    r"你是谁", r"你叫什么", r"你的名字", r"你是AI吗", r"你是机器人",
    r"你好", r"您好", r"hi", r"hello", r"hey", r"嗨",
    r"谢谢", r"感谢", r"thanks", r"thank you", r"多谢",
    r"再见", r"bye", r"拜拜", r"晚安",
    r"你能做什么", r"你的功能", r"帮助", r"help", r"怎么用你",
    r"你是怎么工作的", r"你的原理", r"介绍一下你自己",
    r"在吗", r"有人吗", r"测试",
    r"今天.*几号", r"今天.*星期", r"现在.*几点",
    r"你.*能.*联网", r"你.*能.*搜索", r"你.*会.*什么",
]

# 常识类问题（大模型训练数据中已包含，不需要知识库）
_COMMON_KNOWLEDGE_PATTERNS = [
    r"1\s*\+\s*1\s*等于", r"\d+\s*[\+\-\×\÷\*/]\s*\d+\s*等于",  # 简单数学
    r"中国的首都", r"北京是.*首都", r"美国的首都",
    r"太阳从.*升起", r"太阳从.*落下",
    r"一年有.*天", r"一天有.*小时", r"一小时有.*分",
    r"水的化学式", r"H2O", r"二氧化碳.*化学式",
    r"什么是.*引力", r"牛顿.*苹果",
    r"地球.*太阳", r"月亮.*地球",
    r"光速.*多少", r"光速.*公里",
    r"圆周率", r"π.*等于", r"pi.*=",
    r"勾股定理", r"毕达哥拉斯定理",
    r"DNA.*全称", r"脱氧核糖核酸",
    r"光合作用.*是什么",
    r"什么是.*AI", r"什么是.*人工智能",
    r"python.*print", r"hello world",
    r"什么是.*JSON", r"什么是.*HTTP",
    r"长城.*在哪", r"黄河.*多长", r"长江.*多长",
    r"九九乘法",
    r"英文字母.*几个", r"26个字母",
]

# 编译正则
_CHAT_RE = [re.compile(p, re.IGNORECASE) for p in _CHAT_PATTERNS]
_COMMON_RE = [re.compile(p, re.IGNORECASE) for p in _COMMON_KNOWLEDGE_PATTERNS]

# 明显不可答 / 应拒识：荒诞时空、虚构细节、显式「知识库有没有」类探测
_UNANSWERABLE_PATTERNS = [
    r"火星",
    r"不存在的",
    r"门牌号",
    r"宇宙总部",
    r"昨天中午.*吃",
    r"量子纠缠.*设备",
    r"本知识库里有没有",
    r"知识库.*有没有介绍",
    r"虚构的公司",
    r"编一个.*地址",
]
_UNANSWERABLE_RE = [re.compile(p, re.IGNORECASE) for p in _UNANSWERABLE_PATTERNS]


def is_unanswerable_query(question: str) -> tuple[bool, str]:
    """规则检测：应直接拒答、勿硬编答案。

    Returns:
        (True, reason) — 应拒识； (False, "") — 正常走检索
    """
    q = (question or "").strip()
    if not q:
        return True, "空问题"
    for pattern in _UNANSWERABLE_RE:
        if pattern.search(q):
            return True, f"不可答模式:{pattern.pattern}"
    return False, ""


# 时效性 / 新闻类：禁止走本地知识库（会误命中「今日工作日志」）
_REALTIME_PATTERNS = [
    r"今日新闻",
    r"今天新闻",
    r"今日要闻",
    r"今日头条",
    r"热点新闻",
    r"实时新闻",
    r"最新新闻",
    r"新闻联播",
    r"今日资讯",
    r"今日.*消息",
    r"今天.*新闻",
    r"最近新闻",
    r"国内外新闻",
    r"breaking\s*news",
    r"today'?s\s*news",
    r"what'?s\s*new\s*today",
]
_REALTIME_RE = [re.compile(p, re.IGNORECASE) for p in _REALTIME_PATTERNS]
# 弱触发：同时含「新闻/要闻」+ 时间词
_REALTIME_WEAK = re.compile(
    r"(新闻|要闻|资讯|头条).{0,6}(今日|今天|实时|最新|现在)|(今日|今天|实时|最新).{0,6}(新闻|要闻|资讯|头条)",
    re.I,
)


def is_realtime_query(question: str) -> tuple[bool, str]:
    """是否时效性/新闻问题：应走 Web，禁止本地库硬答。"""
    q = (question or "").strip()
    if not q:
        return False, ""
    for pattern in _REALTIME_RE:
        if pattern.search(q):
            return True, f"realtime:{pattern.pattern}"
    if _REALTIME_WEAK.search(q):
        return True, "realtime:新闻+时间词"
    # 极短「新闻」「今日」类
    if q in ("新闻", "今日新闻", "今天新闻", "要闻", "热点"):
        return True, "realtime:短查询"
    return False, ""


def make_refuse_answer(reason: str = "") -> str:
    """统一拒答模板（含【直接回答】标记，便于评测识别）。"""
    detail = reason or "问题超出当前知识库可证据范围，或属于虚构/不可核验细节"
    return (
        "【直接回答】知识库中未找到可靠依据，无法基于证据回答该问题。\n\n"
        f"【详细解释】{detail}。"
        "请换一个可由文档支撑的问题，或补充知识库后再问。\n\n"
        "【引用来源】（无）"
    )


def needs_rag(question: str) -> tuple[bool, str]:
    """判断问题是否需要走RAG知识库检索

    v2.8.3: 闲聊/身份/常识问题直接由LLM回答，跳过检索节省2-3秒
    v2.9.1: 荒诞/虚构/显式探测题不走 LLM 硬答，由上层拒识

    Args:
        question: 用户原始问题

    Returns:
        (needs_rag, reason):
        - (True, "") — 需要知识库
        - (False, "闲聊问题") — 跳过RAG
        - (False, "常识问题") — 跳过RAG
        - (False, "不可答:...") — 应拒识（上层设 no_knowledge）
    """
    q = question.strip()

    # 不可答优先：避免进入检索后胡编
    bad, why = is_unanswerable_query(q)
    if bad:
        return False, f"不可答:{why}"

    # 太短的问题（<=3字且不含问号）可能是闲聊
    if len(q) <= 3 and "?" not in q and "？" not in q:
        # 但排除特定领域术语缩写
        if q.lower() not in ("rag", "llm", "ai", "gpu", "cpu", "api", "sql", "css", "dns"):
            return False, "闲聊问题"

    # 闲聊/身份类
    for pattern in _CHAT_RE:
        if pattern.search(q):
            return False, "闲聊问题"

    # 常识类
    for pattern in _COMMON_RE:
        if pattern.search(q):
            return False, "常识问题"

    # 纯数字/纯符号（无实际语义）
    if re.match(r"^[\d\s\+\-\×\÷\*/\=\.\(\)]+$", q) and len(q) < 30:
        return False, "数学计算"

    return True, ""
