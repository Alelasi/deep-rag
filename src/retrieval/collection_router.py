"""查询域 → 知识库路由，避免「论文库答 INTJ」类错库问题

策略：
1. 规则关键词匹配（零 LLM、毫秒级）
2. 返回推荐 collection 列表（按优先级）
3. 可选：检测用户所选库是否与问题域冲突
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# 域 → 关键词（小写匹配前会 lower；中文保持原样）
DOMAIN_KEYWORDS = {
    "psychology": [
        "mbti", "intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
        "istj", "isfj", "estj", "esfj", "istp", "isfp", "estp", "esfp",
        "九型", "人格", "认知功能", "主导功能", "功能堆栈", "依恋", "荣格",
        "外倾", "内倾", "ni", "ne", "si", "se", "ti", "te", "fi", "fe",
        "心理", "投射", "阴影", "防御机制", "依恋类型", "安全型", "焦虑型",
        "四个维度", "八个认知", "维度", "外向", "内向", "感觉", "直觉",
        "判断", "知觉", "功能排序", "类型学",
    ],
    "thesis": [
        "论文", "实验", "准确率", "数据集", "基线", "消融", "损失函数",
        "过拟合", "类别不平衡", "召回率", "f1", "深度学习", "神经网络",
        "训练", "验证集", "混淆矩阵", "采样", "真正例", "假正例", "假负例",
        "tp", "fp", "fn", "tn", "分类", "评估",
    ],
    "work": [
        "deeprag", "rag", "langchain", "langgraph", "求职", "简历", "面试",
        "agent", "向量库", "chroma", "qdrant", "ollama",
    ],
    "social": [
        "社会", "制度", "政治", "经济史", "社会学",
    ],
    "ideas": [
        "奇思", "脑洞", "科幻设定",
    ],
}

# 域 → 优先 collection（与 build 脚本一致）
DOMAIN_COLLECTIONS = {
    "psychology": ["proj_psychology", "proj_work"],
    "thesis": ["proj_thesis", "proj_work"],
    "work": ["proj_work", "proj_zhesi"],
    "social": ["proj_social", "proj_work"],
    "ideas": ["proj_ideas", "proj_work"],
}

# 明显「人格类型代码」单独加强（避免被误判为普通英文）
_TYPE_CODE = re.compile(
    r"\b(I[NST][FT][JP]|E[NST][FT][JP])\b",
    re.I,
)


def detect_domains(question: str) -> List[str]:
    """返回匹配到的域列表（按命中强度粗排）。"""
    if not question:
        return []
    q = question.strip()
    q_lower = q.lower()
    scores = {}

    if _TYPE_CODE.search(q):
        scores["psychology"] = scores.get("psychology", 0) + 5

    for domain, kws in DOMAIN_KEYWORDS.items():
        s = 0
        for kw in kws:
            if kw.isascii():
                if kw.lower() in q_lower:
                    s += 2 if len(kw) <= 4 else 1
            else:
                if kw in q:
                    s += 2
        if s:
            scores[domain] = scores.get(domain, 0) + s

    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


def recommend_collections(question: str, available: Optional[List[str]] = None) -> List[str]:
    """推荐 collection 顺序；无域命中则返回 available 原序或默认 work。"""
    domains = detect_domains(question)
    ordered: List[str] = []
    for d in domains:
        for c in DOMAIN_COLLECTIONS.get(d, []):
            if c not in ordered:
                ordered.append(c)
    if available:
        # 只保留实际存在的库，再拼上其余有数据的库
        avail_set = set(available)
        ordered = [c for c in ordered if c in avail_set]
        for c in available:
            if c not in ordered:
                ordered.append(c)
        return ordered
    return ordered or ["proj_work", "proj_psychology"]


def collection_conflicts_with_query(
    question: str, collection: str
) -> Tuple[bool, str, Optional[str]]:
    """用户手选库是否与问题域冲突。

    Returns:
        (冲突?, 说明, 建议 collection)
    """
    domains = detect_domains(question)
    if not domains:
        return False, "", None

    primary = domains[0]
    preferred = DOMAIN_COLLECTIONS.get(primary, [])
    if not preferred:
        return False, "", None

    # 选中的库在推荐列表里 → 不冲突
    if collection in preferred:
        return False, "", preferred[0]

    # 强冲突：心理问题却选了 thesis
    if primary == "psychology" and (
        "thesis" in collection or collection == "proj_social"
    ):
        return (
            True,
            f"问题偏「{primary}/人格心理」，当前库「{collection}」不匹配，易答非所问",
            preferred[0],
        )
    if primary == "thesis" and "psychology" in collection:
        return (
            True,
            f"问题偏论文/实验，当前库「{collection}」偏心理",
            preferred[0],
        )
    # 其它：弱提示但不强制
    if preferred[0] != collection:
        return (
            True,
            f"更推荐使用「{preferred[0]}」（检测到域={primary}）",
            preferred[0],
        )
    return False, "", preferred[0]


def docs_match_query_domain(question: str, docs: list, min_hits: int = 1) -> bool:
    """粗检：检索到的文档是否至少沾边问题域关键词。"""
    domains = detect_domains(question)
    if not domains:
        return True  # 无法判断则放行
    # 收集该域关键词
    kws = []
    for d in domains[:2]:
        kws.extend(DOMAIN_KEYWORDS.get(d, [])[:20])
    if _TYPE_CODE.search(question or ""):
        kws.extend(["人格", "功能", "MBTI", "mbti", "认知"])

    if not docs:
        return False
    blob = " ".join(
        (d.get("content") if isinstance(d, dict) else getattr(d, "content", ""))
        or ""
        for d in docs[:8]
    )
    blob_l = blob.lower()
    hits = 0
    for kw in kws:
        if not kw:
            continue
        if kw.isascii():
            if kw.lower() in blob_l:
                hits += 1
        elif kw in blob:
            hits += 1
        if hits >= min_hits:
            return True
    return False
