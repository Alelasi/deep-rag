"""双Agent精准模式 — v2.8.6 (v2.9: Prompt模板化, v2.9.2: A2A协议集成)

核心优化：
1. 极速模式：跳过LLM对比，用本地关键词启发式检测矛盾（省2-3s）
2. 双答案展示：矛盾时两个答案都返回，让用户自行分辨
3. 5种提示词策略：直接/分析/苏格拉底/思维链/精简（v2.9: 从PromptManager加载）
4. 每个Agent可指定不同模型（支持交叉配置）
5. agree时直接取较长答案，跳过merge调用
6. v2.9.2: A2A协议集成 — 使用Agent Card和Task状态机管理协作
"""
import json
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import SILICONFLOW_API_KEY, ZHIPU_API_KEY, get_temperature
from src.state import GradedDocument, Citation

log = logging.getLogger("deeprag")

# v2.9.2: A2A协议集成
try:
    from src.agents.a2a_protocol import get_a2a_protocol, TaskStatus
    _a2a_available = True
except ImportError:
    _a2a_available = False
    log.warning("[DualAgent] A2A协议模块不可用")

# === 5种提示词策略（v2.9: 从PromptManager加载，失败时降级到硬编码）===
_FALLBACK_STRATEGIES = {
    "direct": """你是知识库问答专家。根据文档直接回答。
格式：1-2句结论 → 2-3句细节 → 引用来源。精炼准确，只基于文档。答案不超过200字。""",

    "analytical": """你是严谨的分析专家。根据文档分析回答。
格式：问题核心(1句) → 关键证据(2-3条) → 结论(1-2句)。标注来源。答案不超过250字。""",

    "socratic": """你是苏格拉底式问答专家。通过追问方式回答。
格式：先指出问题的核心前提 → 列出文档中的关键证据 → 通过逻辑推导得出结论。答案不超过250字。""",

    "chain_of_thought": """你是逻辑推理专家。按思维链方式回答。
格式：Step1:理解问题 → Step2:提取文档关键信息 → Step3:推理分析 → Step4:结论。答案不超过300字。""",

    "concise": """你是极简回答专家。只给结论。
格式：直接给出1-3句话的准确结论。不超过100字。不要解释过程。""",
}

# PromptManager名称映射
_STRATEGY_PM_NAMES = {
    "direct": "strategy_direct",
    "analytical": "strategy_analytical",
    "socratic": "strategy_socratic",
    "chain_of_thought": "strategy_cot",
    "concise": "strategy_concise",
}

def _load_strategies() -> dict:
    """从PromptManager加载所有策略（v2.9新增）"""
    strategies = {}
    try:
        from src.tools.modules.prompt_manager import get_pipeline_prompt
        for key, pm_name in _STRATEGY_PM_NAMES.items():
            try:
                strategies[key] = get_pipeline_prompt(pm_name)
            except Exception:
                strategies[key] = _FALLBACK_STRATEGIES[key]
    except ImportError:
        strategies = _FALLBACK_STRATEGIES.copy()
    return strategies

PROMPT_STRATEGIES = _load_strategies()

# 默认使用 direct + analytical
AGENT_A_PROMPT = PROMPT_STRATEGIES["direct"]
AGENT_B_PROMPT = PROMPT_STRATEGIES["analytical"]

# === 对比Prompt（v2.9: 从PromptManager加载）===
_FALLBACK_COMPARE_PROMPT = """对比两个回答是否存在事实矛盾。输出JSON：
{"verdict":"agree|conflict|partial","conflict_points":[],"recommendation":"merge|prefer_a|prefer_b|re_search"}
agree=一致, conflict=矛盾, partial=部分差异。merge=融合, prefer_a/b=选某方, re_search=需重搜。"""

def _load_compare_prompt() -> str:
    """加载对比Prompt（v2.9新增）"""
    try:
        from src.tools.modules.prompt_manager import get_pipeline_prompt
        return get_pipeline_prompt("compare")
    except Exception:
        return _FALLBACK_COMPARE_PROMPT

COMPARE_PROMPT = _load_compare_prompt()


def _build_context(question: str, docs: list[GradedDocument], max_docs: int = 3) -> str:
    """构造文档上下文（v2.9.1: 用PromptCompressor替代截断到300字/篇）"""
    sorted_docs = sorted(docs, key=lambda d: -d.get("relevance_score", 0))[:max_docs]

    # v2.9.1: 智能压缩（替代 content[:300] 截断）
    try:
        from src.llm.prompt_compressor import get_compressor
        compressor = get_compressor()
        compressed = compressor.compress(question, sorted_docs, max_tokens=300)
        if compressed:
            return compressed
    except Exception:
        pass

    # 降级到原始截断
    parts = []
    for i, doc in enumerate(sorted_docs, 1):
        content = doc.get("content", doc.get("text", ""))[:300]
        parts.append(f"[文档{i}] {doc.get('source', '?')} 第{doc.get('page', 0)}块\n{content}\n")
    return "\n---\n".join(parts)


def _get_llm_for_model(model_name: str, temperature: float = 0.3):
    """根据模型名直接创建LLM实例"""
    from langchain_openai import ChatOpenAI

    if model_name and model_name.startswith("glm-") and not model_name.startswith("THUDM"):
        return ChatOpenAI(
            model=model_name, temperature=temperature,
            api_key=ZHIPU_API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
    else:
        actual_model = model_name or "THUDM/GLM-Z1-9B-0414"
        return ChatOpenAI(
            model=actual_model, temperature=temperature,
            api_key=SILICONFLOW_API_KEY,
            base_url="https://api.siliconflow.cn/v1",
        )


def _invoke_agent(prompt_template: str, question: str, context: str,
                  model_name: str, temperature: float = None) -> str:
    """单个Agent调用指定模型生成答案"""
    from langchain_core.messages import HumanMessage, SystemMessage

    if temperature is None:
        temperature = get_temperature("generation")

    prompt = f"问题：{question}\n\n参考文档：\n{context}\n\n请按你的策略回答。"

    try:
        llm = _get_llm_for_model(model_name, temperature)
        response = llm.invoke([
            SystemMessage(content=prompt_template),
            HumanMessage(content=prompt),
        ])
        return response.content or "模型未返回有效内容。"
    except Exception as e:
        log.error(f"[DualAgent] {model_name} 调用失败: {e}")
        return f"调用失败: {e}"


def dual_generate(question: str, docs: list[GradedDocument],
                  model_a: str = "glm-4-flash",
                  model_b: str = "glm-4-flash",
                  strategy_a: str = "direct",
                  strategy_b: str = "analytical") -> dict:
    """双Agent并行生成答案

    Args:
        model_a: Agent A使用的模型名
        model_b: Agent B使用的模型名
        strategy_a: Agent A的提示词策略
        strategy_b: Agent B的提示词策略
    """
    context = _build_context(question, docs)
    prompt_a = PROMPT_STRATEGIES.get(strategy_a, AGENT_A_PROMPT)
    prompt_b = PROMPT_STRATEGIES.get(strategy_b, AGENT_B_PROMPT)
    results = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_invoke_agent, prompt_a, question, context, model_a)
        future_b = executor.submit(_invoke_agent, prompt_b, question, context, model_b)

        t0 = time.time()
        for future in as_completed([future_a, future_b]):
            elapsed = time.time() - t0
            if future == future_a:
                results["answer_a"] = future.result()
                results["elapsed_a"] = round(elapsed, 2)
                results["strategy_a"] = strategy_a
            else:
                results["answer_b"] = future.result()
                results["elapsed_b"] = round(elapsed, 2)
                results["strategy_b"] = strategy_b

    results["total_elapsed"] = round(time.time() - t0, 2)
    return results


def _local_conflict_check(question: str, answer_a: str, answer_b: str) -> dict:
    """本地启发式矛盾检测 — 无需LLM调用，零延迟

    检测规则：
    1. 关键数字/术语不一致
    2. 答案长度差异过大（>3倍）
    3. 否定词对立（一个说"是"，一个说"不是"）
    4. 关键实体差异
    """
    conflict_points = []

    # 规则1：提取数字对比
    nums_a = set(re.findall(r'\d+\.?\d*', answer_a))
    nums_b = set(re.findall(r'\d+\.?\d*', answer_b))
    # 过滤掉文档编号等无关数字
    meaningful_nums_a = {n for n in nums_a if len(n) >= 1 and n not in {'1', '2', '3', '0'}}
    meaningful_nums_b = {n for n in nums_b if len(n) >= 1 and n not in {'1', '2', '3', '0'}}

    if meaningful_nums_a and meaningful_nums_b:
        diff_nums = meaningful_nums_a.symmetric_difference(meaningful_nums_b)
        if len(diff_nums) >= 2:
            conflict_points.append(f"数字差异: A有{meaningful_nums_a}, B有{meaningful_nums_b}")

    # 规则2：长度差异
    len_ratio = max(len(answer_a), len(answer_b)) / max(min(len(answer_a), len(answer_b)), 1)
    if len_ratio > 3:
        conflict_points.append(f"长度差异大: A={len(answer_a)}字 vs B={len(answer_b)}字")

    # 规则3：否定对立
    neg_words = ["不是", "错误", "不正确", "并非", "没有", "不存在", "不能"]
    pos_words = ["是", "正确", "存在", "有", "可以"]
    a_has_neg = any(w in answer_a for w in neg_words)
    b_has_neg = any(w in answer_b for w in neg_words)
    a_has_pos = any(w in answer_a for w in pos_words)
    b_has_pos = any(w in answer_b for w in pos_words)
    if (a_has_neg and b_has_pos) or (a_has_pos and b_has_neg):
        # 检查是否针对同一主体
        if a_has_neg and b_has_neg:
            pass  # 都是否定，不算矛盾
        elif a_has_neg != b_has_neg:
            conflict_points.append("可能存在肯定/否定对立")

    # 规则4：关键英文术语/实体差异（如Ni vs Ti, Beijing vs Shanghai）
    eng_a = set(re.findall(r'\b[A-Z][a-z]{1,}(?:\s[A-Z][a-z]+)*\b', answer_a))
    eng_b = set(re.findall(r'\b[A-Z][a-z]{1,}(?:\s[A-Z][a-z]+)*\b', answer_b))
    # 过滤常见非实体词
    common_words = {"The", "This", "That", "Agent", "Step", "DNA", "RAG", "CRAG", "Web", "LLM"}
    eng_a_filtered = eng_a - common_words
    eng_b_filtered = eng_b - common_words
    if eng_a_filtered and eng_b_filtered:
        unique_eng = eng_a_filtered.symmetric_difference(eng_b_filtered)
        if len(unique_eng) >= 2:
            conflict_points.append(f"英文实体差异: A有{eng_a_filtered}, B有{eng_b_filtered}")

    # 规则5：中文关键实体差异（提取2-4字词组）
    terms_a = set(re.findall(r'[\u4e00-\u9fa5]{2,4}', answer_a))
    terms_b = set(re.findall(r'[\u4e00-\u9fa5]{2,4}', answer_b))
    unique_a = terms_a - terms_b
    unique_b = terms_b - terms_a
    if len(unique_a) > 10 and len(unique_b) > 10:
        conflict_points.append(f"术语差异较大: A独有{len(unique_a)}词, B独有{len(unique_b)}词")

    if conflict_points:
        return {
            "verdict": "partial",
            "conflict_points": conflict_points,
            "recommendation": "show_both",
        }
    else:
        return {
            "verdict": "agree",
            "conflict_points": [],
            "recommendation": "merge",
        }


def compare_answers(question: str, answer_a: str, answer_b: str,
                    docs: list[GradedDocument],
                    model_name: str = "glm-4-flash") -> dict:
    """LLM对比两个答案 — 用最快模型"""
    from langchain_core.messages import HumanMessage, SystemMessage

    context = _build_context(question, docs, max_docs=2)
    prompt = f"问题：{question}\n文档：{context}\n\n## 回答A：\n{answer_a}\n\n## 回答B：\n{answer_b}\n\n判断是否有事实矛盾。"

    try:
        # v2.9.1: 使用Constrained Decoding替代手动JSON解析
        from src.llm.constrained_decoder import get_structured_llm, COMPARE_SCHEMA, safe_parse_json
        llm = _get_llm_for_model(model_name, temperature=get_temperature("comparison"))
        structured_llm = get_structured_llm(llm, COMPARE_SCHEMA)
        data = structured_llm.invoke([
            SystemMessage(content=COMPARE_PROMPT),
            HumanMessage(content=prompt),
        ])

        # 处理不同返回类型
        if isinstance(data, dict):
            pass  # 结构化输出成功
        elif isinstance(data, str):
            data = safe_parse_json(data) or {
                "verdict": "agree", "conflict_points": [],
                "recommendation": "merge", "reasoning": "解析失败"
            }
        else:
            data = data.dict() if hasattr(data, 'dict') else dict(data)
    except Exception as e:
        log.warning(f"[DualAgent] 对比调用失败: {e}")
        data = {"verdict": "agree", "conflict_points": [],
                "recommendation": "merge", "reasoning": f"对比失败: {e}"}

    return {
        "verdict": data.get("verdict", "agree"),
        "conflict_points": data.get("conflict_points", []),
        "recommendation": data.get("recommendation", "merge"),
        "reasoning": data.get("reasoning", ""),
    }


def merge_answers(question: str, answer_a: str, answer_b: str,
                  docs: list[GradedDocument],
                  model_name: str = "glm-4-flash") -> str:
    """融合两个一致的答案"""
    from langchain_core.messages import HumanMessage, SystemMessage

    MERGE_PROMPT = """融合两个回答的优点。采纳A的简洁结论+B的分析过程，去重，保留引用。"""
    prompt = f"问题：{question}\n\n## 回答A：\n{answer_a}\n\n## 回答B：\n{answer_b}\n\n融合输出最终答案。"

    try:
        llm = _get_llm_for_model(model_name, temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=MERGE_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content or answer_a
    except Exception as e:
        log.warning(f"[DualAgent] 融合失败: {e}")
        return answer_a if len(answer_a) >= len(answer_b) else answer_b


def arbitrate_answer(question: str, answer_a: str, answer_b: str,
                     conflict_points: list[str],
                     new_docs: list[GradedDocument],
                     model_name: str = "glm-4-flash") -> str:
    """仲裁生成 — 矛盾时基于新检索文档生成最终答案"""
    from langchain_core.messages import HumanMessage, SystemMessage

    ARBITRATE_PROMPT = """你是仲裁专家。两个Agent给出矛盾答案。基于新文档证据判断谁正确，给出最终答案。"""
    context = _build_context(question, new_docs, max_docs=4)
    conflict_text = "\n".join(f"- {p}" for p in conflict_points)
    prompt = f"问题：{question}\n\n矛盾点：\n{conflict_text}\n\n## 回答A：\n{answer_a}\n\n## 回答B：\n{answer_b}\n\n## 新证据：\n{context}\n\n仲裁并给出最终答案。"

    try:
        llm = _get_llm_for_model(model_name, temperature=get_temperature("arbitration"))
        response = llm.invoke([
            SystemMessage(content=ARBITRATE_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content or answer_a
    except Exception as e:
        log.warning(f"[DualAgent] 仲裁失败: {e}")
        return answer_a


def self_consistency_generate(
    question: str,
    docs: list[GradedDocument],
    model_name: str = "glm-4-flash",
    strategy: str = "direct",
    n_samples: int = 3,
    temperature: float = 0.7,
) -> dict:
    """Self-Consistency多次采样投票 — v2.9.2新增

    面试要点（04-17 CoT）：
    - 对同一个问题用较高temperature生成N条推理路径
    - 取最终答案里出现最多次的（多数投票）
    - 能在CoT基础上进一步提升5-15%准确率

    Args:
        model_name: 使用的模型
        strategy: 提示词策略
        n_samples: 采样次数（默认3次）
        temperature: 采样温度（默认0.7，较高以增加多样性）

    Returns:
        {
            "answer": 投票后的最终答案,
            "samples": [所有采样结果],
            "vote_counts": {答案: 票数},
            "consistency": 一致性分数（0-1）
        }
    """
    context = _build_context(question, docs)
    prompt_template = PROMPT_STRATEGIES.get(strategy, AGENT_A_PROMPT)

    # 多次采样
    samples = []
    with ThreadPoolExecutor(max_workers=min(n_samples, 4)) as executor:
        futures = [
            executor.submit(_invoke_agent, prompt_template, question, context, model_name, temperature)
            for _ in range(n_samples)
        ]
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                samples.append({
                    "index": i,
                    "answer": result,
                    "length": len(result),
                })
            except Exception as e:
                log.warning(f"[SelfConsistency] 采样{i}失败: {e}")

    if not samples:
        return {
            "answer": "采样失败，无有效结果。",
            "samples": [],
            "vote_counts": {},
            "consistency": 0.0,
        }

    # 提取关键信息进行投票
    # 简化版：使用答案的前100字作为投票key（实际可用embedding聚类）
    vote_keys = []
    for sample in samples:
        # 提取答案的核心部分（去除引用、格式化字符）
        answer = sample["answer"]
        # 简化：取前100字作为key
        key = answer[:100].strip()
        vote_keys.append(key)

    # 统计票数
    from collections import Counter
    vote_counts = Counter(vote_keys)

    # 找到票数最多的答案
    best_key, best_count = vote_counts.most_common(1)[0]
    consistency = best_count / len(samples)

    # 找到对应的完整答案
    best_answer = None
    for sample in samples:
        if sample["answer"][:100].strip() == best_key:
            best_answer = sample["answer"]
            break

    if best_answer is None:
        best_answer = samples[0]["answer"]

    log.info(
        f"[SelfConsistency] {n_samples}次采样, "
        f"一致性: {consistency:.2f}, "
        f"最佳答案票数: {best_count}/{len(samples)}"
    )

    return {
        "answer": best_answer,
        "samples": samples,
        "vote_counts": dict(vote_counts),
        "consistency": consistency,
        "total_samples": len(samples),
        "winning_votes": best_count,
    }


def precision_generate(question: str, docs: list[GradedDocument],
                       re_search_fn=None,
                       model_a: str = "glm-4-flash",
                       model_b: str = "glm-4-flash",
                       compare_model: str = "glm-4-flash",
                       strategy_a: str = "direct",
                       strategy_b: str = "analytical",
                       fast_mode: bool = True,
                       show_both_on_conflict: bool = True,
                       use_a2a: bool = True,
                       use_self_consistency: bool = False,
                       sc_samples: int = 3) -> dict:
    """精准模式主入口 — 双Agent并行+矛盾检测

    Args:
        model_a: Agent A模型
        model_b: Agent B模型
        compare_model: 对比用模型
        strategy_a/b: 提示词策略 (direct/analytical/socratic/chain_of_thought/concise)
        fast_mode: True=本地启发式检测(零延迟), False=LLM对比(更准确但慢2-3s)
        show_both_on_conflict: True=矛盾时返回两个答案让用户分辨, False=自动仲裁
        use_a2a: 是否使用A2A协议管理任务
        use_self_consistency: 是否使用Self-Consistency多次采样投票（面试要点04-17）
        sc_samples: Self-Consistency采样次数（默认3次）

    极速模式流程: 双Agent并行(5-7s) → 本地检测(0ms) → 一致取较长/矛盾返回双答案
    标准模式流程: 双Agent并行(5-7s) → LLM对比(2-3s) → 融合/仲裁

    v2.9.2: A2A协议集成 — 使用Task状态机管理任务生命周期
    v2.9.2: Self-Consistency集成 — 多次采样投票提升准确率
    """
    t0 = time.time()
    history = []

    # v2.9.2: A2A协议集成 — 创建Task跟踪
    a2a_task = None
    if use_a2a and _a2a_available:
        try:
            protocol = get_a2a_protocol()
            a2a_task = protocol.delegate_task(
                from_agent="coordinator",
                to_agent="precision_agent",
                task_type="precision_generate",
                payload={"question": question, "fast_mode": fast_mode},
            )
            protocol.execute_task(a2a_task)
            history.append(f"[A2A] 任务创建: {a2a_task.task_id}")
        except Exception as e:
            log.warning(f"[DualAgent] A2A任务创建失败: {e}")

    if not docs:
        return {
            "answer": "未找到相关文档，无法生成答案。",
            "citations": [],
            "answer_a": "", "answer_b": "",
            "verdict": "no_docs", "conflict_points": [],
            "recommendation": "none", "re_searched": False,
            "elapsed": round(time.time() - t0, 2),
            "history": ["无文档，跳过双Agent生成"],
            "show_both": False,
        }

    # Step 1: 双Agent并行生成（v2.9.2: 支持Self-Consistency）
    log.info(f"[Precision] 双Agent并行: A={model_a}({strategy_a}), B={model_b}({strategy_b}), fast={fast_mode}")

    if use_self_consistency:
        # v2.9.2: Self-Consistency模式 — 多次采样投票
        log.info(f"[Precision] Self-Consistency模式: {sc_samples}次采样")
        sc_result_a = self_consistency_generate(
            question, docs, model_a, strategy_a, n_samples=sc_samples
        )
        sc_result_b = self_consistency_generate(
            question, docs, model_b, strategy_b, n_samples=sc_samples
        )
        answer_a = sc_result_a["answer"]
        answer_b = sc_result_b["answer"]
        dual_result = {
            "answer_a": answer_a,
            "answer_b": answer_b,
            "elapsed_a": 0,  # Self-Consistency内部已计时
            "elapsed_b": 0,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
        }
        history.append(f"Self-Consistency A: {sc_result_a['consistency']:.2f}一致性, {sc_result_a['total_samples']}次采样")
        history.append(f"Self-Consistency B: {sc_result_b['consistency']:.2f}一致性, {sc_result_b['total_samples']}次采样")
    else:
        # 标准双Agent模式
        dual_result = dual_generate(question, docs, model_a, model_b, strategy_a, strategy_b)
        answer_a = dual_result["answer_a"]
        answer_b = dual_result["answer_b"]

    history.append(f"Agent A({model_a}/{strategy_a}): {len(answer_a)}字")
    history.append(f"Agent B({model_b}/{strategy_b}): {len(answer_b)}字")
    log.info(f"[Precision] A: {len(answer_a)}字, B: {len(answer_b)}字")

    # Step 2: 对比答案
    if fast_mode:
        # 极速模式：本地启发式检测，零延迟
        comparison = _local_conflict_check(question, answer_a, answer_b)
        history.append(f"本地检测(0ms): {comparison['verdict']} → {comparison['recommendation']}")
    else:
        # 标准模式：LLM对比
        comparison = compare_answers(question, answer_a, answer_b, docs, compare_model)
        history.append(f"LLM对比({compare_model}): {comparison['verdict']} → {comparison['recommendation']}")

    verdict = comparison["verdict"]
    conflict_points = comparison["conflict_points"]
    recommendation = comparison["recommendation"]

    log.info(f"[Precision] 对比: {verdict}, 建议: {recommendation}")

    # Step 3: 根据对比结果处理
    re_searched = False
    show_both = False
    final_answer = ""

    if verdict == "agree":
        # 一致 → 取较长答案
        final_answer = answer_a if len(answer_a) >= len(answer_b) else answer_b
        history.append("策略: 一致，取较长答案")

    elif recommendation == "show_both" and show_both_on_conflict:
        # 矛盾 → 返回两个答案让用户分辨
        show_both = True
        final_answer = f"【Agent A（{strategy_a}）】\n{answer_a}\n\n---\n\n【Agent B（{strategy_b}）】\n{answer_b}"
        history.append("策略: 矛盾，返回双答案供用户分辨")

    elif recommendation == "merge":
        # partial一致 → 融合
        final_answer = merge_answers(question, answer_a, answer_b, docs, compare_model)
        history.append("策略: 融合两答案")

    elif recommendation == "prefer_a":
        final_answer = answer_a
        history.append("策略: 采用Agent A")

    elif recommendation == "prefer_b":
        final_answer = answer_b
        history.append("策略: 采用Agent B")

    elif recommendation == "re_search" and re_search_fn:
        # 矛盾 → 重新搜索
        log.info(f"[Precision] 检测到矛盾，触发重新搜索")
        history.append(f"矛盾点: {conflict_points}")
        try:
            new_docs = re_search_fn(question, conflict_points)
            if new_docs:
                existing_ids = {d.get("doc_id") for d in docs}
                combined_docs = list(docs)
                for d in new_docs:
                    if d.get("doc_id") not in existing_ids:
                        combined_docs.append(d)
                        existing_ids.add(d.get("doc_id"))
                final_answer = arbitrate_answer(question, answer_a, answer_b,
                                                 conflict_points, combined_docs, compare_model)
                re_searched = True
                history.append(f"重新检索{len(new_docs)}篇，仲裁生成{len(final_answer)}字")
                docs = combined_docs
            else:
                final_answer = answer_a
                history.append("重新搜索无结果，采用Agent A")
        except Exception as e:
            log.error(f"[Precision] 重新搜索失败: {e}")
            final_answer = answer_a
            history.append(f"重新搜索失败: {e}")
    else:
        final_answer = merge_answers(question, answer_a, answer_b, docs, compare_model)
        history.append("策略: 默认融合")

    # 提取引用
    citations = []
    for doc in docs:
        source = doc.get("source", "")
        page = doc.get("page", 0)
        if source and (source in final_answer or f"第{page}块" in final_answer):
            citations.append(Citation(
                text=doc.get("content", doc.get("text", ""))[:200],
                source=source,
                page=page,
            ))

    elapsed = round(time.time() - t0, 2)
    history.append(f"总耗时: {elapsed}s")

    # v2.9.2: 更新A2A任务状态
    if a2a_task and _a2a_available:
        try:
            a2a_task.status = TaskStatus.COMPLETED
            a2a_task.completed_at = time.time()
            history.append(f"[A2A] 任务完成: {a2a_task.task_id}")
        except Exception:
            pass

    return {
        "answer": final_answer,
        "citations": citations,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "strategy_a": dual_result.get("strategy_a", strategy_a),
        "strategy_b": dual_result.get("strategy_b", strategy_b),
        "elapsed_a": dual_result["elapsed_a"],
        "elapsed_b": dual_result["elapsed_b"],
        "verdict": verdict,
        "conflict_points": conflict_points,
        "recommendation": recommendation,
        "re_searched": re_searched,
        "show_both": show_both,
        "elapsed": elapsed,
        "history": history,
        # v2.9.2: A2A任务信息
        "a2a_task_id": a2a_task.task_id if a2a_task else None,
    }
