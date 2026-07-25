"""答案生成Agent — 带引用溯源的生成（v2.7：限流重试 + 响应缓存, v2.9：Prompt模板化, v2.9.2: Prompt Cache集成）

格式：精简答案在前 → 解释在后 → 引用来源最后
"""
from src.config import get_llm, get_temperature, LLM_MODEL, LLM_BACKEND
from langchain_core.messages import HumanMessage, SystemMessage
from src.state import GradedDocument, Citation
from src.llm.rate_limiter import retry_with_backoff
from src.retrieval.cache import make_llm_cache_key, get_cached_llm_response, set_cached_llm_response

# v2.9.2: Prompt Cache集成
try:
    from src.llm.prompt_cache import get_prompt_cache_manager
    _prompt_cache_available = True
except ImportError:
    _prompt_cache_available = False

# v2.9.2: Prompt Templates集成
try:
    from src.llm.prompt_templates import PromptBuilder, PromptTemplates
    _prompt_templates_available = True
except ImportError:
    _prompt_templates_available = False

# v2.9: 从PromptManager加载，失败时降级到硬编码
_FALLBACK_SYSTEM_PROMPT = """你是知识库问答专家。根据检索到的文档回答用户问题。

回答格式：
1. 【直接回答】1-3句话直接回答问题核心
2. 【详细解释】展开解释，可引用文档
3. 【引用来源】[1] 文件名 第N块

引用规则（必须遵守）：
- 每个事实性断言后必须标注引用编号，如"INTJ的主导功能是Ni[1]"
- [N]对应参考文档编号（[1]=文档1, [2]=文档2...）
- 未标注引用的断言视为幻觉
- 引用编号必须与提供的参考文档列表对应
- 纯推理/总结性语句可不标注引用

MBTI 功能堆栈规则（必须遵守）：
- 若文档中出现明确行如「INTJ: Ni-Te-Fi-Se」，【直接回答】必须使用该四功能顺序，不得改写为 Si-Fe 等
- 若提供【对话上下文】或【一致性硬约束】，不得与已确认堆栈矛盾
- 不得把「荣格八维原则/抽象讨论」误写成某类型的功能堆栈

规则：只基于文档与给定对话上下文回答，不使用外部知识。答案精炼。
"""

def _build_system_prompt_with_templates() -> str:
    """使用五要素模板构建SYSTEM_PROMPT（v2.9.2新增）

    五要素：
    1. Role — 知识库问答专家
    2. Task — 根据文档回答问题
    3. Context — 引用规则和格式要求
    4. Format — 三段式输出格式
    5. Examples — 标准问答示例
    """
    if not _prompt_templates_available:
        return _FALLBACK_SYSTEM_PROMPT

    try:
        builder = PromptBuilder("generation_system")

        # Role: 角色设定
        builder.role("你是知识库问答专家，擅长根据检索到的文档准确回答问题，并标注引用来源。")

        # Task: 任务描述
        builder.task("根据检索到的文档回答用户问题，确保每个事实性断言都有引用支撑。")

        # Format: 输出格式
        builder.format("""三段式输出格式：
1. 【直接回答】1-3句话直接回答问题核心
2. 【详细解释】展开解释，可引用文档
3. 【引用来源】[1] 文件名 第N块""")

        # Constraints: 约束条件
        builder.constraint("每个事实性断言后必须标注引用编号，如'INTJ的主导功能是Ni[1]'")
        builder.constraint("[N]对应参考文档编号（[1]=文档1, [2]=文档2...）")
        builder.constraint("未标注引用的断言视为幻觉")
        builder.constraint("引用编号必须与提供的参考文档列表对应")
        builder.constraint("纯推理/总结性语句可不标注引用")
        builder.constraint("只基于文档回答，不使用外部知识")

        # Examples: 示例
        builder.example(
            "INTJ的主导功能是什么？",
            "INTJ的主导功能是内向直觉（Ni）[1]。Ni使INTJ擅长洞察事物的内在联系和未来趋势，具有强烈的预见能力[1]。"
        )

        return builder.render()
    except Exception as e:
        log.warning(f"[Generator] 五要素模板构建失败: {e}")
        return _FALLBACK_SYSTEM_PROMPT


def _get_system_prompt() -> str:
    """获取SYSTEM_PROMPT（v2.9: 优先从PromptManager加载，v2.9.2: 五要素模板）"""
    # 优先使用五要素模板
    if _prompt_templates_available:
        return _build_system_prompt_with_templates()

    # 降级到PromptManager
    try:
        from src.tools.modules.prompt_manager import get_pipeline_prompt
        return get_pipeline_prompt("generation_system")
    except Exception:
        return _FALLBACK_SYSTEM_PROMPT

SYSTEM_PROMPT = _get_system_prompt()


def generate_answer(
    question: str,
    relevant_docs: list[GradedDocument],
    force_regenerate: bool = False,
    prior_context: str = "",
) -> dict:
    """基于相关文档生成带引用的回答（v2.8：缓存 + 限流重试；v2.9.3：多轮 prior_context）"""
    if not relevant_docs:
        return {
            "answer": "根据现有知识库资料，未找到与此问题相关的信息。",
            "citations": [],
        }

    prior_context = (prior_context or "").strip()

    # v2.7: 检查LLM响应缓存（v2.8: force_regenerate时跳过；有 prior 时纳入 cache key）
    context_preview = "".join(d.get("content", "")[:200] for d in relevant_docs[:3])
    cache_key = make_llm_cache_key(
        question + ("\n" + prior_context[:200] if prior_context else ""),
        context_preview,
        LLM_MODEL or LLM_BACKEND,
    )
    if not force_regenerate:
        cached = get_cached_llm_response(cache_key)
        if cached:
            answer = cached
            citations = []
            for doc in relevant_docs:
                if doc["source"] in answer or f"第{doc['page']}块" in answer:
                    citations.append(Citation(
                        text=doc.get("content", doc.get("text", ""))[:200],
                        source=doc["source"],
                        page=doc["page"],
                    ))
            return {"answer": answer, "citations": citations}

    llm = get_llm(temperature=get_temperature("generation"))
    if llm is None:
        return generate_answer_offline(question, relevant_docs)

    # v2.8.1: 只取top-2相关文档（v2.9.1: 用PromptCompressor替代粗暴截断）
    # 会话已确认事实优先保留
    dialog_docs = [d for d in relevant_docs if str(d.get("source", "")).startswith("会话")]
    other_docs = [d for d in relevant_docs if d not in dialog_docs]
    sorted_docs = dialog_docs[:1] + sorted(
        other_docs, key=lambda d: -d.get("relevance_score", 0)
    )[:2]

    # v2.9.1: 智能压缩文档上下文（替代 content[:400] 截断）
    from src.llm.prompt_compressor import get_compressor
    compressor = get_compressor()
    context = compressor.compress(question, sorted_docs, max_tokens=400)

    # 构造文档上下文（v2.9.1: compressor已包含来源标记）
    if not context:
        # 降级到原始截断
        context_parts = []
        for i, doc in enumerate(sorted_docs, 1):
            content = doc.get("content", doc.get("text", ""))[:400]
            context_parts.append(
                f"[文档{i}] 来源: {doc['source']}, 第{doc['page']}块\n"
                f"内容:\n{content}\n"
            )
        context = "\n---\n".join(context_parts)

    prior_block = f"{prior_context}\n" if prior_context else ""
    prompt = f"""{prior_block}问题：{question}

参考文档（{len(relevant_docs)}篇）：
{context}

请按格式回答：先【直接回答】问题核心（1-3句），再【详细解释】（可引用文档），最后【引用来源】列出引用清单。
若文档含明确功能堆栈（如 INTJ: Ni-Te-Fi-Se），【直接回答】必须原样使用该顺序，禁止与【一致性硬约束】矛盾。"""

    system_content = SYSTEM_PROMPT
    if prior_context:
        system_content = SYSTEM_PROMPT + "\n\n" + prior_context[:800]

    # v2.9.2: 使用Prompt Cache构建缓存友好的消息
    if _prompt_cache_available:
        cache_manager = get_prompt_cache_manager()
        cache_messages = cache_manager.build_messages(
            system_prompt=system_content,
            user_query=prompt,
            context=context,
        )
        # 转换为LangChain消息格式
        lc_messages = []
        for msg in cache_messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                from langchain_core.messages import AIMessage
                lc_messages.append(AIMessage(content=msg["content"]))
        messages = lc_messages
    else:
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt),
        ]

    # v2.8.1: 带重试的LLM调用（禁用思考模式，num_predict=300）
    @retry_with_backoff(max_retries=3, base_delay=1.5)
    def _invoke_llm():
        from src.config import LLM_BACKEND, LLM_MODEL
        if LLM_BACKEND == "ollama":
            # v2.8.2: 使用原生API，think模式由全局开关控制
            from src.llm.ollama_helper import ollama_chat
            ollama_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
            content, _thinking = ollama_chat(
                ollama_messages, LLM_MODEL,
                temperature=get_temperature("generation"), num_predict=300,
            )
            if content and len(content.strip()) >= 5:
                return content
            elif _thinking:
                # 思考模式下 content 可能为空，取 thinking 最后几句
                lines = [l.strip() for l in _thinking.strip().split("\n") if l.strip()]
                return "\n".join(lines[-3:]) if lines else "模型未返回有效内容。"
            else:
                return "模型未返回有效内容。"
        else:
            # 其他后端用 langchain
            response = llm.invoke(messages)
            content = response.content
            if not content or len(content.strip()) < 5:
                content = "模型未返回有效内容。"
            return content

    answer = _invoke_llm()

    # v2.7: 缓存LLM响应
    set_cached_llm_response(cache_key, answer)

    # 提取引用
    citations = []
    for doc in relevant_docs:
        if doc["source"] in answer or f"第{doc['page']}块" in answer:
            citations.append(Citation(
                text=doc.get("content", doc.get("text", ""))[:200],
                source=doc["source"],
                page=doc["page"],
            ))

    return {"answer": answer, "citations": citations}


def generate_answer_offline(question: str, relevant_docs: list[GradedDocument]) -> dict:
    """离线版生成（基于关键词的智能摘要，不是简单取前2句）"""
    if not relevant_docs:
        return {"answer": "未找到相关信息。", "citations": []}

    import jieba

    # 按相关度排序取top3
    sorted_docs = sorted(relevant_docs, key=lambda d: -d["relevance_score"])[:3]

    # 提取问题关键词
    question_keywords = set(jieba.cut(question))
    # 过滤停用词
    stopwords = {"的", "是", "了", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那"}
    question_keywords = question_keywords - stopwords

    # 从文档中提取与问题最相关的句子
    key_sentences = []
    citations = []

    for i, doc in enumerate(sorted_docs, 1):
        content = doc.get("content", doc.get("text", ""))
        # 分句
        sentences = [s.strip() for s in content.replace('。', '.|').replace('？', '?|').replace('！', '!|').replace('\n', '|').split('|') if len(s.strip()) > 5]

        # 找包含问题关键词的句子
        relevant_sentences = []
        for sent in sentences:
            sent_keywords = set(jieba.cut(sent)) - stopwords
            overlap = question_keywords & sent_keywords
            if len(overlap) >= 1:
                relevant_sentences.append((sent, len(overlap)))

        # 按关键词重叠数排序，取前2句
        relevant_sentences.sort(key=lambda x: -x[1])
        for sent, _ in relevant_sentences[:2]:
            key_sentences.append(sent)

        citations.append(Citation(text=content[:200], source=doc["source"], page=doc["page"]))

    # 构造回答
    if key_sentences:
        answer_parts = [
            f"根据知识库资料：\n",
            "\n".join(key_sentences[:5]),
            f"\n\n引用来源："
        ]
        for i, cite in enumerate(citations, 1):
            answer_parts.append(f"\n[{i}] {cite['source']} 第{cite['page']}块")
    else:
        answer_parts = [
            f"根据知识库资料，关于「{question}」：\n",
            sorted_docs[0].get("content", sorted_docs[0].get("text", ""))[:300],
            f"\n\n引用来源：\n[1] {citations[0]['source']} 第{citations[0]['page']}块"
        ]

    return {"answer": "".join(answer_parts), "citations": citations}


def generate_answer_stream(
    question: str,
    relevant_docs: list[GradedDocument],
    prior_context: str = "",
):
    """流式生成带引用的回答（generator）；支持多轮 prior_context"""
    if not relevant_docs:
        yield "根据现有知识库资料，未找到与此问题相关的信息。"
        return
    llm = get_llm(temperature=get_temperature("generation"))
    if llm is None:
        result = generate_answer_offline(question, relevant_docs)
        yield result["answer"]
        return

    prior_context = (prior_context or "").strip()
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        content = doc.get("content", doc.get("text", ""))
        context_parts.append(
            f"[文档{i}] 来源: {doc['source']}, 第{doc['page']}块\n"
            f"相关度: {doc.get('relevance_score', 0):.0%}\n"
            f"内容:\n{content}\n"
        )
    context = "\n---\n".join(context_parts)

    prior_block = f"{prior_context}\n" if prior_context else ""
    prompt = f"""{prior_block}问题：{question}

参考文档（{len(relevant_docs)}篇）：
{context}

请按格式回答：先【直接回答】问题核心（1-3句），再【详细解释】（可引用文档），最后【引用来源】列出引用清单。
若文档含明确功能堆栈，【直接回答】必须沿用该顺序，禁止与【一致性硬约束】矛盾。"""

    system_content = SYSTEM_PROMPT + (f"\n\n{prior_context[:800]}" if prior_context else "")
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=prompt),
    ]

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


# === v2.8.3: 直接回答（跳过RAG的闲聊/常识问题）===

_DIRECT_PROMPT = """你是一个智能助手。请直接回答用户的问题，简洁友好。
不需要引用任何文档，用你自己的知识回答即可。"""


def generate_direct_answer(question: str) -> str:
    """v2.8.3: 直接LLM回答（跳过RAG，用于闲聊/常识问题）

    Returns:
        str: 回答文本
    """
    from src.config import LLM_BACKEND, LLM_MODEL

    if LLM_BACKEND == "ollama":
        from src.llm.ollama_helper import ollama_chat
        ollama_messages = [
            {"role": "system", "content": _DIRECT_PROMPT},
            {"role": "user", "content": question},
        ]
        content, _ = ollama_chat(
            ollama_messages, LLM_MODEL,
            temperature=0.5, num_predict=300, think=False,
        )
        return content or "抱歉，我没有理解你的问题。"
    else:
        llm = get_llm(temperature=0.5)
        if llm is None:
            return "抱歉，当前没有可用的LLM。"
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=_DIRECT_PROMPT),
            HumanMessage(content=question),
        ])
        return response.content or "抱歉，我没有理解你的问题。"


def generate_direct_answer_stream(question: str):
    """v2.8.3: 直接LLM流式回答（跳过RAG，用于闲聊/常识问题）"""
    llm = get_llm(temperature=0.5)
    if llm is None:
        yield "抱歉，当前没有可用的LLM。"
        return

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=_DIRECT_PROMPT),
        HumanMessage(content=question),
    ]

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
