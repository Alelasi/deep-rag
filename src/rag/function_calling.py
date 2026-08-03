"""
DeepRAG Function Calling 模式（v2.9新增，从 god-module src/graph.py 抽出）

使用原生 Function Calling 让 LLM 自主决策调用工具，替代 ReAct 模式的文本 JSON 解析方式。
"""
import logging

log = logging.getLogger("deeprag")


FC_SYSTEM_PROMPT = """你是一个智能知识库助手。请根据用户问题，使用提供的工具检索信息并回答。

工作流程：
1. 先用 search_knowledge_base 搜索本地知识库
2. 如果知识库无结果或不相关，用 web_search 搜索互联网
3. 可选：用 check_error_book 检查是否有历史错题记录
4. 检索到足够信息后，调用 generate_answer 生成最终答案

注意：
- 每次调用工具后，系统会返回结果供你参考
- 最多调用5次工具，然后必须生成答案
- 如果已有足够信息，直接调用 generate_answer"""


def function_calling_query(question: str, collection_name: str = "default",
                           max_iterations: int = 5) -> dict:
    """Function Calling 模式查询（v2.9新增）

    使用原生 Function Calling 让 LLM 自主决策调用工具，
    替代 ReAct 模式的文本 JSON 解析方式。

    优势：
    - LLM 原生返回 tool_calls（结构化输出，无需正则解析JSON）
    - 工具 schema 由 GLM_TOOLS 定义，新增工具只需加schema
    - 循环调用直到 LLM 决定生成答案

    Args:
        question:        用户问题
        collection_name: 知识库名称
        max_iterations:  最大工具调用轮次

    Returns:
        完整结果dict（与query()格式一致）
    """
    from src.config import get_llm_with_fallback
    from src.agents.glm_tools import GLM_TOOLS, execute_tool
    from src.agents.generator import generate_answer, generate_direct_answer
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    import time as _time
    import json as _json

    log.info(f"[FC v2.9] 开始Function Calling查询: {question[:50]}")
    t0 = _time.time()

    llm = get_llm_with_fallback()
    if llm is None:
        log.warning("[FC] LLM不可用，降级到enhanced模式")
        from src.pipeline.run import query  # 惰性导入，避免循环依赖
        return query(question, collection_name, max_retries=2, mode="enhanced")

    # 绑定工具到LLM
    try:
        llm_with_tools = llm.bind_tools(GLM_TOOLS)
    except Exception as e:
        log.warning(f"[FC] bind_tools失败({e})，降级到enhanced模式")
        from src.pipeline.run import query  # 惰性导入，避免循环依赖
        return query(question, collection_name, max_retries=2, mode="enhanced")

    # 构建初始对话
    messages = [
        SystemMessage(content=FC_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    # 收集检索到的文档
    all_docs = []
    used_tools = []
    history = []
    should_generate = False

    # FC 循环
    for iteration in range(max_iterations):
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            log.warning(f"[FC] LLM调用失败(轮次{iteration}): {e}")
            if all_docs:
                break
            from src.pipeline.run import query  # 惰性导入，避免循环依赖
            return query(question, collection_name, max_retries=2, mode="enhanced")

        # 检查是否有 tool_calls
        tool_calls = getattr(response, 'tool_calls', None)

        if not tool_calls:
            # LLM 直接返回文本答案（没有调用工具）
            text = response.content if hasattr(response, 'content') else str(response)
            log.info(f"[FC] LLM直接返回答案(轮次{iteration}), 长度{len(text)}")
            history.append(f"[FC] LLM直接回答 (轮次{iteration})")

            elapsed = _time.time() - t0
            return {
                "question": question,
                "collection_name": collection_name,
                "answer": text,
                "citations": [],
                "retrieved_docs": all_docs,
                "graded_docs": all_docs,
                "relevant_count": len(all_docs),
                "history": history + [f"[FC] 总耗时 {elapsed:.1f}s"],
                "current_step": "done",
                "hallucination_score": 1.0 if all_docs else 0.5,
                "fact_check_passed": True,
                "unsupported_claims": [],
                "conflicts": [],
                "web_results": [],
                "retry_count": iteration,
                "need_human_review": False,
                "errors": [],
            }

        # 处理 tool_calls
        messages.append(response)  # 将 AI 响应（含 tool_calls）加入对话

        for tc in tool_calls:
            tool_name = tc.get("name", "") if isinstance(tc, dict) else tc.get("name", "")
            tool_args = tc.get("args", {}) if isinstance(tc, dict) else tc.get("args", {})
            tool_id = tc.get("id", f"call_{iteration}_{tool_name}") if isinstance(tc, dict) else getattr(tc, "id", f"call_{iteration}_{tool_name}")

            log.info(f"[FC] 轮次{iteration}: 调用工具 {tool_name}({tool_args})")
            used_tools.append(tool_name)
            history.append(f"[FC] 轮次{iteration}: {tool_name}")

            # 如果是 generate_answer，结束循环
            if tool_name == "generate_answer":
                log.info(f"[FC] LLM决定生成答案: {tool_args.get('summary', '')[:100]}")
                history.append("[FC] LLM决定生成答案")
                messages.append(ToolMessage(
                    content=f"已准备好生成答案。共检索到{len(all_docs)}篇文档。",
                    tool_call_id=tool_id,
                ))
                should_generate = True
                break

            # 执行工具
            result_str = execute_tool(tool_name, tool_args, collection_name=collection_name)

            # 将工具结果加入对话
            messages.append(ToolMessage(
                content=result_str,
                tool_call_id=tool_id,
            ))

            # 收集检索到的文档
            try:
                result_data = _json.loads(result_str)
                for doc in result_data.get("results", []):
                    all_docs.append({
                        "doc_id": doc.get("doc_id", ""),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", ""),
                        "page": doc.get("page", 0),
                        "metadata": {},
                        "similarity": doc.get("score", 0.0),
                        "relevance_score": doc.get("score", 0.0),
                    })
            except (_json.JSONDecodeError, KeyError):
                pass

        if should_generate:
            break

    # 用检索到的文档生成最终答案
    elapsed_retrieval = _time.time() - t0
    log.info(f"[FC] 检索完成: {len(all_docs)}篇文档, {elapsed_retrieval:.1f}s, 工具: {used_tools}")

    if not all_docs:
        history.append("[FC] 未检索到文档，直接LLM回答")
        answer = generate_direct_answer(question)
        elapsed = _time.time() - t0
        return {
            "question": question,
            "collection_name": collection_name,
            "answer": answer,
            "citations": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "relevant_count": 0,
            "history": history + [f"[FC] 总耗时 {elapsed:.1f}s"],
            "current_step": "done",
            "hallucination_score": 0.5,
            "fact_check_passed": True,
            "unsupported_claims": [],
            "conflicts": [],
            "web_results": [],
            "retry_count": len(used_tools),
            "need_human_review": False,
            "errors": [],
        }

    # 用 generator 生成结构化答案
    try:
        gen_result = generate_answer(question, all_docs)
        answer = gen_result.get("answer", "")
        citations = gen_result.get("citations", [])
    except Exception as e:
        log.warning(f"[FC] 生成答案失败: {e}，使用LLM直接回答")
        answer = generate_direct_answer(question)
        citations = []

    elapsed = _time.time() - t0
    tool_chain = " → ".join(used_tools) if used_tools else "直接回答"
    history.append(f"[FC] 答案生成完成, 总耗时 {elapsed:.1f}s, 工具链: {tool_chain}")

    return {
        "question": question,
        "collection_name": collection_name,
        "answer": answer,
        "citations": citations,
        "retrieved_docs": all_docs,
        "graded_docs": all_docs,
        "relevant_count": len(all_docs),
        "history": history,
        "current_step": "done",
        "hallucination_score": 1.0,
        "fact_check_passed": True,
        "unsupported_claims": [],
        "conflicts": [],
        "web_results": [],
        "retry_count": len(used_tools),
        "need_human_review": False,
        "errors": [],
    }
