"""Ollama 原生 API 工具 — 动态思考模式开关

v2.8.2: 支持前端动态切换思考模式（简单问题关闭/复杂问题开启）。
- think=False: 禁用思考，速度快（适合简单问答）
- think=True: 启用思考，质量高（适合复杂推理）
langchain_ollama 的 ChatOllama 不支持 think 参数，必须用原生 API。
"""
import json as _json
import urllib.request
import logging
import os

log = logging.getLogger(__name__)

# 模块级思考模式开关（可通过 set_think_mode() 或环境变量 OLLAMA_THINK 控制）
_think_enabled = os.getenv("OLLAMA_THINK", "false").lower() == "true"


def set_think_mode(enabled: bool):
    """设置全局思考模式开关（由前端 UI 调用）"""
    global _think_enabled
    _think_enabled = enabled
    os.environ["OLLAMA_THINK"] = "true" if enabled else "false"
    log.info(f"[Ollama] Think mode: {'ON' if enabled else 'OFF'}")


def get_think_mode() -> bool:
    """获取当前思考模式状态"""
    return _think_enabled


def ollama_chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    num_predict: int = 300,
    think: bool = None,
    timeout: int = 60,
) -> tuple[str, str]:
    """调用 Ollama 原生 /api/chat 接口

    Args:
        messages: [{"role": "system/user/assistant", "content": "..."}]
        model: 模型名称，如 "qwen3:4b"
        temperature: 温度
        num_predict: 最大生成 token 数
        think: 是否启用思考模式（None=使用全局开关，True=强制开启，False=强制关闭）
        timeout: 超时秒数

    Returns:
        (content, thinking) — content 是最终答案，thinking 是思考过程（如果启用）
    """
    # think=None 时使用全局开关
    if think is None:
        think = _think_enabled

    # 非 qwen3 模型不支持 think 参数，忽略
    if "qwen3" not in model:
        think = False

    payload = _json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict if not think else 800,  # 思考模式需要更多token
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = _json.loads(resp.read())

    msg = data.get("message", {})
    content = msg.get("content", "")
    thinking = msg.get("thinking", "")
    return content, thinking


def ollama_chat_or_fallback(
    messages: list[dict],
    model: str,
    llm,
    temperature: float = 0.3,
    num_predict: int = 300,
    think: bool = False,
) -> str:
    """调用 Ollama 原生 API，如果失败则回退到 langchain

    用于 doc_grader 等需要兼容多种后端的场景。
    """
    from src.config import LLM_BACKEND, LLM_MODEL

    if LLM_BACKEND == "ollama":
        try:
            ollama_messages = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                              for m in messages]
            content, thinking = ollama_chat(
                ollama_messages, LLM_MODEL,
                temperature=temperature,
                num_predict=num_predict,
                think=think,
            )
            if content and len(content.strip()) >= 5:
                return content
            elif thinking:
                lines = [l.strip() for l in thinking.strip().split("\n") if l.strip()]
                return "\n".join(lines[-3:]) if lines else ""
            else:
                log.warning("[Ollama] Empty response, falling back to langchain")
        except Exception as e:
            log.warning(f"[Ollama] Native API failed: {e}, falling back to langchain")

    # 回退到 langchain
    from langchain_core.messages import HumanMessage, SystemMessage
    lc_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    response = llm.invoke(lc_messages)
    return response.content if hasattr(response, "content") else str(response)
