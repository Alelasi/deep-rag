"""SSE流式接口 — v2.9.1新增

标准Server-Sent Events协议，支持跨进程/跨服务调用。

Web Interface Guidelines合规：
- Content-Type: text/event-stream
- Cache-Control: no-cache
- Connection: keep-alive
- CORS: Access-Control-Allow-Origin
- 错误处理：连接中断时返回明确的错误事件

用法：
    # 启动SSE服务器
    python -m src.api.sse_stream

    # 或在代码中调用
    from src.api.sse_stream import start_sse_server
    start_sse_server(port=8080)

    # 客户端请求
    curl -N "http://localhost:8080/api/stream?q=INTJ的主导功能&collection=mbti"
"""
import json
import logging
from typing import Optional

log = logging.getLogger("deeprag")

# 懒加载FastAPI（可选依赖）
_app = None


def _create_app():
    """创建FastAPI应用（懒加载）"""
    global _app
    if _app is not None:
        return _app

    try:
        from fastapi import FastAPI, Query
        from fastapi.middleware.cors import CORSMiddleware
        from sse_starlette.sse import EventSourceResponse
    except ImportError as e:
        log.error(f"[SSE] FastAPI或sse-starlette未安装: {e}")
        log.error("[SSE] 请安装: pip install fastapi sse-starlette uvicorn")
        raise

    app = FastAPI(title="DeepRAG SSE API", version="2.9.1")

    # CORS：允许跨域（开发环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制域名
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        """健康检查端点"""
        return {
            "status": "ok",
            "version": "2.9.1",
            "service": "DeepRAG SSE API",
        }

    @app.get("/api/stream")
    async def stream_query(
        q: str = Query(..., description="用户问题"),
        collection: str = Query("default", description="知识库集合名"),
        mode: str = Query("enhanced", description="检索模式: enhanced/agentic/hybrid/function_calling"),
    ):
        """SSE流式查询接口

        返回标准SSE格式：
        data: {"type":"token","content":"..."}\\n\\n
        data: {"type":"metadata","citations":[...]}\\n\\n
        data: [DONE]\\n\\n
        """
        async def event_generator():
            try:
                # 导入graph（延迟导入避免循环依赖）
                from src.graph import stream_query

                for chunk in stream_query(q, collection_name=collection, mode=mode):
                    yield {
                        "event": "message",
                        "data": json.dumps(chunk, ensure_ascii=False),
                    }

                # 发送结束标记
                yield {"event": "done", "data": "[DONE]"}

            except Exception as e:
                log.error(f"[SSE] 流式查询错误: {e}")
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"error": str(e), "type": "stream_error"},
                        ensure_ascii=False,
                    ),
                }
                yield {"event": "done", "data": "[DONE]"}

        return EventSourceResponse(event_generator())

    @app.get("/api/skills")
    async def list_skills():
        """列出已注册的Skill（v2.9.1）"""
        try:
            from src.tools.skill_system import get_skill_registry
            registry = get_skill_registry()
            metadata_list = registry.list_metadata()
            return {
                "skills": [
                    {
                        "name": m.name,
                        "description": m.description,
                        "risk": m.risk,
                        "version": m.version,
                    }
                    for m in metadata_list
                ],
                "total": len(metadata_list),
            }
        except Exception as e:
            return {"error": str(e), "skills": [], "total": 0}

    @app.get("/api/metrics")
    async def get_metrics():
        """获取LLM Gateway调用统计（v2.9.1）"""
        try:
            from src.llm.gateway import get_gateway
            gateway = get_gateway()
            return gateway.get_metrics()
        except Exception as e:
            return {"error": str(e)}

    _app = app
    return app


def get_app():
    """获取FastAPI应用实例"""
    return _create_app()


def start_sse_server(port: int = 8080, host: str = "0.0.0.0"):
    """启动SSE服务器

    Args:
        port: 监听端口
        host: 监听地址
    """
    app = _create_app()

    try:
        import uvicorn
    except ImportError:
        log.error("[SSE] uvicorn未安装，请运行: pip install uvicorn")
        raise

    log.info(f"[SSE] 启动SSE服务器: http://{host}:{port}")
    log.info(f"[SSE] 流式接口: http://{host}:{port}/api/stream?q=测试")
    log.info(f"[SSE] 健康检查: http://{host}:{port}/api/health")

    uvicorn.run(app, host=host, port=port, log_level="info")


def stream_sse(question: str, collection_name: str = "default", mode: str = "enhanced"):
    """将generator输出格式化为SSE字符串（非HTTP，用于内嵌场景）

    Yields:
        SSE格式的字符串: "data: {...}\\n\\n"
    """
    from src.graph import stream_query

    for chunk in stream_query(question, collection_name=collection_name, mode=mode):
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    # 直接运行: python -m src.api.sse_stream
    import argparse
    parser = argparse.ArgumentParser(description="DeepRAG SSE Server")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    args = parser.parse_args()

    start_sse_server(port=args.port, host=args.host)
