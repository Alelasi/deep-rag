"""DeepRAG FastAPI — 生产向 HTTP 服务

端点：
- POST /query        — 单次 RAG 查询
- POST /query/stream — 流式 SSE 查询
- POST /index        — 索引文档目录（路径沙箱）
- GET  /health       — 存活
- GET  /ready        — 就绪
- GET  /metrics      — Prometheus 文本指标
- GET  /collections  — 已加载集合
- GET  /version      — 版本与能力开关

启动：
  uvicorn api:app --host 0.0.0.0 --port 8000
  # 或 python scripts/api.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# 保证项目根在 path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config import (
    ENABLE_AGENTIC_RAG,
    ENABLE_SELF_RAG_LOOP,
    PACKAGE_VERSION,
    CAPABILITY_VERSION,
    VECTOR_DB,
)
from src.graph import query as rag_query, get_indexer, _indexers
from src.security import (
    audit_log,
    get_rate_limiter,
    is_auth_enabled,
    is_jwt_enabled,
    sanitize_question,
    validate_index_path,
    verify_api_key,
    verify_jwt,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

START_TIME = time.time()
REQUEST_COUNT = {
    "query": 0,
    "query_stream": 0,
    "index": 0,
    "health": 0,
    "errors": 0,
}

# CORS：禁止默认 '*'。从 CORS_ALLOW_ORIGINS（兼容旧 CORS_ORIGINS）读取逗号分隔白名单；
# 缺省仅本地开发来源（不含 '*'）。
_cors_raw = os.getenv("CORS_ALLOW_ORIGINS", os.getenv("CORS_ORIGINS", "")).strip()
if _cors_raw:
    CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    # 安全默认：仅本地来源，避免任意站点跨域访问
    CORS_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]

# 仅在显式配置 '*' 时放开为通配符（并关闭凭据，否则 CORSMiddleware 会拒绝）
_allow_origins = ["*"] if "*" in CORS_ORIGINS else CORS_ORIGINS
_allow_credentials = "*" not in CORS_ORIGINS

app = FastAPI(
    title="DeepRAG API",
    description="生产向 RAG API：Corrective RAG + 可选 Self-RAG + 鉴权/限流/审计",
    version=PACKAGE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _security_startup_check() -> None:
    """启动时明确提示安全默认值状态。"""
    if not is_auth_enabled():
        logger.warning(
            "安全告警：未配置 API Key（DEEP_RAG_API_KEY），API 以开发模式运行，"
            "不做任何鉴权。生产环境请设置该变量以启用强制鉴权。"
        )
    else:
        logger.info("API 鉴权已启用（DEEP_RAG_API_KEY 已配置）。")
    if is_jwt_enabled():
        logger.info("可选 JWT 鉴权已启用。")
    if "*" in CORS_ORIGINS:
        logger.warning(
            "安全告警：CORS 允许通配符 '*'，存在跨站请求风险；生产请改用显式来源白名单。"
        )


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    collection_name: str = Field("default", description="知识库集合名")
    max_retries: int = Field(2, ge=0, le=5, description="Corrective 最大重试")


class Citation(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    hallucination_score: float
    fact_check_passed: bool
    relevant_count: int
    conflicts: list[dict]
    history: list[str]
    mode: str
    no_knowledge: bool = False
    used_mock_web: bool = False
    request_id: str = ""
    warnings: list[str] = []


class IndexRequest(BaseModel):
    collection_name: str = Field(..., min_length=1)
    docs_dir: str = Field(..., description="文档目录（须在 INDEX_ALLOWED_ROOTS 内）")


class IndexResponse(BaseModel):
    collection_name: str
    indexed_chunks: int
    request_id: str = ""


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def require_auth(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """鉴权 + 限流。返回 request_id。"""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    client = _client_ip(request)

    token = x_api_key or authorization
    if is_auth_enabled():
        # 优先 API Key；开启 JWT 时允许 Bearer JWT 通过
        authed = verify_api_key(token) or (is_jwt_enabled() and verify_jwt(token))
        if not authed:
            audit_log(
                "auth_denied",
                request_id=request_id,
                client=client,
                detail={"path": str(request.url.path)},
                level="warning",
            )
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    allowed, remaining = get_rate_limiter().allow(client)
    request.state.rate_remaining = remaining
    if not allowed:
        audit_log(
            "rate_limited",
            request_id=request_id,
            client=client,
            detail={"path": str(request.url.path)},
            level="warning",
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return request_id


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    rid = getattr(request.state, "request_id", None)
    if rid:
        response.headers["X-Request-Id"] = rid
    remaining = getattr(request.state, "rate_remaining", None)
    if remaining is not None:
        response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


@app.get("/health")
async def health():
    REQUEST_COUNT["health"] += 1
    return {
        "status": "healthy",
        "version": PACKAGE_VERSION,
        "capability_version": CAPABILITY_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(time.time() - START_TIME),
        "auth_enabled": is_auth_enabled(),
    }


@app.get("/version")
async def version():
    return {
        "package_version": PACKAGE_VERSION,
        "capability_version": CAPABILITY_VERSION,
        "vector_db": VECTOR_DB,
        "enable_agentic_rag": ENABLE_AGENTIC_RAG,
        "enable_self_rag_loop": ENABLE_SELF_RAG_LOOP,
        "auth_enabled": is_auth_enabled(),
    }


@app.get("/ready")
async def readiness():
    checks: Dict[str, Any] = {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }
    try:
        get_indexer("default")
        checks["checks"]["vector_db"] = {"status": "up", "type": VECTOR_DB}
    except Exception as e:
        checks["status"] = "not_ready"
        checks["checks"]["vector_db"] = {"status": "down", "error": str(e)[:200]}

    checks["checks"]["agentic_rag"] = {
        "status": "enabled" if ENABLE_AGENTIC_RAG else "disabled"
    }
    checks["checks"]["self_rag_loop"] = {
        "status": "enabled" if ENABLE_SELF_RAG_LOOP else "disabled"
    }
    if checks["status"] != "ready":
        raise HTTPException(status_code=503, detail=checks)
    return checks


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    uptime = int(time.time() - START_TIME)
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.05)
        memory = psutil.virtual_memory()
        mem_used = memory.used
        mem_pct = memory.percent
    except Exception:
        cpu_percent = 0.0
        mem_used = 0
        mem_pct = 0.0

    return f"""# HELP deeprag_uptime_seconds Application uptime in seconds
# TYPE deeprag_uptime_seconds gauge
deeprag_uptime_seconds {uptime}

# HELP deeprag_requests_total Total requests by endpoint
# TYPE deeprag_requests_total counter
deeprag_requests_total{{endpoint="query"}} {REQUEST_COUNT["query"]}
deeprag_requests_total{{endpoint="query_stream"}} {REQUEST_COUNT["query_stream"]}
deeprag_requests_total{{endpoint="index"}} {REQUEST_COUNT["index"]}
deeprag_requests_total{{endpoint="health"}} {REQUEST_COUNT["health"]}
deeprag_requests_total{{endpoint="errors"}} {REQUEST_COUNT["errors"]}

# HELP deeprag_cpu_usage_percent CPU usage percentage
# TYPE deeprag_cpu_usage_percent gauge
deeprag_cpu_usage_percent {cpu_percent}

# HELP deeprag_memory_usage_bytes Memory usage in bytes
# TYPE deeprag_memory_usage_bytes gauge
deeprag_memory_usage_bytes {mem_used}

# HELP deeprag_memory_usage_percent Memory usage percentage
# TYPE deeprag_memory_usage_percent gauge
deeprag_memory_usage_percent {mem_pct}

# HELP deeprag_info Application information
# TYPE deeprag_info gauge
deeprag_info{{version="{PACKAGE_VERSION}",capability="{CAPABILITY_VERSION}",vector_db="{VECTOR_DB}",agentic_rag="{ENABLE_AGENTIC_RAG}"}} 1
"""


@app.get("/collections")
async def list_collections(request_id: str = Depends(require_auth)):
    return {"collections": list(_indexers.keys()), "request_id": request_id}


@app.post("/index", response_model=IndexResponse)
async def index_docs(req: IndexRequest, request: Request, request_id: str = Depends(require_auth)):
    REQUEST_COUNT["index"] += 1
    client = _client_ip(request)
    try:
        docs_path = validate_index_path(req.docs_dir)
    except PermissionError as e:
        REQUEST_COUNT["errors"] += 1
        audit_log("index_denied", request_id=request_id, client=client, detail={"err": str(e)}, level="warning")
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        REQUEST_COUNT["errors"] += 1
        raise HTTPException(status_code=400, detail=str(e))

    try:
        indexer = get_indexer(req.collection_name)
        count = indexer.index_directory(str(docs_path))
    except Exception as e:
        REQUEST_COUNT["errors"] += 1
        logger.exception("Index failed")
        audit_log("index_error", request_id=request_id, client=client, detail={"err": str(e)}, level="error")
        raise HTTPException(status_code=500, detail="Index failed")

    audit_log(
        "index_ok",
        request_id=request_id,
        client=client,
        detail={"collection": req.collection_name, "chunks": count, "path": str(docs_path)},
    )
    return IndexResponse(
        collection_name=req.collection_name,
        indexed_chunks=count,
        request_id=request_id,
    )


def _to_citations(raw: List[Any]) -> List[Citation]:
    out: List[Citation] = []
    for c in raw or []:
        if isinstance(c, dict):
            out.append(
                Citation(
                    source=c.get("source"),
                    page=c.get("page"),
                    text=c.get("text"),
                )
            )
    return out


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request, request_id: str = Depends(require_auth)):
    REQUEST_COUNT["query"] += 1
    client = _client_ip(request)
    try:
        question, warnings = sanitize_question(req.question)
    except ValueError as e:
        REQUEST_COUNT["errors"] += 1
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = rag_query(
            question,
            collection_name=req.collection_name,
            max_retries=req.max_retries,
        )
    except Exception as e:
        REQUEST_COUNT["errors"] += 1
        logger.exception("Query failed")
        audit_log(
            "query_error",
            request_id=request_id,
            client=client,
            detail={"err": str(e)[:300]},
            level="error",
        )
        raise HTTPException(status_code=500, detail="Query failed")

    audit_log(
        "query_ok",
        request_id=request_id,
        client=client,
        detail={
            "collection": req.collection_name,
            "no_knowledge": bool(result.get("no_knowledge")),
            "hallucination_score": result.get("hallucination_score"),
        },
    )

    return QueryResponse(
        question=question,
        answer=result.get("answer", ""),
        citations=_to_citations(result.get("citations", [])),
        hallucination_score=float(result.get("hallucination_score") or 0.0),
        fact_check_passed=bool(result.get("fact_check_passed", True)),
        relevant_count=int(result.get("relevant_count") or 0),
        conflicts=result.get("conflicts") or [],
        history=result.get("history") or [],
        mode="agentic" if ENABLE_AGENTIC_RAG else "hybrid",
        no_knowledge=bool(result.get("no_knowledge")),
        used_mock_web=bool(result.get("used_mock_web")),
        request_id=request_id,
        warnings=warnings,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request, request_id: str = Depends(require_auth)):
    REQUEST_COUNT["query_stream"] += 1
    client = _client_ip(request)
    try:
        question, warnings = sanitize_question(req.question)
    except ValueError as e:
        REQUEST_COUNT["errors"] += 1
        raise HTTPException(status_code=400, detail=str(e))

    def event_stream():
        try:
            if warnings:
                yield f"data: {json.dumps({'type': 'warning', 'content': warnings, 'request_id': request_id}, ensure_ascii=False)}\n\n"
            result = rag_query(
                question,
                collection_name=req.collection_name,
                max_retries=req.max_retries,
            )
            for step in result.get("history", []):
                yield f"data: {json.dumps({'type': 'step', 'content': step}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'content': result.get('answer', ''), 'no_knowledge': bool(result.get('no_knowledge'))}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'request_id': request_id})}\n\n"
            audit_log("query_stream_ok", request_id=request_id, client=client, detail={})
        except Exception as e:
            REQUEST_COUNT["errors"] += 1
            logger.exception("Stream query failed")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Query failed', 'request_id': request_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    REQUEST_COUNT["errors"] += 1
    rid = getattr(request.state, "request_id", "")
    logger.exception("Unhandled error rid=%s", rid)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": rid},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
