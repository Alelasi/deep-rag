"""DeepRAG FastAPI - 对外 HTTP 服务

提供 REST API 接口：
- POST /query        — 单次 RAG 查询
- POST /query/stream — 流式 SSE 查询
- POST /index        — 索引文档目录
- GET  /health       — 健康检查
- GET  /collections  — 列出已索引的集合

启动：uvicorn api:app --host 0.0.0.0 --port 8000
"""
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.graph import query as rag_query, get_indexer, _indexers
from src.config import ENABLE_AGENTIC_RAG, VECTOR_DB

log = logging.getLogger("deeprag.api")

app = FastAPI(
    title="DeepRAG API",
    description="自纠错多源知识 Agent — Corrective RAG + Self-RAG + Agentic RAG",
    version="2.0.0",
)

# CORS（开发环境放开，生产环境需收紧）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === 请求/响应模型 ===

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    collection_name: str = Field("default", description="知识库集合名")
    max_retries: int = Field(2, ge=0, le=5, description="最大重试次数")


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
    mode: str  # "hybrid" or "agentic"


class IndexRequest(BaseModel):
    collection_name: str = Field(..., min_length=1)
    docs_dir: str = Field(..., description="文档目录绝对路径")


class IndexResponse(BaseModel):
    collection_name: str
    indexed_chunks: int


# === 端点 ===

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "agentic_rag_enabled": ENABLE_AGENTIC_RAG,
        "vector_db": VECTOR_DB,
    }


@app.get("/collections")
async def list_collections():
    """列出已加载的知识库集合"""
    return {"collections": list(_indexers.keys())}


@app.post("/index", response_model=IndexResponse)
async def index_docs(req: IndexRequest):
    """索引指定目录的文档到知识库"""
    docs_path = Path(req.docs_dir)
    if not docs_path.exists():
        raise HTTPException(status_code=400, detail=f"目录不存在: {req.docs_dir}")
    if not docs_path.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录: {req.docs_dir}")

    indexer = get_indexer(req.collection_name)
    count = indexer.index_directory(str(docs_path))
    log.info(f"Indexed {count} chunks into collection '{req.collection_name}'")

    return IndexResponse(collection_name=req.collection_name, indexed_chunks=count)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """执行一次 RAG 查询"""
    try:
        result = rag_query(
            req.question,
            collection_name=req.collection_name,
            max_retries=req.max_retries,
        )
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=req.question,
        answer=result.get("answer", ""),
        citations=[Citation(**c) for c in result.get("citations", [])],
        hallucination_score=result.get("hallucination_score", 0.0),
        fact_check_passed=result.get("fact_check_passed", True),
        relevant_count=result.get("relevant_count", 0),
        conflicts=result.get("conflicts", []),
        history=result.get("history", []),
        mode="agentic" if ENABLE_AGENTIC_RAG else "hybrid",
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """流式 SSE 查询 — 按 Pipeline 步骤推送中间状态"""
    def event_stream():
        try:
            # 同步执行，每步产出一个 event（简化版：完成后一次性返回所有 history）
            result = rag_query(
                req.question,
                collection_name=req.collection_name,
                max_retries=req.max_retries,
            )
            # 推送各步骤历史
            for step in result.get("history", []):
                yield f"data: {json.dumps({'type': 'step', 'content': step}, ensure_ascii=False)}\n\n"
            # 最终答案
            yield f"data: {json.dumps({'type': 'answer', 'content': result.get('answer', '')}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            log.exception("Stream query failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
