# DeepRAG API 文档（API.md）

本文档以 `scripts/api.py`（FastAPI 生产服务）与 `src/tools/mcp_server.py`（MCP 服务）的实际代码为准。端点与字段均取自源码，未臆造。

服务启动见 [`DEVELOPMENT.md`](DEVELOPMENT.md#3-本地运行方式)。根入口 `api.py` 仅转发 `scripts.api.app`：

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

交互式文档：`/docs`（Swagger UI）、`/redoc`。

---

## 1. 鉴权与限流

- 由 `require_auth` 依赖统一处理，作用于受保护端点（`/query`、`/query/stream`、`/index`、`/collections`）。
- 鉴权开关由 `src.security.is_auth_enabled()` 控制；开启后需在请求头携带：
  - `X-API-Key: <key>`，或
  - `Authorization: <token>`。
- 未通过返回 `401`；密钥缺失/无效由 `src.security.verify_api_key()` 判定。
- 限流由 `src.security.get_rate_limiter().allow(client)` 控制；超限返回 `429`，响应头带 `X-RateLimit-Remaining`。
- 所有受保护请求均经 `audit_log` 记录，并在响应头回写 `X-Request-Id`。
- 安全响应头（中间件 `add_security_headers`）：`X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer`。

CORS：由环境变量 `CORS_ORIGINS` 控制（逗号分隔，默认含 `http://localhost:8501` 与 `*`）。

---

## 2. FastAPI 端点

### 2.1 概览

| 方法 | 路径 | 受保护 | 说明 |
|------|------|--------|------|
| GET | `/health` | 否 | 存活探针 |
| GET | `/ready` | 否 | 就绪探针（检查向量库等） |
| GET | `/metrics` | 否 | Prometheus 文本指标 |
| GET | `/version` | 否 | 版本与能力开关 |
| GET | `/collections` | 是 | 已加载集合列表 |
| POST | `/index` | 是 | 索引文档目录（路径沙箱） |
| POST | `/query` | 是 | 单次 RAG 查询 |
| POST | `/query/stream` | 是 | 流式 SSE RAG 查询 |

---

### 2.2 GET `/health`

存活探针，无需鉴权。

**响应字段**（JSON）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定 `"healthy"` |
| `version` | string | 包版本（`PACKAGE_VERSION`） |
| `capability_version` | string | 能力版本（`CAPABILITY_VERSION`） |
| `timestamp` | string | UTC ISO8601 时间 |
| `uptime_seconds` | int | 进程启动至今秒数 |
| `auth_enabled` | bool | 鉴权是否开启 |

---

### 2.3 GET `/ready`

就绪探针；向量库不可用时返回 `503`。

**响应字段**（JSON）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ready"` 或 `"not_ready"` |
| `timestamp` | string | UTC ISO8601 时间 |
| `checks` | object | 各依赖检查结果 |
| `checks.vector_db` | object | `{status: "up"|"down", type: <VECTOR_DB>}`；down 时含 `error` |
| `checks.agentic_rag` | object | `{status: "enabled"|"disabled"}` |
| `checks.self_rag_loop` | object | `{status: "enabled"|"disabled"}` |

---

### 2.4 GET `/metrics`

Prometheus 文本格式（`Content-Type: text/plain`）。需安装 `psutil` 才有 CPU/内存指标（缺失时填 0）。

**暴露指标**：

| 指标 | 类型 | 说明 |
|------|------|------|
| `deeprag_uptime_seconds` | gauge | 进程运行时长 |
| `deeprag_requests_total{endpoint=...}` | counter | 分端点请求计数（query/query_stream/index/health/errors） |
| `deeprag_cpu_usage_percent` | gauge | CPU 使用率 |
| `deeprag_memory_usage_bytes` | gauge | 内存使用字节 |
| `deeprag_memory_usage_percent` | gauge | 内存使用率 |
| `deeprag_info{...}` | gauge | 版本/向量库/agentic_rag 信息（固定值 1） |

---

### 2.5 GET `/version`

版本与能力开关（无需鉴权，但属只读信息）。

**响应字段**（JSON）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `package_version` | string | 包版本 |
| `capability_version` | string | 能力版本 |
| `vector_db` | string | 当前向量库后端（`VECTOR_DB`） |
| `enable_agentic_rag` | bool | Agentic RAG 开关 |
| `enable_self_rag_loop` | bool | Self-RAG 闭环开关 |
| `auth_enabled` | bool | 鉴权开关 |

---

### 2.6 GET `/collections`

返回已加载的索引器集合名。需鉴权。

**响应字段**（JSON）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `collections` | list[string] | 已 `get_indexer()` 加载的集合名（`_indexers` 的 keys） |
| `request_id` | string | 请求 ID |

---

### 2.7 POST `/index`

索引指定文档目录。**路径受沙箱约束**：`docs_dir` 必须在 `INDEX_ALLOWED_ROOTS` 内，否则 `validate_index_path()` 抛 `PermissionError`（403）或 `ValueError`（400）。

**请求体**（`IndexRequest`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `collection_name` | string | 是 | 目标集合名（非空） |
| `docs_dir` | string | 是 | 文档目录（须在允许根目录内） |

**响应字段**（`IndexResponse`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `collection_name` | string | 集合名 |
| `indexed_chunks` | int | 成功索引的文档块数 |
| `request_id` | string | 请求 ID |

**错误**：路径越权 `403`；参数非法 `400`；索引失败 `500`（均带审计日志）。

---

### 2.8 POST `/query`

单次 RAG 查询，返回完整结构化结果。需鉴权。

**请求体**（`QueryRequest`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 用户问题（非空） |
| `collection_name` | string | 否 | 集合名，默认 `"default"` |
| `max_retries` | int | 否 | Corrective 最大重试（0–5，默认 2） |

**响应字段**（`QueryResponse`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 清洗后的问题 |
| `answer` | string | 生成的答案 |
| `citations` | list[Citation] | 引用（`source` / `page` / `text`，均可空） |
| `hallucination_score` | float | 幻觉分数（0–1，越低越可信） |
| `fact_check_passed` | bool | 事实校验是否通过 |
| `relevant_count` | int | 相关文档数 |
| `conflicts` | list[dict] | 冲突检测结果 |
| `history` | list[string] | 处理步骤历史 |
| `mode` | string | `"agentic"` 或 `"hybrid"` |
| `no_knowledge` | bool | 是否判定无相关知识 |
| `used_mock_web` | bool | 是否使用了 mock Web 兜底 |
| `request_id` | string | 请求 ID |
| `warnings` | list[string] | 输入清洗警告 |

**错误**：问题非法 `400`；查询失败 `500`。

---

### 2.9 POST `/query/stream`

流式 SSE 查询，事件流格式（`text/event-stream`，每行 `data: <json>\n\n`）。需鉴权。请求体与 `/query` 相同（`QueryRequest`）。

**事件类型**（`type` 字段）：

| type | 含义 | 关键字段 |
|------|------|----------|
| `warning` | 输入清洗警告 | `content`（警告列表）、`request_id` |
| `step` | 处理步骤 | `content`（单步文本） |
| `answer` | 最终答案 | `content`（答案）、`no_knowledge` |
| `done` | 结束 | `request_id` |
| `error` | 异常 | `message`、`request_id` |

---

## 3. MCP Tools 说明

服务实现：`src/tools/mcp_server.py`（协议版本 `2025-03-26`）。通信方式：

- **stdio**（默认，本地 / Claude Desktop）；
- **Streamable HTTP**（单端点 `/mcp`，`--transport http --port 8080`）；Agent Card 在 `/.well-known/agent-card.json`。

MCP 三类能力：**Tools**（有副作用）、**Resources**（只读）、**Prompts**（模板）。底层为 JSON-RPC 2.0，由 `MCPServer.handle_request` 分发（`initialize` / `tools/list` / `tools/call` / `resources/list` / `resources/read` / `prompts/list` / `prompts/get` / `ping`）。

### 3.1 Tools（有副作用操作）

| 工具名 | 说明 | 必填参数 | 可选参数 |
|--------|------|----------|----------|
| `vector_search` | 向量语义检索，从知识库搜索相关文档块 | `query` | `collection_name`（默认 `default`）、`top_k`（默认 5） |
| `exact_match` | 精确匹配查询（SQLite），查特定条目/定义/术语 | `query` | `collection_name`（默认 `default`） |
| `graph_search` | 知识图谱查询，检索实体间关系 | `entity` | `relation`（默认 `""`） |
| `web_search` | 网络搜索（DuckDuckGo），获取实时信息 | `query` | `max_results`（默认 3） |
| `rag_query` | 完整 RAG 问答（分析→检索→研究→校验→生成） | `question` | `collection_name`（默认 `default`）、`mode`（`simple`/`smart`/`expanded`/`precision`，默认 `smart`） |

工具执行器 `ToolExecutor` 实际调用 `src.retrieval.agentic_tools`（`VectorSearchTool`/`ExactMatchTool`/`GraphSearchTool`/`WebSearchTool`）与 `src.graph.query`。结果为 JSON 文本（`content[].text`）。

### 3.2 Resources（只读数据）

| URI | 说明 | 提供内容 |
|-----|------|----------|
| `deeprag://collections` | 知识库集合列表 | 各集合 `name` 与 `count` |
| `deeprag://config` | 系统配置（不含敏感密钥） | LLM 后端/模型、embedding、reranker/web_fallback 开关 |
| `deeprag://stats` | 系统统计 | 经 `src.llm.gateway.get_metrics()` 的调用/延迟/缓存指标 |

### 3.3 Prompts（提示词模板）

| 模板名 | 参数 | 说明 |
|--------|------|------|
| `rag_answer` | `question`（必填）、`context`（必填）、`style`（可选：concise/detailed/analytical） | RAG 问答标准模板（含引用规范） |
| `fact_check` | `claim`（必填）、`evidence`（必填） | 事实核查模板 |
| `code_review` | `code`（必填）、`language`（可选，默认 python） | 代码审查模板 |

### 3.4 MCP 快速验证

```bash
# stdio（本地）
python -m src.tools.mcp_server

# Streamable HTTP
python -m src.tools.mcp_server --transport http --port 8080
# 能力声明：curl http://localhost:8080/.well-known/agent-card.json
```

Claude Desktop 配置示例（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "deeprag": {
      "command": "python",
      "args": ["-m", "src.tools.mcp_server"]
    }
  }
}
```

---

## 4. 相关文档

- [`DEVELOPMENT.md`](DEVELOPMENT.md)：环境搭建、目录结构、ChromaDB 安全规范、贡献规范。
- `README.md`：项目总览。
- `CLAUDE.md`：开关清单与工作指南。
