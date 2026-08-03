# DeepRAG 开发指南（DEVELOPMENT）

面向贡献者与二次开发者的工程手册。所有条目以仓库实际代码为准（参考 `CLAUDE.md`、`pyproject.toml`、`src/config.py`）。指标与跑分数字以 `docs/审计真相表_P0.md` 为准，本文不写死任何未复测的数值。

---

## 1. 开发环境搭建

### 1.1 Python 版本要求

- **Python >= 3.11**（`pyproject.toml` 中 `requires-python = ">=3.11"`）。
- 推荐使用虚拟环境（项目已带 `.venv/`，可复用或新建）：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 1.2 依赖安装

项目使用 `pyproject.toml` 管理依赖与可选分组（`[project.optional-dependencies]`）：

```bash
# 最小可运行（核心 + 开发工具）
pip install -e ".[dev]"

# 按需开启额外能力
pip install -e ".[dev,ui,api,llm,qdrant,reranker,observability]"
```

可选分组说明（以 `pyproject.toml` 为准）：

| 分组 | 用途 | 代表依赖 |
|------|------|----------|
| `llm` | LangChain 多后端（OpenAI/Ollama/Anthropic 等） | `langchain`, `langchain-openai`, `langchain-ollama` |
| `ui` | Streamlit Web 界面 | `streamlit>=1.40.0` |
| `api` | FastAPI HTTP 服务 | `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx` |
| `qdrant` | Qdrant 向量库（生产推荐后端） | `qdrant-client` |
| `pgvector` | PostgreSQL + pgvector | `psycopg2-binary`, `pgvector` |
| `reranker` | CrossEncoder 精排 | `sentence-transformers` |
| `dev` | 测试与质量 | `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `black` |
| `observability` | 链路追踪 / token 计费 | `langfuse`, `tiktoken` |

> 注：`chromadb` 为核心依赖，已写入 `[project].dependencies`，无需额外可选分组。

### 1.3 环境变量

复制模板并填写（请勿提交真实密钥）：

```bash
cp .env.example .env
```

关键开关（详见 `.env.example` 与 `src/config.py`）：

- `LLM_BACKEND`：LLM 后端，`auto` / `ollama` / `openai` / `anthropic` / `zhipu` / `none`（规则模式）。
- `VECTOR_DB`：向量库后端，默认 `qdrant`（可选 `chromadb` / `lancedb` / `faiss` / `pgvector`）。
- `ENABLE_SELF_RAG_LOOP`：Self-RAG 闭环，默认 `false`（fast 直出）；`true` 开启 regenerate。
- `ENABLE_AGENTIC_RAG`：是否启用 Agentic 路由。
- `ENABLE_RERANKER`：CrossEncoder 精排（7B 显存建议关）。
- `CHROMA_SERVER_HOST` / `CHROMA_SERVER_PORT`：ChromaDB 服务器模式地址（默认 `localhost:8000`）。
- `CHROMA_DB_PATH`：ChromaDB 数据目录。**仅用于 `chroma run --path` 启动服务器**，代码内禁止直连。

---

## 2. 项目目录结构

### 2.1 仓库顶层

```
deep-rag/
├── api.py                  # 根入口：from scripts.api import app（uvicorn api:app）
├── app.py                  # Streamlit Web UI 主入口
├── scripts/api.py          # FastAPI 生产服务实现
├── start_mcp_server.py     # MCP Server 启动入口
├── src/                    # 源码包（见 2.2）
├── evaluation/             # 离线评测集与脚本（golden / e2e_20）
├── tests/                  # pytest 用例
├── data/                   # 示例文档与数据
├── docs/                   # 项目文档
├── pyproject.toml          # 依赖与工具配置（ruff/black/mypy/pytest）
├── requirements.txt        # 备用纯依赖清单
├── .env.example            # 环境变量模板
└── README.md / CLAUDE.md   # 项目说明与工作指南
```

### 2.2 `src/` 各子包职责

> 数量随仓库演进变化，以实际文件为准。

| 子包 / 模块 | 职责 |
|-------------|------|
| `src/config.py` | 全局配置（后端开关、密钥、向量库客户端 `get_chroma_client()`）。 |
| `src/graph.py` | LangGraph 主状态机；对外提供 `query()` 与 `get_indexer()`。 |
| `src/agents/` | RAG 各阶段 Agent 节点（查询分析、文档评分、答案生成、事实校验、冲突解决等）。 |
| `src/retrieval/` | 检索层：混合检索（`hybrid.py`）、Reranker、Agentic 工具（`agentic_tools.py`）、路由器、Web 兜底、Qdrant/LanceDB 适配等。 |
| `src/evaluation/` | 离线启发式四指标评测（非官方 `ragas` 包，中文 jieba/bigram）。 |
| `src/intent/` | 查询意图识别。 |
| `src/llm/` | LLM 网关（`gateway.py`），统一多后端调用与 metrics。 |
| `src/observability/` | 可观测性：LangFuse 追踪、token 计费（按需启用）。 |
| `src/reliability/` | 可靠性：熔断器 `CircuitBreaker` 与降级策略（`degrade.py`）。 |
| `src/security/` | 安全：API 鉴权（`api_auth`）、限流（`rate_limiter`）、输入清洗（`input_guard`）、审计日志（`audit`）。 |
| `src/tools/` | 工具与集成：`mcp_server.py`（MCP 服务）、`modules/`（工具子模块）。 |
| `src/api/` | API 辅助：`sse_stream.py`（SSE 流式封装）。 |
| `src/ui/` | UI 辅助组件。 |

---

## 3. 本地运行方式

### 3.1 Streamlit Web UI

```bash
streamlit run app.py
# 默认 http://localhost:8501
# 或指定端口：streamlit run app.py --server.port 8501
```

### 3.2 FastAPI 服务

两种等价启动方式（根 `api.py` 仅转发 `scripts.api.app`，避免 import 路径分裂）：

```bash
# 方式 A（推荐，库内入口）
uvicorn api:app --host 0.0.0.0 --port 8000

# 方式 B（直接运行脚本）
python scripts/api.py
```

交互式文档：启动后访问 `http://localhost:8000/docs`（Swagger UI）与 `/redoc`。
端点清单见 [`API.md`](API.md)。

### 3.3 MCP Server

```bash
# stdio 模式（本地，Claude Desktop 默认）
python start_mcp_server.py
# 或：python -m src.tools.mcp_server

# Streamable HTTP 模式（远程）
python -m src.tools.mcp_server --transport http --port 8080
```

---

## 4. 运行测试

```bash
# 全量（部分用例依赖外部服务，以实际输出为准，勿写死通过率）
pytest

# 指定用例
pytest tests/test_self_rag_route.py -v

# 覆盖率
pytest tests/ --cov=src --cov-report=html
# 报告：htmlcov/index.html
```

测试配置见 `pyproject.toml` 的 `[tool.pytest.ini_options]`（`testpaths = ["tests"]`、`addopts = "-q"`）。

---

## 5. ChromaDB 安全规范（重要）

### 5.1 红线

> **禁止**使用 `chromadb.PersistentClient` 连接已有库路径。
> **必须**使用 ChromaDB 服务器模式 + `chromadb.HttpClient`。

### 5.2 原因

- ChromaDB 的 HNSW 索引由进程内 compactor 维护。若多个进程（或同进程多次）用 `PersistentClient` 直接打开**同一个**数据库目录，compactor 并发写同一文件会导致 **HNSW 索引损坏、库不可读**。
- 本项目历史上因此发生 **5 次**库损坏（`chroma_db_corrupted_*` 目录即为残骸），故在 `src/config.py::get_chroma_client()` 中明确封禁 `PersistentClient`。
- 服务器模式由唯一的 ChromaDB 进程持有数据目录，统一调度 compactor，从架构上避免并发写冲突。

### 5.3 正确用法

**第一步：先启动 ChromaDB 服务器**（独占持有数据目录）

```bash
# CHROMA_DB_PATH 仅用于此处；详见 .env.example
export CHROMA_DB_PATH=./chroma_data
chroma run --path "$CHROMA_DB_PATH" --port 8000
```

**第二步：业务代码只通过 `get_chroma_client()` 取 HttpClient**

```python
from src.config import get_chroma_client

client = get_chroma_client()          # 内部：chromadb.HttpClient(host=..., port=...)
collections = client.list_collections()
```

对应配置项（默认）：`CHROMA_SERVER_HOST=localhost`、`CHROMA_SERVER_PORT=8000`。

### 5.4 注意事项

- host 请用 `localhost`，不要与 `127.0.0.1` 混用（客户端/服务器解析一致性）。
- `CHROMA_DB_PATH` 仅供 `chroma run --path` 使用；**代码内任何位置都不得出现 `PersistentClient`**。
- 默认向量库后端为 `qdrant`（`VECTOR_DB=qdrant`）；仅当 `VECTOR_DB=chromadb` 时才会连 Chroma HttpClient。
- 新增任何向量库访问代码，必须通过 `src/config.py` 的客户端工厂获取连接，禁止自行 new 客户端。

---

## 6. 贡献与 Commit 规范

参考 `CLAUDE.md` 与 `.pre-commit-config.yaml`、`.github/` 工作流。

### 6.1 代码风格（自动校验）

- **格式化**：`black`（行宽 100），**导入顺序**：`isort`（profile=black）。
- **Lint**：`ruff`（见 `pyproject.toml` `[tool.ruff]`，行宽 100，目标 py311）。
- **类型 / 文档**：公开函数建议加类型提示与 docstring。
- 提交前可本地运行：

```bash
ruff check src tests
black --check src tests
```

### 6.2 日常流程

```bash
# 1. 改代码 → 2. 跑测试 → 3. 检查覆盖率 → 4. 提交
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

### 6.3 Commit 规范

- 采用 Conventional Commits 风格：`type: 简述`。
- 常用 `type`：`feat`（新功能）、`fix`（修复）、`docs`（文档）、`refactor`（重构）、`test`（测试）、`chore`（杂项）、`perf`（性能）。
- 示例：

```
feat: Multi-Agent 优化 v2.1 - 并行度 2.0x、成本 -40%、熔断器
fix: 修正 Corrective RAG 重试耗尽后未兜底
docs: 新增 DEVELOPMENT.md 与 API.md
```

- 涉及 ChromaDB / 向量库的改动，必须确认未引入 `PersistentClient` 直连。
- 仅做文档改动时标 `docs:`，不影响代码与测试。

### 6.4 诚实口径约定

- 指标、准确率、覆盖率等数字以实测输出与 `docs/审计真相表_P0.md` 为准；**不得写死未复测的跑分**（如历史报告中的 95% / 99.5%）。
- 历史实验数据仅作叙述，不与当前 `pytest` 结果混为一谈。
- 贡献文档时同样遵循「以项目实际代码为准，不臆造端点/工具名」。

---

## 7. 相关文档

- [`API.md`](API.md)：FastAPI 端点与 MCP tools 说明。
- `README.md`：项目总览与快速开始。
- `CLAUDE.md`：项目工作指南与开关清单。
- `docs/审计真相表_P0.md`：指标与入口的权威口径。
- `CONTRIBUTING.md`：贡献流程细则。
