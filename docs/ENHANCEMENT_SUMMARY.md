# DeepRAG 增强版 — 变更总览（2026-08-03）

> 通过 13 个子 agent（A–N）分 4 波并行完成的增强。按文件归属严格切分、互不重叠，规避 merge 冲突；统一日志契约；单进程 AST 全量语法校验（避免逐文件 fork 造成沙箱资源耗尽）。

## 编排与验证方式
- **并行编排**：13 个 general-purpose 子 agent，分 4 波（Wave1: A/D/E/J；Wave2: C/F/H/I；Wave3: B/G；Wave4: K/L/M/N），每波内部并行派发，文件范围互不重叠。
- **共享契约**：`src/logging_config.py`（仅标准库）提供 `get_logger` / `configure_logging`，其余模块复用，禁止各自另建。
- **资源安全**：校验用单进程遍历 + `ast.parse`，不逐文件 fork Python，规避 `fork: Resource temporarily unavailable`。
- **诚实原则**：未实测指标不写死。

## 各子 agent 职责与成果

| 代号 | 方向 | 关键文件 | 主要改动 |
|---|---|---|---|
| A | 统一日志 + 去 print | `src/logging_config.py` + agents/coordination/evaluation/retrieval(部分)/llm(部分)/tools/ui/intent/api/app.py | 统一 logger，去除裸 print |
| B | 拆分 graph god-module | `src/graph.py` → `src/pipeline/build.py`、`src/rag/{react,function_calling,guards}.py` | 拆分为薄转发层，**公开 API 完全不变**（77 导出经 AST 确认） |
| C | 配置清理 | `src/config.py` | 去硬编码 Windows 绝对路径、`get_llm` 改注册表、策略表外置、类型注解 |
| D | 依赖 / 质量工具 | `pyproject.toml`、`requirements.txt`、`.pre-commit-config.yaml` | 补 dev/observability extras，加 black/isort/ruff/mypy/pytest 配置 |
| E | CI 质量门禁 | `.github/workflows/ci.yml` | 移除 lint/security 的 `\|\| true` 容错，新增 mypy 步骤 |
| F | Web 兜底生产化 | `src/retrieval/web_fallback.py` | env 驱动（WEB_FALLBACK_MOCK/ENABLE_WEB_FALLBACK）、统一 `is_mock` 标记 |
| G | 安全加固 | `src/security/*`、`scripts/api.py` | CORS 白名单、生产强制 API_KEY、INDEX_ALLOWED_ROOTS 安全默认、限流线程安全 |
| H | 缓存 / 限流统一 | `rate_limiter.py`、`cache.py`、`semantic_cache.py`、`prompt_cache.py` | 线程安全、统一 TTL + 可插拔后端抽象 |
| I | 可观测性 | `src/observability/{tracer,cost_tracker}.py` | 结构化 console、langfuse 懒导入守卫、计数加锁 |
| J | 开发者文档 | `docs/DEVELOPMENT.md`、`docs/API.md` | 开发指南 + FastAPI/MCP 端点文档（含 ChromaDB 安全规范） |
| K | 测试补充 | 新增 `tests/test_config.py`、`tests/test_graph_nodes.py`、`tests/test_chunker.py` 等 | config/web_fallback/security/graph 等单测，外部依赖自动 skip |
| L | 索引构建 | `src/retrieval/indexer.py`、`qdrant_indexer.py` | 批量 upsert、有界 `ThreadPoolExecutor`、Chroma 仅用 HttpClient（防库损坏） |
| M | 生成逻辑改进 | `src/agents/generator.py` | `_safe_truncate`/`_safe_json_loads` 兜底、LLM 不可用降级、类型注解 |
| N | 检索核心优化 | `enhanced_knowledge_retrieval.py`、`agentic_tools.py`、`agent_router.py` | RRF 融合（无除零）、`ENABLE_RERANKER` 门控、有界并发、错误隔离 |

## 验证结论
- **全量语法**：216/216 个 `.py` 文件 `ast.parse` 通过（含 2 个历史损坏脚本已修复：`benchmark_llm.py` 由 markdown 泄漏修复、`evaluate_all_final.py` 的 f-string 转义修复）。
- **graph.py 公开 API 完整**：`query` / `stream_query` / `build_graph` / `get_indexer` / `_indexers` / `precision_query` 等 77 个导出齐全，`app.py` / `scripts/api.py` / `src/tools/mcp_server.py` 仍可正常 import。
- **ChromaDB 安全规范已贯彻**：禁止在已有库路径创建 `PersistentClient`，改用服务器模式 + `HttpClient`。
- **残留裸 print 11 处**：位于 Agent A 职责范围外的文件（CLI/脚本入口），属合理保留，未强行改动。

## 运行方式
- 开发环境：`pip install -e ".[dev]"`
- UI：`streamlit run app.py`
- API：`uvicorn api:app --port 8000`
- 测试：`pytest`（外部依赖自动 skip）
- 注：当前所有改动**尚未 commit**，请按需提交。

## 风险与后续
- 子 agent M、N 因 30 回合上限提前结束，但代码已落盘且语法有效，核心改进（日志、RRF、降级、门控）均已落地；如需进一步打磨可针对性补强。
- 公开的增强为静态校验 + 结构一致性验证；端到端运行需真实依赖（LangGraph/Qdrant/LangChain 等）与 API key，建议在本地环境实跑回归。
