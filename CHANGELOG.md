# Changelog

All notable changes to DeepRAG will be documented in this file.

## [0.2.9] - 2026-07-25

### Added
- Showcase website with interactive architecture visualizations (Mermaid.js, D3.js, Chart.js)
- Railway deployment configuration (`railway.json`, `Dockerfile.streamlit`)
- Qdrant Cloud seeding script (`scripts/seed_qdrant_cloud.py`)
- GitHub Actions CI pipeline (test, lint, security, docker)
- GitHub Pages auto-deployment for showcase
- Comprehensive README with audit truth table
- CONTRIBUTING.md, SECURITY.md, LICENSE

### Fixed
- CI pipeline: resolved lint (F821), security (dependency install), test (collection errors), docker (reranker timeout)
- `generator.py`: F821 undefined name 'log' → replaced with `print`
- `app.py`: F821 undefined name 'get_indexer_cached' → moved definition to module level
- Dockerfile: removed `reranker` dependency to avoid 19min build timeout
- README: fixed broken relative path links

### Changed
- Dependabot open-pull-requests-limit reduced from 10 to 3
- CI lint restricted to `src/` only (tests/ has legacy syntax issues)
- CI security tools installed separately with `|| true` fallback
- Docker build timeout set to 25 minutes

## [0.2.8] - 2026-07-20

### Added
- v2.8.2: ENABLE_RERANKER config fix (was always creating Reranker instance)
- Dynamic thinking mode (qwen3 series only)
- Model dynamic switching via `/api/tags`
- Ollama environment optimization (Flash Attention, KV Cache quantization, Keep Alive)

### Fixed
- ENABLE_RERANKER configuration not taking effect in graph.py
- 7B model latency reduced from 24.4s to 5.7s after disabling Reranker

## [0.2.1] - 2026-06-08

### Added
- Initial public release
- Agentic RAG with LangGraph
- Multi-vector-database support (ChromaDB, Qdrant, FAISS, LanceDB, pgvector)
- Hybrid retrieval (BM25 + vector + RRF)
- Self-RAG and Corrective RAG pipelines
- Offline heuristic evaluation engine
- Streamlit production UI (6-Tab)
