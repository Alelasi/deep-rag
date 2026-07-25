# reliability

轻量可靠性组件（**不是**完整 SRE / 99.5% SLA 平台）。

| 模块 | 作用 |
|------|------|
| `degrade.py` | 进程内熔断器 + LLM 不可用时的标准降级答案 |

## 与其它模块分工

| 能力 | 代码位置 |
|------|----------|
| 多 LLM 切换降级 | `src.config.get_llm_with_fallback` |
| 模型路由包装 | `src/llm/model_router*.py` |
| Multi-Agent 重试/熔断 | `src/agents/coordination/retry_handler.py`（**未挂入主 graph**） |
| LLM HTTP 限流重试 | `src/llm/rate_limiter.py` |
| API 鉴权/限流/审计 | `src/security/*` + `scripts/api.py` |
| 健康检查 | `GET /health` `/ready` `/metrics` |

## 简历表述

✅「实现进程内熔断与上游不可用降级模板；多模型 fallback 与 API 限流分离」  
❌「三层降级已保证 99.5% 生产可用性」（无压测与生产证据，勿写）
