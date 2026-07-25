# 免费 LLM 模型横向对比报告 v3

- 时间：2026-07-18T13:02:46
- 模型数：7 · 每模型 6 题（知识/数学/代码/中文/JSON/排错）
- 评分：关键词+结构+中文+代码+指令+延迟 规则启发式；失败计 0；**非人工金标**
- 免费档易 429：本报告仅收录完整 6/6 或已补测完成的结果
- 模型 ID 以 2026-07-18 各平台 `/models` 实测为准

## 综合排名

| 排名 | 模型 | 厂商 | 综合分 | 成功率 | 平均延迟 |
|-----:|------|------|-------:|-------:|---------:|
| 1 | Cerebras GPT-OSS-120B | Cerebras | **8.82** | 6/6 | 805ms |
| 2 | Groq Llama3.1-8B | Groq | **8.64** | 6/6 | 1071ms |
| 3 | Zhipu GLM-4-Flash | Zhipu | **8.09** | 6/6 | 15274ms |
| 4 | Silicon GLM-Z1-9B | SiliconFlow | **7.95** | 6/6 | 15114ms |
| 5 | Groq Qwen3.6-27B | Groq | **7.92** | 6/6 | 1942ms |
| 6 | Silicon Qwen2.5-7B | SiliconFlow | **7.15** | 6/6 | 2985ms |
| 7 | Zhipu GLM-4.5-Flash | Zhipu | **5.36** | 6/6 | 18709ms |

## 分厂商最佳

- **Cerebras**: Cerebras GPT-OSS-120B · 分 8.82 · 805ms
- **Groq**: Groq Llama3.1-8B · 分 8.64 · 1071ms
- **Zhipu**: Zhipu GLM-4-Flash · 分 8.09 · 15274ms
- **SiliconFlow**: Silicon GLM-Z1-9B · 分 7.95 · 15114ms

## 结论与 DeepRAG 建议

| 场景 | 推荐 | 理由 |
|------|------|------|
| **极速演示** | Cerebras GPT-OSS-120B / Groq Llama3.1-8B | 延迟约 0.8–1.1s |
| **中文问答** | Groq Llama3.1-8B / Silicon GLM-Z1-9B / Zhipu Flash | 中文题更稳 |
| **当前默认（本机）** | Silicon GLM-Z1-9B | 与 DeepRAG 现配置一致，质量尚可但偏慢(约 15s) |
| **降级链** | Groq → Cerebras → Silicon → Zhipu | 多厂商冗余，避开单家 429 |

## 原始数据

- JSON：`docs/free_model_benchmark_data_v3.json`
- 旧版 v2：`docs/free_model_benchmark_report_v2.md`（2026-07-15）
- 脚本：`scripts/benchmark_free_models_v3.py` / `scripts/finalize_free_model_report.py`
- **密钥仅从环境变量读取**（已清理 v1/v2 硬编码）
