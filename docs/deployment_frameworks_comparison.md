# 大模型部署框架评测对比报告

> 面试要点（04-20 部署框架）：vLLM/SGLang/TGI/llama.cpp各有侧重，选型要看场景

## 1. 框架概览

| 框架 | 核心创新 | 最佳场景 | 开发语言 |
|------|---------|---------|---------|
| **vLLM** | PagedAttention + Continuous Batching | 高吞吐LLM API | Python |
| **SGLang** | RadixAttention（共享前缀） | Agent/多轮对话/Few-shot | Python |
| **TGI** | HF生态集成 | 企业级+HF生态 | Rust |
| **llama.cpp** | C++重写+GGUF | CPU/Mac/边缘 | C++ |
| **Ollama** | 简化部署+模型管理 | 本地开发/快速原型 | Go |

## 2. 性能对比（7B模型，单卡RTX 4060）

| 指标 | vLLM | SGLang | TGI | llama.cpp | Ollama |
|------|------|--------|-----|-----------|--------|
| 吞吐量(tok/s) | ~50 | ~45 | ~40 | ~30 | ~35 |
| 首token延迟 | 低 | 最低 | 中 | 中 | 中 |
| 显存利用率 | 90%+ | 90%+ | 80% | 70% | 75% |
| 并发支持 | 优秀 | 优秀 | 良好 | 一般 | 一般 |
| 量化支持 | GPTQ/AWQ/FP8 | GPTQ/AWQ/FP8 | 多种 | GGUF | GGUF |

## 3. 场景选型建议

### 3.1 生产环境高吞吐API
**推荐：vLLM**
- PagedAttention显存利用率90%+
- Continuous Batching动态调度
- 适合高并发场景

### 3.2 Agent/多轮对话
**推荐：SGLang**
- RadixAttention共享前缀KV Cache
- System Prompt只存一份，多请求复用
- 首token延迟降低2-3倍

### 3.3 HuggingFace生态
**推荐：TGI**
- 与HF Hub深度集成
- 企业级特性（鉴权、metrics、健康检查）
- 开箱即用

### 3.4 本地/Mac/边缘部署
**推荐：llama.cpp/Ollama**
- CPU推理优化
- GGUF量化格式
- 个人开发者首选

## 4. DeepRAG项目选型

### 当前选择：Ollama
**原因**：
1. 简化部署，一键启动
2. 支持多种量化模型
3. 本地开发友好
4. 社区活跃

**局限**：
1. 并发支持有限
2. 显存利用率不如vLLM
3. 缺少PagedAttention优化

### 未来优化方向
1. **生产环境**：迁移到vLLM/SGLang
2. **Agent场景**：优先评估SGLang（RadixAttention）
3. **边缘部署**：保留llama.cpp/Ollama

## 5. 面试回答要点

### 5.1 核心区别
- **vLLM**：PagedAttention（虚拟内存灵感）+ Continuous Batching
- **SGLang**：RadixAttention（共享前缀树）
- **TGI**：HF生态集成
- **llama.cpp**：CPU优化+GGUF格式

### 5.2 选型决策
```
生产API高吞吐 → vLLM
Agent/多轮对话 → SGLang
HF生态用户 → TGI
本地/Mac/边缘 → llama.cpp/Ollama
```

### 5.3 关键洞见
1. SGLang在前缀共享场景比vLLM强（Agent、Few-shot）
2. PagedAttention把显存利用率从30%拉到90%+
3. GGUF是文件格式，不是量化算法
4. MoE模型部署比Dense复杂（专家并行、All-to-All通信）

## 6. 参考资料

- [vLLM论文](https://arxiv.org/abs/2309.06180)
- [SGLang论文](https://arxiv.org/abs/2312.07104)
- [TGI文档](https://huggingface.co/docs/text-generation-inference)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
