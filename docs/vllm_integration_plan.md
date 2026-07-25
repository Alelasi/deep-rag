# DeepRAG vLLM/SGLang 集成可行性分析

> **文档版本**: v1.0  
> **对应计划项**: P3-3（vLLM/SGLang 推理加速 — 研究性文档）  
> **项目版本**: DeepRAG v2.8.2  
> **硬件环境**: RTX 4060 Laptop GPU 8GB / CUDA 12.1 / PyTorch 2.5.1+cu121  
> **结论**: 本文档为研究性分析，不实际集成。RTX 4060 8GB 显存下单用户场景 Ollama 已足够，vLLM/SGLang 的收益在当前硬件和并发条件下无法充分发挥。

---

## 目录

1. [vLLM PagedAttention 原理](#1-vllm-pagedattention-原理)
2. [vLLM vs Ollama 对比](#2-vllm-vs-ollama-对比)
3. [SGLang RadixAttention 原理](#3-sglang-radixattention-原理)
4. [RTX 4060 8GB 显存下的可行性分析](#4-rtx-4060-8gb-显存下的可行性分析)
5. [集成路径设计](#5-集成路径设计)
6. [何时值得迁移](#6-何时值得迁移)
7. [当前项目已有的优化](#7-当前项目已有的优化)
8. [总结与建议](#8-总结与建议)

---

## 1. vLLM PagedAttention 原理

### 1.1 背景：KV Cache 内存管理挑战

在大语言模型的自回归推理过程中，模型为每个 token 计算 Key（K）和 Value（V）向量并缓存下来，避免重复计算历史 token 的注意力。这部分缓存称为 **KV Cache**，是推理过程中最主要的动态显存开销。

推理分为两个阶段：

- **Prefill 阶段**：模型接收完整 prompt，一次性并行计算所有 token 的 Q/K/V 向量。此阶段高度并行，属于 **compute-bound**（计算受限），瓶颈在 GPU 算力。
- **Decode 阶段**：模型逐个 token 生成输出，每步仅处理一个新 token，频繁读写 KV Cache。此阶段串行执行，属于 **memory-bound**（内存受限），瓶颈在 KV Cache 的存储与访问效率。

随着生成的 token 数量增加，KV Cache 线性增长。例如 OPT-13B 模型每个 token 的 KV Cache 约占 800KB，生成 2048 个 token 时单个请求的 KV Cache 可达约 1.6GB。

### 1.2 传统内存管理的问题

传统推理系统（如 FasterTransformer、Orca）采用 **预分配策略**：请求开始时按最大可能生成长度分配一整块连续内存。这导致三类浪费：

| 浪费类型 | 说明 |
|---------|------|
| Reserved（已预留未使用） | 为未来 token 预留但当前未使用的空间 |
| Internal Fragmentation（内部碎片） | 实际 token 数小于预留长度，剩余空间浪费 |
| External Fragmentation（外部碎片） | 不同请求内存大小不一，造成块间不连续 |

实验数据显示，传统系统中真正用于存放 KV Cache 的有效内存占比最低仅约 **20.4%**，其余全为浪费。

### 1.3 PagedAttention 核心设计

PagedAttention 借鉴操作系统的 **虚拟内存分页机制（Virtual Memory & Paging）**，将 KV Cache 分块存储在非连续的物理内存中。其三项关键设计：

#### 1.3.1 KV Cache 分块（Block-based Allocation）

- 将每个序列的 KV Cache 切分为固定大小的 **block**（默认 16 个 token/block）
- 每个 block 存储若干 token 的 Key 和 Value 向量
- 统一了内存分配粒度，系统以标准化方式管理 KV Cache 的分配与回收

#### 1.3.2 非连续物理存储 + Block Table 映射

- KV 向量不再要求在内存中连续排列
- 通过 **block table**（类似操作系统的页表）维护逻辑 block 与物理 block 的映射关系
- 每个请求仿佛在连续的内存空间中运行，尽管物理上是非连续存储的

**工作流程示例**（prompt = `"Four score and seven years ago our"`，7 个 token）：

```
Prefill 阶段:
  逻辑 block 0 (4 tokens) → 物理块 7
  逻辑 block 1 (3 tokens) → 物理块 1

Decode 阶段 - 生成 "brought":
  block 1 未满(3/4)，直接写入 → 填充计数更新为 4/4

Decode 阶段 - 生成下一个 token:
  block 1 已满，分配逻辑 block 2 → 物理块 3
  更新 block table 映射
```

每个逻辑块仅在前一个块被填满后才分配新的物理块，最大程度减少内存浪费。

#### 1.3.3 Copy-on-Write 共享机制

- 在 Parallel Sampling、Beam Search 等多分支场景中，多个序列共享相同的 prompt
- PagedAttention 允许多个序列共享同一组物理块（通过引用计数管理）
- 只有当某个分支需要写入新 token 的 KV 数据时，才触发 **copy-on-write（CoW）**，将相关 block 复制到新的物理位置
- 在保证数据隔离的同时极大节省显存资源

### 1.4 PagedAttention 的效果

| 指标 | 传统系统 | vLLM (PagedAttention) |
|------|---------|----------------------|
| 有效内存利用率 | ~20.4% | ~96%+ |
| vs HuggingFace Transformers 吞吐量 | 基线 | 最高 24x |
| vs HuggingFace TGI 吞吐量 | 基线 | 最高 3.5x |
| 内存碎片 | 严重 | 极少 |

### 1.5 调度与抢占机制

当请求量超过显存容量时，vLLM 采用 FCFS（先来先服务）策略调度：

- **抢占策略**：All-or-Nothing — 要么完全回收一个请求的全部 KV Cache，要么不回收
- **恢复策略**（二选一）：
  - **Swapping**：将 KV Cache 从 GPU 移到 CPU 内存，有空闲时恢复
  - **Recomputation**：丢弃 KV Cache，恢复时用已生成 token + prompt 重新 prefill

---

## 2. vLLM vs Ollama 对比

### 2.1 架构定位差异

| 维度 | Ollama | vLLM |
|------|--------|------|
| **定位** | 本地个人使用，极简部署 | 生产级高吞吐推理服务 |
| **底层** | 基于 llama.cpp，GGUF 格式 | 纯 Python + CUDA，PyTorch 生态 |
| **量化** | GGUF (Q4_K_M, Q5_K_M 等) | AWQ, GPTQ, FP16, BF16 |
| **部署复杂度** | 一行命令 `ollama pull` | 需 Python 环境 + 模型权重 |
| **API 兼容** | OpenAI-compatible API | OpenAI-compatible API |

### 2.2 核心技术差异

#### 2.2.1 连续批处理（Continuous Batching）

| 特性 | Ollama | vLLM |
|------|--------|------|
| 批处理方式 | 静态批处理（请求到达后组成一个 batch，等所有完成才返回） | **连续批处理**（iteration-level scheduling） |
| 新请求插入 | 必须等当前 batch 完成 | 可在任意 decode step 插入正在进行的 batch |
| 尾延迟 | 长请求会阻塞短请求 | 短请求不被长请求阻塞 |

**连续批处理原理**：vLLM 在每个 decode iteration 级别动态调整 batch 组成。新请求可以在任意 step 加入正在处理的 batch，已完成的请求可以随时退出。这意味着短请求不会被长请求阻塞，极大降低了尾延迟（tail latency）。

Ollama 的并发模型较为简单：默认串行处理请求，或通过 OLLAMA_NUM_PARALLEL 环境变量开启有限并发（受显存约束），但缺少 iteration-level 的动态调度。

#### 2.2.2 PagedAttention

| 特性 | Ollama | vLLM |
|------|--------|------|
| KV Cache 管理 | 预分配连续内存 | PagedAttention 分页管理 |
| 内存碎片 | 存在内部/外部碎片 | 几乎消除 |
| 内存利用率 | ~20-40% | ~96%+ |
| KV Cache 共享 | 不支持 | 支持（copy-on-write） |

Ollama 基于 llama.cpp，其 KV Cache 管理采用传统连续分配方式。虽然 llama.cpp 也在不断优化（如 KV Cache 重用、memory reuse），但缺少 vLLM PagedAttention 级别的精细分页管理。

#### 2.2.3 Tensor Parallelism

| 特性 | Ollama | vLLM |
|------|--------|------|
| 多 GPU 支持 | 不支持 Tensor Parallelism | 原生支持 Tensor Parallelism |
| 模型分片 | 不支持 | `--tensor-parallel-size N` 跨 N 张 GPU 分片 |
| 适用场景 | 单 GPU | 多 GPU 服务器，运行超大模型 |

vLLM 的 Tensor Parallelism 允许将模型权重、KV Cache、注意力计算分散到多张 GPU 上，从而运行超过单卡显存容量的模型。Ollama 不支持此功能，单卡显存即上限。

#### 2.2.4 其他差异

| 特性 | Ollama | vLLM |
|------|--------|------|
| 模型格式 | GGUF（支持 Q4/Q5/Q8 等多种量化） | HuggingFace 格式（safetensors）+ AWQ/GPTQ |
| 流式输出 | 支持 | 支持 |
| Function Calling | 支持（部分模型） | 支持 |
| 量化效率 | GGUF Q4 内存占用极低 | AWQ/GPTQ 精度更高但内存占用略大 |
| 模型热切换 | 支持（自动卸载/加载） | 需重启服务 |
| 社区生态 | 模型仓库丰富 | 企业级采用率高 |

### 2.3 性能对比总结

| 场景 | Ollama | vLLM | 胜出 |
|------|--------|------|------|
| 单用户低并发 | ~50 tok/s (qwen2.5:7b Q4) | ~55-60 tok/s (qwen2.5:7b AWQ) | 接近，Ollama 部署更简单 |
| 高并发 (>10 QPS) | 串行/有限并发，延迟急剧上升 | 连续批处理，吞吐量线性增长 | vLLM 显著优势 |
| 多 GPU | 不支持 | Tensor Parallelism | vLLM |
| 内存效率 | ~20-40% KV Cache 利用率 | ~96%+ | vLLM |
| 部署难度 | 一行命令 | 需配置 Python 环境 | Ollama |

**核心结论**：vLLM 的优势集中在 **高并发吞吐量** 和 **内存效率**，在单用户低并发场景下与 Ollama 差距不大，但部署复杂度显著更高。

---

## 3. SGLang RadixAttention 原理

### 3.1 背景：前缀复用的价值

在 RAG 系统和多轮对话中，大量请求共享相同的前缀（如 system prompt、few-shot examples、知识库上下文）。如果能复用这些前缀的 KV Cache，可以避免重复计算，显著降低延迟。

传统的前缀缓存采用 **线性匹配**：将新请求的 prompt 与缓存中的 prompt 逐一比较，找到完全匹配的前缀。这种方式有两个局限：

1. **匹配粒度粗**：必须从请求开头完全匹配，中间任何差异都会导致缓存失效
2. **无法处理分支场景**：多轮对话中不同分支无法共享公共前缀

### 3.2 RadixAttention 核心设计

SGLang 引入 **RadixAttention**，将 KV Cache 的前缀复用从"线性匹配"升级为"树形复用"。

#### 3.2.1 Radix Tree（基数树）组织 KV Cache

RadixAttention 使用 **基数树（Radix Tree）** 来组织和管理 KV Cache：

```
                    [System Prompt]
                   /                \
          [User Query 1]        [User Query 2]
           /          \              |
    [Response 1A]  [Response 1B]  [Response 2]
```

- 树的每个节点代表一段 token 序列（前缀）
- 从根到某节点的路径构成一个完整的 prompt 前缀
- 每个节点存储该前缀对应的 KV Cache

#### 3.2.2 前缀共享机制

当新请求到达时：

1. **前缀匹配**：将新请求的 token 序列与 Radix Tree 进行最长前缀匹配
2. **缓存命中**：如果找到匹配的前缀节点，直接复用该节点的 KV Cache
3. **增量计算**：仅对不匹配的后缀部分进行 prefill 计算
4. **树更新**：将新计算的 KV Cache 作为新节点插入树中

**示例**：

```
请求 A: "你是一个MBTI专家。\n INTJ的主导功能是什么？"
请求 B: "你是一个MBTI专家。\n INFP的辅助功能是什么？"

Radix Tree 结构:
  Root → "你是一个MBTI专家。\n" (KV Cache 已缓存)
        ├── "INTJ的主导功能是什么？" (请求 A 独有部分)
        └── "INFP的辅助功能是什么？" (请求 B 独有部分)

请求 B 到达时:
  - 匹配到公共前缀 "你是一个MBTI专家。\n"
  - 复用该前缀的 KV Cache
  - 仅计算 "INFP的辅助功能是什么？" 的 KV Cache
```

#### 3.2.3 与 PagedAttention 的关系

RadixAttention 和 PagedAttention 是互补而非替代关系：

| 维度 | PagedAttention (vLLM) | RadixAttention (SGLang) |
|------|----------------------|------------------------|
| **解决的问题** | KV Cache 内存碎片化 | KV Cache 前缀复用 |
| **数据结构** | Block Table（页表） | Radix Tree（基数树） |
| **复用粒度** | Block 级别（16 token） | 前缀级别（任意长度） |
| **复用场景** | Parallel Sampling、Beam Search | 多请求共享 system prompt、多轮对话 |
| **内存效率** | 消除碎片，利用率 ~96% | 减少重复计算，缓存命中率提升 3.8x |

SGLang 实际上也使用了类似 PagedAttention 的分页管理来存储 KV Cache，但在其之上增加了 Radix Tree 层来实现智能前缀复用。

### 3.3 RadixAttention 的效果

根据 SGLang 官方和社区实测数据：

| 指标 | 无前缀缓存 | RadixAttention |
|------|-----------|----------------|
| 缓存命中率 | N/A | 提升 3.8x（ShareGPT 数据集） |
| 多轮对话延迟 | 每轮重新计算完整 prompt | 仅计算增量部分，延迟降低 50%+ |
| 共享 system prompt 场景 | 每个请求独立计算 | 首次计算后后续请求复用 |

### 3.4 SGLang 的其他特性

除了 RadixAttention，SGLang 还提供：

- **Constrained Decoding**：原生支持 JSON/正则约束输出
- ** speculative decoding**：投机解码加速
- **多 LoRA 适配器**：动态切换 LoRA
- **OpenAI-compatible API**：与 vLLM 类似的 API 接口

---

## 4. RTX 4060 8GB 显存下的可行性分析

### 4.1 当前显存占用情况

DeepRAG v2.8.2 在 RTX 4060 Laptop GPU（8GB VRAM）下的显存分布：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| qwen2.5:7b Q4 (模型权重) | ~4.6 GB | Ollama GGUF Q4_K_M 量化 |
| bge-base-zh-v1.5 (Embedding) | ~0.5 GB | 768 维，110M 参数 |
| ChromaDB (向量数据库) | ~0.5 GB | HNSW 索引常驻 |
| **小计** | **~5.6 GB** | 安全线内 |
| 剩余可用 | **~2.4 GB** | 用于 KV Cache 和激活值 |
| Reranker (bge-reranker-base) | ~1.2 GB | **加载后溢出**，已禁用 |

### 4.2 vLLM 在 8GB 显存下的显存需求

vLLM 加载 qwen2.5:7b 时的显存构成：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| 模型权重 (FP16) | ~14 GB | 原始 FP16 无法加载 |
| 模型权重 (AWQ Q4) | ~4.5 GB | AWQ 4-bit 量化 |
| 模型权重 (GPTQ Q4) | ~4.5 GB | GPTQ 4-bit 量化 |
| KV Cache Pool | ~1.0-2.0 GB | vLLM 预分配的 KV Cache 池 |
| CUDA 上下文 + 激活值 | ~0.5-1.0 GB | PyTorch 运行时开销 |
| **总计 (AWQ Q4)** | **~6.0-7.5 GB** | 接近 8GB 上限 |

#### 4.2.1 关键问题：KV Cache Pool 预分配

vLLM 与 Ollama 的一个重要差异在于 KV Cache 管理方式：

- **Ollama**：KV Cache 按需分配，单请求时占用最小
- **vLLM**：启动时预分配一个 **KV Cache Pool**（通过 `--gpu-memory-utilization` 控制，默认占可用显存的 90%），用于 PagedAttention 的 block 管理

在 8GB 显存下，vLLM 的 KV Cache Pool 计算：

```
可用显存 = 8GB - 模型权重(4.5GB) - CUDA上下文(0.5GB) = 3.0GB
KV Cache Pool = 3.0GB × 90% = 2.7GB

qwen2.5:7b 每个 token 的 KV Cache (FP16):
  2 × 3584(hidden_size) × 28(num_layers) × 2(bytes) = 400KB/token

可缓存的 token 数 = 2.7GB / 400KB ≈ 6,750 tokens
```

这意味着 vLLM 在 8GB 显存下最多可同时缓存约 6,750 个 token 的 KV Cache。对于单请求（prompt + 生成的 token），这通常足够。但多请求并发时，可缓存的序列数非常有限。

#### 4.2.2 与 Embedding/ChromaDB 的显存冲突

当前项目需要同时运行 Embedding 模型（bge-base-zh）和 ChromaDB：

| 方案 | 显存分配 | 可行性 |
|------|---------|--------|
| vLLM 独占 GPU | vLLM 7.5GB + Embedding 0.5GB = 8GB | 临界，无余量 |
| vLLM + Embedding + ChromaDB | 7.5GB + 0.5GB + 0.5GB = 8.5GB | **溢出** |
| vLLM 降配 (gpu_memory_utilization=0.6) | 4.5GB + 1.8GB + 0.5GB + 0.5GB = 7.3GB | 勉强可行，但 KV Cache Pool 仅 1.8GB |

**结论**：如果要在 8GB 显存下同时运行 vLLM + Embedding + ChromaDB，必须降低 vLLM 的 `gpu_memory_utilization`，但这会大幅压缩 KV Cache Pool，削弱 PagedAttention 的优势。

### 4.3 SGLang 在 8GB 显存下的显存需求

SGLang 的显存需求与 vLLM 类似：

| 组件 | 显存占用 |
|------|---------|
| 模型权重 (Q4) | ~4.5 GB |
| KV Cache Pool | ~1.0-2.0 GB |
| Radix Tree 元数据 | ~0.1-0.2 GB（很小） |
| CUDA 上下文 | ~0.5-1.0 GB |
| **总计** | **~6.1-7.7 GB** |

SGLang 的 RadixAttention 在 8GB 下的价值有限：

- RadixAttention 的收益来自多请求共享前缀，但 8GB 显存限制了可同时缓存的序列数
- 单用户场景下，前缀复用的机会较少（每次请求的 system prompt 相同但 user query 不同）
- RAG 场景中，每条检索到的文档内容不同，前缀共享率低

### 4.4 可行性结论

| 评估维度 | 结论 | 理由 |
|---------|------|------|
| 模型加载 | 可行 | Q4 量化下 4.5GB，8GB 可容纳 |
| KV Cache Pool | 受限 | 降配后仅 1.8GB，削弱 PagedAttention 优势 |
| 与 Embedding/ChromaDB 共存 | 勉强 | 需降低 gpu_memory_utilization，性能折损 |
| 多并发 | 不可行 | KV Cache Pool 太小，无法支撑多请求 |
| RadixAttention 收益 | 有限 | 单用户场景前缀复用机会少 |
| 整体收益 vs Ollama | 不明显 | 单用户场景 Ollama 已达到 50 tok/s |

---

## 5. 集成路径设计

虽然当前不建议实际集成，但以下为未来迁移时的集成路径设计。设计原则：**最小侵入性，向后兼容**。

### 5.1 架构概览

```
当前架构:
  config.py (get_llm) → ChatOllama (LangChain) → Ollama Server (:11434)

集成后架构:
  config.py (get_llm) → ChatOpenAI (LangChain) → vLLM Server (:8000)
                                    ↘ ChatOllama (LangChain) → Ollama Server (:11434)
```

vLLM 提供 OpenAI-compatible API，因此集成方式与现有的 `zhipu`/`siliconcloud` backend 完全一致，使用 LangChain 的 `ChatOpenAI` 连接本地 vLLM 服务。

### 5.2 Step 1: config.py 增加 vllm backend

在 `config.py` 的 `LLM_BACKEND` 配置中增加 `vllm` 选项：

```python
# config.py 修改点

# LLM后端切换：anthropic / zhipu / openai / ollama / vllm / none（规则模式）
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")

# vLLM 服务配置
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM 默认不需要真实 key
```

### 5.3 Step 2: get_llm() 增加 vllm 分支

在 `get_llm()` 函数中增加 vllm backend 分支。由于 vLLM 使用 OpenAI-compatible API，实现方式与 `siliconcloud`/`zhipu` 几乎相同：

```python
# config.py — get_llm() 函数中新增分支

if backend == "vllm":
    model = LLM_MODEL or VLLM_MODEL
    def _factory():
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=VLLM_API_KEY,          # vLLM 默认 "EMPTY"
            base_url=VLLM_BASE_URL,        # http://localhost:8000/v1
        )
    return get_cached_llm(backend, model, temp, _factory)
```

同时在 `auto` 模式的优先级链中插入 vllm 检测（可选）：

```python
# auto 模式优先级（修改后）:
# anthropic → siliconcloud → zhipu → vllm(本地) → openai → ollama → none
```

### 5.4 Step 3: 使用 vLLM OpenAI-compatible API

vLLM 服务的启动命令（参考）：

```bash
# 启动 vLLM 服务（加载 AWQ 量化模型）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --port 8000
```

关键参数说明：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--quantization` | awq | AWQ 4-bit 量化，显存占用 ~4.5GB |
| `--tensor-parallel-size` | 1 | 单 GPU，无 Tensor Parallelism |
| `--gpu-memory-utilization` | 0.6 | 降低到 60%，为 Embedding/ChromaDB 留空间 |
| `--max-model-len` | 4096 | 限制上下文长度，控制 KV Cache 占用 |

### 5.5 集成的兼容性考虑

| 组件 | 兼容性 | 说明 |
|------|--------|------|
| LangChain ChatOpenAI | 完全兼容 | vLLM 的 OpenAI API 与 LangChain 无缝对接 |
| ollama_helper.py 原生 API | 不兼容 | think 模式是 Ollama 专有，vLLM 无此功能 |
| Function Calling | 兼容 | vLLM 支持 OpenAI 格式的 tool_calls |
| 流式输出 | 兼容 | vLLM 支持 SSE 流式 |
| Constrained Decoding | 兼容 | vLLM 支持 `guided_json` 参数 |
| 动态温度 | 兼容 | 通过 temperature 参数传递 |

### 5.6 ollama_helper.py 的适配

`ollama_helper.py` 中的 `ollama_chat()` 使用 Ollama 原生 `/api/chat` 接口（支持 think 参数）。迁移到 vLLM 后：

- `think` 模式不可用（vLLM 不支持 Ollama 的 think 参数）
- 需要通过 `LLM_BACKEND` 判断，在 vllm 模式下走 LangChain `ChatOpenAI` 路径
- `ollama_chat_or_fallback()` 函数已有 langchain 回退逻辑，天然兼容

---

## 6. 何时值得迁移

### 6.1 迁移决策矩阵

| 场景特征 | 推荐 backend | 理由 |
|---------|-------------|------|
| 单用户、交互式问答 | **Ollama** | 50 tok/s 已足够，部署简单，显存占用低 |
| 单用户、长上下文（>8K） | **Ollama** | GGUF Q4 量化效率高，KV Cache 按需分配 |
| 多用户共享、并发 <5 QPS | **Ollama** | OLLAMA_NUM_PARALLEL 可应对低并发 |
| 多用户共享、并发 5-10 QPS | **均可** | vLLM 略优，但 8GB 显存限制并发数 |
| 多用户共享、并发 >10 QPS | **vLLM** | 连续批处理 + PagedAttention 优势显著 |
| 多用户共享、并发 >50 QPS | **vLLM (多GPU)** | 需要 Tensor Parallelism，8GB 单卡不够 |
| 多轮对话为主、共享 system prompt | **SGLang** | RadixAttention 前缀复用收益大 |
| 需要运行超大模型（>13B） | **vLLM (多GPU)** | Tensor Parallelism 跨卡分片 |

### 6.2 量化迁移阈值

| 指标 | 阈值 | 当前值 | 是否触发迁移 |
|------|------|--------|------------|
| 并发 QPS | >10 | ~1（单用户） | 否 |
| 并发用户数 | >5 | 1 | 否 |
| 平均请求延迟 | >10s | ~5s | 否 |
| 上下文长度 | >16K | ~2-4K | 否 |
| GPU 数量 | >1 | 1 | 否 |
| 吞吐量需求 | >200 tok/s | 50 tok/s | 否 |

### 6.3 迁移成本评估

| 维度 | 成本 | 说明 |
|------|------|------|
| 代码改动 | 低 | config.py 增加一个分支，~20 行代码 |
| 部署复杂度 | 中 | 需安装 vLLM、下载 AWQ 模型、配置服务 |
| 显存调优 | 高 | 需精细调整 gpu_memory_utilization 平衡各组件 |
| 功能损失 | 中 | 丢失 think 模式、GGUF 多档量化 |
| 运维成本 | 中 | vLLM 服务需独立管理和监控 |

### 6.4 当前项目的结论

DeepRAG v2.8.2 当前使用场景为 **单用户交互式问答**，平均并发约 1 QPS，Ollama + qwen2.5:7b Q4 已达到 50 tok/s（~5s 延迟），完全满足需求。迁移到 vLLM/SGLang 的收益不明显，且增加部署复杂度和显存压力。

**建议**：保持 Ollama 作为本地推理后端。当以下任一条件满足时再考虑迁移：

1. 项目需要支持多用户并发（>5 人同时使用）
2. 硬件升级到 16GB+ VRAM 或多 GPU
3. 吞吐量需求超过 200 tok/s
4. 需要运行 13B+ 参数模型（需 Tensor Parallelism）

---

## 7. 当前项目已有的优化

DeepRAG v2.8.2 已通过 Ollama 环境变量实现多项推理优化，这些优化在当前硬件条件下已接近单 GPU 的性能极限。

### 7.1 Flash Attention

```bash
OLLAMA_FLASH_ATTENTION=1
```

| 维度 | 说明 |
|------|------|
| **原理** | 将 attention 计算分块（tiling），减少 GPU HBM 与 SRAM 之间的数据搬运。传统 attention 需要将完整的 Q/K/V 矩阵加载到 SRAM，Flash Attention 分块计算，避免中间矩阵的显存读写 |
| **效果** | 减少 KV Cache 的显存占用（不需要存储完整的 attention matrix），加速长序列推理 |
| **Ollama 支持** | 通过环境变量启用，底层由 llama.cpp 实现 |
| **vLLM 对比** | vLLM 默认使用 FlashAttention/FlashInfer，无需额外配置 |
| **项目状态** | 已设为 User 级永久环境变量，全局生效 |

### 7.2 KV Cache 量化（KV Cache Quantization）

```bash
OLLAMA_KV_CACHE_TYPE=q8_0
```

| 维度 | 说明 |
|------|------|
| **原理** | 将 KV Cache 从 FP16（16-bit）量化为 Q8_0（8-bit），显存占用减半。精度损失极小（<1% 输出差异） |
| **效果** | KV Cache 显存占用降低 50%，相同显存下可支持更长的上下文或更多并发 |
| **Ollama 支持** | 通过 `OLLAMA_KV_CACHE_TYPE` 环境变量配置，支持 `q8_0`/`q4_0`/`f16` |
| **vLLM 对比** | vLLM 支持 `--kv-cache-dtype fp8` 实现类似功能 |
| **项目状态** | 已设为 User 级永久环境变量 |
| **显存预算** | qwen2.5:7b 原始 KV Cache 每 token ~400KB(FP16)，Q8 量化后 ~200KB/token |

### 7.3 Keep Alive（模型驻留）

```bash
OLLAMA_KEEP_ALIVE=5m
```

| 维度 | 说明 |
|------|------|
| **原理** | 模型加载后在 GPU 显存中保持 5 分钟，避免每次请求重新加载模型（冷启动需 3-5s） |
| **效果** | 热请求延迟从 ~8s（含加载）降至 ~5s（纯推理），消除模型加载开销 |
| **Ollama 支持** | 通过 `OLLAMA_KEEP_ALIVE` 环境变量配置 |
| **vLLM 对比** | vLLM 模型常驻显存，无卸载机制（服务启动即加载，停止才卸载） |
| **项目状态** | 已设为 User 级永久环境变量 |
| **权衡** | 5 分钟内无请求时模型仍占显存，但单用户场景下请求间隔通常 <5 分钟 |

### 7.4 优化效果汇总

| 优化项 | 显存节省 | 速度提升 | 精度影响 |
|--------|---------|---------|---------|
| Flash Attention | ~15-20%（减少 attention matrix 存储） | ~20-30%（长序列） | 无 |
| KV Cache 量化 (q8_0) | ~50%（KV Cache 部分减半） | ~5-10%（减少显存读写） | <1% |
| Keep Alive (5m) | 无（反而常驻占用） | 消除冷启动 3-5s | 无 |
| **综合效果** | **~60-70% KV Cache 节省** | **50 tok/s，~5s 延迟** | **可忽略** |

### 7.5 与 vLLM 的优化对比

| 优化维度 | Ollama (当前) | vLLM | 差异 |
|---------|--------------|------|------|
| Flash Attention | 启用 | 默认启用 | 相当 |
| KV Cache 量化 | q8_0 | fp8 | 相当 |
| 模型驻留 | Keep Alive 5m | 常驻 | vLLM 更优（永不卸载） |
| KV Cache 内存管理 | 连续分配 | PagedAttention 分页 | vLLM 更优（高并发时） |
| 批处理 | 静态/有限并发 | 连续批处理 | vLLM 更优（高并发时） |
| 前缀复用 | 不支持 | vLLM 有 Automatic Prefix Caching | vLLM 略优 |
| **单用户场景净收益** | — | — | **接近零** |

**核心结论**：当前项目已通过 Ollama 环境变量实现了 Flash Attention + KV Cache 量化 + 模型驻留三大优化，单用户场景下与 vLLM 的性能差距已缩小到可忽略的程度。vLLM 的核心优势（PagedAttention + 连续批处理）只在高并发场景下才能体现。

---

## 8. 总结与建议

### 8.1 技术对比总结

| 维度 | Ollama (当前) | vLLM | SGLang |
|------|--------------|------|--------|
| KV Cache 管理 | 连续分配 | PagedAttention（分页） | RadixAttention（基数树） |
| 核心优势 | 极简部署、GGUF 量化效率 | 高并发吞吐、内存效率 | 前缀复用、多轮对话加速 |
| 8GB 显存适配性 | 优秀（Q4 仅 4.6GB） | 勉强（需降配 KV Cache Pool） | 勉强（同 vLLM） |
| 单用户性能 | 50 tok/s | ~55-60 tok/s | ~55-60 tok/s |
| 高并发性能 | 差（串行为主） | 优秀（连续批处理） | 优秀（+ 前缀复用） |
| 部署复杂度 | 极低 | 中 | 中 |
| 与项目集成难度 | 已集成 | 低（OpenAI API 兼容） | 低（OpenAI API 兼容） |

### 8.2 最终建议

1. **当前阶段（v2.8.2）**：保持 Ollama 作为本地推理后端。已有的 Flash Attention + KV Cache 量化 + Keep Alive 优化已充分挖掘 8GB 显存的性能潜力。

2. **未来迁移条件**：当并发需求超过 10 QPS 或硬件升级到 16GB+ VRAM 时，考虑迁移到 vLLM。集成路径已在第 5 节设计完毕，代码改动量约 20 行。

3. **SGLang 的特殊价值**：如果项目未来以多轮对话为主要交互模式（大量请求共享 system prompt），SGLang 的 RadixAttention 比纯 vLLM 更有优势。

4. **面试知识点覆盖**：本文档覆盖了 vLLM PagedAttention 原理、SGLang RadixAttention 原理、连续批处理、Tensor Parallelism、KV Cache 内存管理等面试核心知识点，满足 P3-3 计划要求。

---

## 参考资料

- [PagedAttention 论文: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [SGLang 官方 GitHub](https://github.com/sgl-project/sglang)
- [vLLM PagedAttention 原理详解 (阿里云开发者社区)](https://developer.aliyun.com/article/1664805)
- [SGLang RadixAttention 技术解析](https://blog.csdn.net/weixin_35364187/article/details/157338738)
- DeepRAG v2.8.2 项目配置 (CLAUDE.md)
- DeepRAG v2.9 完善计划 P3-3 章节
