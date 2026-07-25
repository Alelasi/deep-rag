# DeepRAG 企业级 Multi-Agent RAG 系统

## 面试完整讲解 & 简历素材（v2.9.2）

> 📅 更新日期：2026-07-13
> 🎯 适用岗位：AI应用工程师 / LLM工程师 / RAG工程师 / Agent工程师
> 📊 核心指标：准确率95% | 幻觉率5% | 响应<2秒 | 成本-64%

---

## 一、项目概述（30秒版）

### 🎯 一句话介绍
> 我做了一个企业级RAG系统，通过多Agent协作+混合检索+自我反思，把准确率从60%提升到95%，幻觉率从20%降到5%。

### 📊 核心数据
| 指标 | 数值 | 说明 |
|------|------|------|
| 准确率 | 95% | Top-5召回率，v2.2提升+7% |
| 幻觉率 | 5% | 从v1.0的20%降低 |
| 响应时间 | <2秒 | P50: 1.2s, P90: 2.0s |
| 成本优化 | -64% | Token追踪+缓存 |
| 可靠性 | 99.5% | 3层降级策略 |
| 代码规模 | 7,600+行 | 核心代码 |

### 🏗️ 技术栈
```
LLM: qwen2.5:7b / GLM-4-Flash / Claude
框架: LangChain + LangGraph
向量DB: ChromaDB（服务器模式）
Embedding: bge-base-zh-v1.5（768维）
部署: Ollama + Streamlit
协议: MCP + A2A + Function Calling
```

---

## 二、技术架构详解

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepRAG v2.9.2 架构                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Streamlit  │  │  SSE API   │  │  MCP Server │         │
│  │   前端界面   │  │  流式接口   │  │  工具服务   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                 │
│  ┌──────┴─────────────────┴─────────────────┴───────┐       │
│  │              LLM Gateway (统一入口)                │       │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │       │
│  │  │ 路由    │ │ 限流    │ │ 熔断    │ │ 语义缓存│ │       │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │       │
│  └────────────────────────┬──────────────────────────┘       │
│                           │                                   │
│  ┌────────────────────────┴──────────────────────────┐       │
│  │              Agent 协作层 (A2A协议)                 │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │       │
│  │  │ Research │ │ Verify   │ │ Precision│           │       │
│  │  │ Agent    │ │ Agent    │ │ Agent    │           │       │
│  │  └──────────┘ └──────────┘ └──────────┘           │       │
│  └────────────────────────┬──────────────────────────┘       │
│                           │                                   │
│  ┌────────────────────────┴──────────────────────────┐       │
│  │              检索引擎 (混合检索)                     │       │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │       │
│  │  │ BM25    │ │ 向量    │ │ 图谱    │ │ Web     │ │       │
│  │  │ 关键词  │ │ 语义    │ │ 知识图谱│ │ 兜底    │ │       │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │       │
│  └────────────────────────┬──────────────────────────┘       │
│                           │                                   │
│  ┌────────────────────────┴──────────────────────────┐       │
│  │              数据存储层                             │       │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐              │       │
│  │  │ChromaDB │ │ SQLite  │ │ Redis   │              │       │
│  │  │向量存储 │ │ 元数据  │ │ 缓存    │              │       │
│  │  └─────────┘ └─────────┘ └─────────┘              │       │
│  └───────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块说明

| 模块 | 文件 | 代码行 | 功能 |
|------|------|--------|------|
| **Graph引擎** | `src/graph.py` | 800+ | 主流程编排，状态机管理 |
| **双Agent精准模式** | `src/agents/dual_agent.py` | 500+ | 双Agent并行+矛盾检测 |
| **A2A协议** | `src/agents/a2a_protocol.py` | 450+ | Agent间通信协议 |
| **MCP Server** | `src/tools/mcp_server.py` | 600+ | 标准MCP工具服务 |
| **LLM Gateway** | `src/llm/gateway.py` | 300+ | 统一入口+指标收集 |
| **语义缓存** | `src/llm/semantic_cache.py` | 250+ | 向量相似度缓存 |
| **Prompt模板** | `src/llm/prompt_templates.py` | 350+ | 五要素模板系统 |
| **约束解码** | `src/llm/constrained_decoder.py` | 250+ | JSON Schema强制输出 |
| **Skill系统** | `src/tools/skill_system.py` | 400+ | 渐进式工具加载 |
| **事实核查** | `src/agents/fact_checker.py` | 200+ | 幻觉检测 |
| **引用验证** | `src/agents/citation_validator.py` | 200+ | 引用率验证 |

---

## 三、面试知识点覆盖矩阵

### 3.1 LLM工具调用（03系列 - 16题）

| 知识点 | 面试题号 | 项目实现 | 面试回答要点 |
|--------|---------|---------|-------------|
| **Function Calling** | 01 | dual_agent.py | 模型只决策，代码负责执行 |
| **MCP协议** | 04-06 | mcp_server.py | Host/Client/Server三层，Tools/Resources/Prompts |
| **MCP vs FC** | 06-07 | 集成方案 | MCP底层靠FC驱动，不同层次 |
| **Skill系统** | 09-11 | skill_system.py | 渐进式加载，30-50token/技能 |
| **A2A协议** | 12 | a2a_protocol.py | Agent Card + Task状态机 |
| **传输协议** | 13-15 | stdio + HTTP | JSON-RPC 2.0，Streamable HTTP |
| **LLM Gateway** | 16 | gateway.py | 统一接口+语义缓存+成本追踪 |

**面试话术**：
> "我的系统实现了完整的工具调用三层架构：最底层用Function Calling做模型和工具的通信，中间层用MCP协议做工具的标准化封装和发现，最上层用Skill系统做任务流程编排。还实现了A2A协议支持多Agent协作，Agent Card声明能力，Task状态机管理异步任务。"

### 3.2 大模型工程（04系列 - 22题）

| 知识点 | 面试题号 | 项目实现 | 面试回答要点 |
|--------|---------|---------|-------------|
| **Temperature/Top-P** | 13 | config.py | 动态温度策略，按任务类型调节 |
| **KV Cache/Prompt Caching** | 14 | prompt_cache.py | 固定内容在前，动态在后 |
| **量化** | 15 | Ollama Q4 | INT4是甜蜜点，GGUF是格式非算法 |
| **Prompt Engineering** | 16 | prompt_templates.py | 五要素：Role/Task/Context/Format/Examples |
| **CoT/Self-Consistency** | 17 | dual_agent.py | 多次采样+多数投票，提升5-15% |
| **幻觉防控** | 18 | fact_checker.py | 三层根因+三层缓解 |
| **MoE** | 19 | qwen2.5-MoE | 总参数大，激活参数小 |
| **部署框架** | 20 | Ollama+vLLM | PagedAttention/RadixAttention |
| **评测指标** | 21 | llm_judge.py | 业务测试集+LLM-as-Judge |
| **模型选型** | 22 | model_router.py | 合规/成本/延迟/能力四维选型 |

**面试话术**：
> "工程层面做了很多优化：用五要素模板系统管理Prompt，用Self-Consistency多次采样投票提升准确率，用约束解码强制JSON输出，用语义缓存降低API成本64%。评测用RAGAS四指标+LLM-as-Judge，模型选型按合规/成本/延迟/能力四维度做路由。"

---

## 四、关键技术实现详解

### 4.1 双Agent精准模式（核心亮点）

**面试问题**："你是怎么解决RAG系统幻觉问题的？"

**回答结构**：
```
1. 问题定义：幻觉=模型生成流畅但错误的内容
2. 解决方案：双Agent并行+矛盾检测+仲裁
3. 技术实现：5种提示词策略+本地启发式检测
4. 效果数据：准确率提升7%，幻觉率降低15%
```

**代码示例**：
```python
from src.agents.dual_agent import precision_generate

# 双Agent精准模式
result = precision_generate(
    question="INTJ的主导功能是什么？",
    docs=retrieved_docs,
    model_a="glm-4-flash",      # Agent A: 直接策略
    model_b="glm-z1-9b",        # Agent B: 分析策略
    strategy_a="direct",         # 简洁结论优先
    strategy_b="analytical",     # 逐步推理
    fast_mode=True,              # 极速模式：本地检测
    use_self_consistency=True,   # Self-Consistency投票
    sc_samples=3,                # 3次采样
)

# 返回结果
print(f"答案: {result['answer']}")
print(f"一致性: {result.get('consistency', 'N/A')}")
print(f"矛盾检测: {result['verdict']}")
```

**技术亮点**：
1. **5种提示词策略**：direct/analytical/socratic/chain_of_thought/concise
2. **极速模式**：本地启发式检测，0ms延迟
3. **Self-Consistency**：多次采样+多数投票，提升5-15%准确率
4. **A2A协议集成**：Task状态机管理任务生命周期

---

### 4.2 MCP Server实现

**面试问题**："你对MCP协议有什么了解？实际用过吗？"

**回答结构**：
```
1. 协议定义：Model Context Protocol，Anthropic 2024年底推出
2. 核心架构：Host/Client/Server三层
3. 三类能力：Tools(有副作用)/Resources(只读)/Prompts(模板)
4. 传输方式：stdio(本地)/Streamable HTTP(远程，2025-03-26更新)
5. 实际经验：实现了自定义MCP Server，暴露5个工具
```

**代码示例**：
```python
from src.tools.mcp_server import MCPServer, TOOLS, RESOURCES, PROMPTS

# 查看已注册的能力
print(f"Tools: {[t['name'] for t in TOOLS]}")
# ['vector_search', 'exact_match', 'graph_search', 'web_search', 'rag_query']

print(f"Resources: {[r['uri'] for r in RESOURCES]}")
# ['deeprag://collections', 'deeprag://config', 'deeprag://stats']

print(f"Prompts: {[p['name'] for p in PROMPTS]}")
# ['rag_answer', 'fact_check', 'code_review']
```

**启动方式**：
```bash
# stdio模式（Claude Desktop默认）
python start_mcp_server.py

# Streamable HTTP模式（2025-03-26规范）
python -m src.tools.mcp_server --transport http --port 8080
```

**Claude Desktop配置**：
```json
{
  "mcpServers": {
    "deeprag": {
      "command": "python",
      "args": ["start_mcp_server.py"]
    }
  }
}
```

---

### 4.3 A2A协议实现

**面试问题**："多Agent系统怎么设计？Agent间怎么通信？"

**回答结构**：
```
1. 协议定义：Agent-to-Agent，Google 2025年4月发布
2. 核心组件：Agent Card(能力声明) + Task(状态机)
3. 与MCP关系：MCP向下连工具，A2A向外连Agent
4. 实现方式：本地函数调用/远程HTTP服务
```

**代码示例**：
```python
from src.agents.a2a_protocol import get_a2a_protocol, export_agent_card_json

# 获取协议实例
protocol = get_a2a_protocol()

# 查看注册的Agent
for agent in protocol.list_agents():
    print(f"{agent.name}: {agent.description}")
    for skill in agent.skills:
        print(f"  - {skill.name}: {skill.description}")

# 委托任务
task = protocol.delegate_task(
    from_agent="coordinator",
    to_agent="research_agent",
    task_type="deep_search",
    payload={"query": "什么是RAG？"},
)

# 查询状态
status = protocol.get_task_status(task.task_id)
print(f"状态: {status['status']}")  # submitted → working → completed

# 导出Agent Card（/.well-known/agent-card.json）
card = export_agent_card_json("agent-card.json")
```

**预定义Agent**：
| Agent | 技能 | 说明 |
|-------|------|------|
| research_agent | deep_search, vector_search | 多轮知识检索 |
| verify_agent | fact_check, cross_validate | 事实核查 |
| precision_agent | precision_generate | 双Agent精准模式 |

---

### 4.4 Prompt Engineering五要素

**面试问题**："怎么写好Prompt？有什么最佳实践？"

**回答结构**：
```
1. 五要素：Role/Task/Context/Format/Examples
2. 核心原则：固定内容在前，动态内容在后
3. 工程实践：测试集+持续迭代，每次只改一处
4. 项目实现：PromptBuilder + 版本控制
```

**代码示例**：
```python
from src.llm.prompt_templates import PromptBuilder, PromptTemplates

# 使用Builder模式
prompt = (PromptBuilder("rag_answer")
    .role("你是知识库问答专家")
    .task("根据文档回答问题")
    .context("参考文档：...")
    .format("结论 → 证据 → 引用")
    .constraint("只基于文档回答")
    .example("什么是RAG？", "RAG是检索增强生成...")
    .render())

# 使用预定义模板
prompt = PromptTemplates.rag_answer(
    question="INTJ的主导功能是什么？",
    context="...",
    style="detailed"
)
```

---

### 4.5 幻觉防控体系

**面试问题**："大模型幻觉怎么解决？"

**回答结构**：
```
1. 根因分析：训练数据/生成机制/对齐目标三层
2. 缓解方案：训练层/推理层/系统层三层组合
3. 项目实现：事实核查+引用验证+双Agent对比
4. 效果数据：幻觉率从20%降到5%
```

**三层缓解**：
```python
# 1. 推理层：Self-Consistency多次采样
from src.agents.dual_agent import self_consistency_generate
result = self_consistency_generate(question, docs, n_samples=3)

# 2. 系统层：事实核查
from src.agents.fact_checker import check_facts
fact_result = check_facts(answer, source_docs)
# 返回: {"hallucination_score": 0.15, "passed": true, ...}

# 3. 系统层：引用验证
from src.agents.citation_validator import validate_citations
citation_result = validate_citations(answer, source_docs)
# 返回: {"citation_rate": 0.85, "orphan_claims": [...]}
```

---

### 4.6 语义缓存与成本优化

**面试问题**："怎么降低LLM API成本？"

**回答结构**：
```
1. 语义缓存：相似问题命中缓存，跳过LLM调用
2. Prompt Caching：固定前缀复用KV Cache
3. Token追踪：记录每次调用的token用量
4. 模型路由：按任务类型选择性价比最优模型
```

**代码示例**：
```python
from src.llm.semantic_cache import get_semantic_cache
from src.llm.prompt_cache import get_prompt_cache_manager

# 语义缓存
cache = get_semantic_cache()
cached = cache.get("INTJ的主导功能是什么？")
if cached:
    return cached  # 命中缓存，跳过LLM调用

# Prompt Cache
manager = get_prompt_cache_manager()
stats = manager.get_stats()
print(f"缓存命中率: {stats['hit_rate']}%")

# Gateway统一入口
from src.llm.gateway import get_gateway
gateway = get_gateway()
metrics = gateway.get_metrics()
print(f"总调用: {metrics['total_calls']}")
print(f"成功率: {metrics['success_rate']}%")
print(f"P50延迟: {metrics['p50_latency_ms']}ms")
```

---

## 五、简历素材（STAR法则）

### 5.1 项目经历写法

**版本1（详细版）**：
```
DeepRAG 企业级 Multi-Agent RAG 系统 | 项目负责人
2025.11 - 2026.06 | 独立开发

【项目背景】
企业级知识库问答系统，解决大模型幻觉问题和知识更新滞后问题。

【技术方案】
• 设计多Agent协作架构：Research Agent负责知识检索，Verify Agent负责事实核查，
  Precision Agent实现双Agent并行+矛盾检测
• 实现A2A协议（Agent-to-Agent），支持Agent Card能力声明和Task状态机，
  实现跨Agent任务委托和异步协作
• 实现MCP Server（Model Context Protocol），将5个核心工具封装为标准MCP服务，
  支持stdio和Streamable HTTP两种传输方式
• 设计五要素Prompt模板系统（Role/Task/Context/Format/Examples），
  支持版本控制和A/B测试
• 实现Self-Consistency多次采样投票机制，通过3次采样+多数投票提升推理准确率5-15%
• 设计语义缓存系统，基于向量相似度匹配相似问题，缓存命中率35%，成本降低64%

【技术栈】
Python, LangChain, LangGraph, ChromaDB, Ollama, qwen2.5:7b,
bge-base-zh-v1.5, MCP Protocol, A2A Protocol, Streamlit

【项目成果】
• 准确率从60%提升到95%（Top-5召回率，+35%）
• 幻觉率从20%降低到5%（-75%）
• 平均响应时间<2秒（P50: 1.2s, P90: 2.0s）
• API成本降低64%（语义缓存+Prompt Caching）
• 代码规模7,600+行核心代码，17,800+行完整代码
```

**版本2（精简版）**：
```
DeepRAG 企业级 Multi-Agent RAG 系统 | 项目负责人
2025.11 - 2026.06

• 设计多Agent协作架构，实现A2A协议和MCP Server，支持标准化工具服务和跨Agent通信
• 实现双Agent精准模式+Self-Consistency投票，准确率从60%提升到95%，幻觉率降至5%
• 设计语义缓存+Prompt Caching系统，API成本降低64%，响应时间<2秒
• 实现五要素Prompt模板系统+约束解码，支持版本控制和JSON Schema强制输出
• 技术栈：Python, LangChain, ChromaDB, Ollama, qwen2.5:7b, MCP/A2A协议
```

---

### 5.2 技能点写法

**LLM/RAG相关**：
```
• 熟悉RAG系统架构，有检索增强生成、混合检索、重排序、幻觉检测等工程实践经验
• 熟悉MCP/A2A协议，有自定义MCP Server和多Agent协作系统开发经验
• 熟悉Prompt Engineering，有五要素模板设计、Few-shot、CoT等实践经验
• 熟悉LLM推理优化，有KV Cache、量化、约束解码等工程实践经验
• 熟悉Agent系统设计，有Function Calling、Skill系统、Tool Use等开发经验
```

**工程相关**：
```
• 熟悉LLM Gateway设计，有多模型路由、限流熔断、语义缓存等实践经验
• 熟悉大模型评测，有RAGAS/LLM-as-Judge/业务测试集等评测体系建设经验
• 熟悉模型部署，有Ollama/vLLM/SGLang等框架使用经验
• 熟悉成本优化，有Token追踪、Prompt Caching、模型路由等降本经验
```

---

### 5.3 面试常问问题及回答

#### Q1: 介绍一下你的项目？

**30秒版**：
> "我做了一个企业级RAG系统，核心亮点三个：一是混合检索把准确率从60%提升到95%，二是双Agent精准模式+Self-Consistency把幻觉率从20%降到5%，三是语义缓存+Prompt Caching降低成本64%。"

**2分钟版**：
> "这是一个企业级Multi-Agent RAG系统，7个月开发，7600+行核心代码。
>
> 架构上分四层：最上层是Streamlit前端和SSE流式接口，中间是LLM Gateway做统一入口、限流熔断和语义缓存，下面是Agent协作层，用A2A协议管理多Agent通信，最底层是混合检索引擎，支持BM25+向量+图谱+Web四种检索方式。
>
> 核心技术有三个：第一是双Agent精准模式，两个Agent用不同策略并行生成答案，然后对比检测矛盾，一致就融合，矛盾就仲裁或重新检索；第二是Self-Consistency多次采样投票，同一个问题采样3次，取票数最多的答案，提升5-15%准确率；第三是语义缓存，用向量相似度匹配相似问题，命中缓存就跳过LLM调用，成本降低64%。
>
> 工程上做了很多优化：MCP Server把工具封装成标准服务，A2A协议支持Agent间任务委托，五要素Prompt模板系统支持版本控制，约束解码强制JSON输出。"

---

#### Q2: 你是怎么解决幻觉问题的？

> "幻觉的根因有三层：训练数据有噪声、生成机制是续写不是查询、对齐目标鼓励不拒答。
>
> 我的缓解方案也是三层组合：
>
> 推理层用Self-Consistency多次采样投票，同一个问题采样3次取多数答案，因为正确答案更容易通过不同推理路径得到。
>
> 系统层有两个：一是事实核查，用LLM对比生成的答案和源文档，检测不支持的断言；二是引用验证，要求每个事实性断言标注来源，未标注的视为幻觉。
>
> 效果上，幻觉率从v1.0的20%降到现在的5%。但要注意，幻觉不可能完全消除，因为它是LLM概率生成机制的固有产物，工程目标是降低发生率+让用户能识别。"

---

#### Q3: MCP和Function Calling有什么区别？

> "这两个不是同一层面的东西。Function Calling是调用语言，定义模型怎么表达我要调哪个函数、参数是什么。MCP是工具生态协议，定义工具怎么标准化封装、注册和被AI客户端发现。
>
> MCP底层其实还是靠Function Calling驱动。当MCP Client连上Server后，会自动拉取工具定义，转换成Function Calling格式传给模型。模型通过tool_calls触发调用，MCP Client再路由到对应的Server执行。
>
> 打个比方，Function Calling像HTTP请求格式，MCP像REST API规范加服务注册发现机制。两者是上下层的配合关系，不是竞争关系。
>
> 我的项目里两个都用了：MCP Server把5个核心工具封装成标准服务，Function Calling做模型和工具的通信。"

---

#### Q4: 你怎么做的模型选型？

> "模型选型不是看排行榜，是看合规、成本、延迟、能力四个维度。
>
> 我用的是模型路由策略：主调度节点用结构化输出稳定的模型，高频推理节点用性价比高的模型，特别难的问题路由给能力更强但更贵的模型。
>
> 具体来说，qwen2.5:7b做默认模型，平衡质量和速度；GLM-4-Flash做高频任务，速度快成本低；GLM-Z1-9B做复杂推理，质量最好但慢。
>
> 合规底线是：敏感数据尽量留在合规可控的链路里，国内ToB项目里数据出境合规是死线。"

---

#### Q5: 你对A2A协议有什么了解？

> "A2A是Google 2025年4月发布的开放协议，专门解决多个AI Agent之间怎么通信协作的问题。
>
> 它和MCP不是竞争关系：MCP是Agent向下连工具和数据，A2A是Agent向外连其他Agent，一纵一横各管一层。
>
> 核心组件有两个：Agent Card是能力声明JSON，类似API文档，声明Agent能做什么；Task是任务状态机，submitted→working→completed/failed，支持异步长任务。
>
> 我的项目实现了A2A协议，预定义了3个Agent：Research Agent负责知识检索，Verify Agent负责事实核查，Precision Agent实现双Agent精准模式。Agent Card支持导出为/.well-known/agent-card.json格式。"

---

## 六、技术深度追问准备

### 6.1 KV Cache相关

**Q: KV Cache是什么？Prompt Caching和它什么关系？**

> "KV Cache是Transformer推理的核心优化。自回归生成时，每生成新token都要对前面所有token算attention，KV Cache把前面token的K/V矩阵缓存起来，新token只算自己的部分，复杂度从O(N³)降到O(N²)。
>
> Prompt Caching是KV Cache在时间维度的延伸：单次内的KV Cache在token之间共享，Prompt Caching在不同请求之间共享相同前缀的KV Cache。
>
> 核心工程原则是：固定内容在前、动态内容在后。把System Prompt放最前面，用户查询放最后面，这样前缀稳定，每次都能命中缓存。"

---

### 6.2 量化相关

**Q: INT8和INT4量化有什么区别？AWQ和GPTQ呢？**

> "INT8通常损失很小，接近免费午餐；INT4是甜蜜点，损失1-3%，体积压到1/4。INT3开始明显冒险，INT2一般不推荐。
>
> AWQ和GPTQ是两种量化算法。GPTQ是逐层量化+误差补偿，数学严谨支持极端低位；AWQ是激活感知+重要权重保护，1%关键权重承担99%输出贡献，推理速度比GPTQ快1.5-2倍。
>
> 要注意，GGUF是文件格式不是量化算法，llama.cpp用的，里面可以存各种量化方案的权重。"

---

### 6.3 MoE相关

**Q: 什么是MoE？为什么DeepSeek V3用MoE？**

> "MoE把Transformer的FFN层替换成N个并行的专家网络，加一个Router选Top-K个处理每个token。
>
> 核心设计哲学是总参数大但激活参数小。DeepSeek V3总参数671B，但每个token只激活37B，约5.5%。这样能用671B的知识量+37B的推理成本，达到Dense模型做不到的学得多+跑得快。
>
> 显存占用按总参数走（所有专家都要加载），但推理速度按激活参数走。这是MoE反直觉但精确的关键点。"

---

## 七、项目版本演进

| 版本 | 时间 | 准确率 | 核心技术 | 代码量 |
|------|------|--------|---------|--------|
| v0.1 | 2025.11 | 60% | 简单RAG | 111行 |
| v1.0 | 2025.12 | 75% | Corrective RAG | 300行 |
| v2.0 | 2026.02 | 82% | Self-RAG | 500行 |
| v2.1 | 2026.03-05 | 88% | 混合检索 | 800行 |
| v2.2 | 2026.06.07 | 95% | 增强检索 | 1,867行 |
| v2.3 | 2026.06.08 | 95% | LLMOps工程化 | +810行 |
| v2.9.2 | 2026.07.13 | 95% | A2A+MCP+Self-Consistency | 7,600+行 |

---

## 八、关键调优参数（实验得出）

| 参数 | 最优值 | 测试范围 | 说明 |
|------|--------|---------|------|
| chunk_size | 800 | 200/500/800/1200/1500 | 文档分块大小 |
| chunk_overlap | 200 | 100/150/200/250 | 25%重叠 |
| max_retries | 2 | 1/2/3 | 收益递减 |
| hallucination_threshold | 0.3 | 0.1-0.5 | 50个答案调优 |
| semantic_cache_threshold | 0.90 | 0.85-0.95 | 相似度阈值 |
| temperature_generation | 0.3 | 0.1-0.7 | 生成任务 |
| temperature_fact_check | 0.0 | 0.0 | 事实核查确定性 |

---

## 九、参考资料

### 论文
- [RAG原始论文](https://arxiv.org/abs/2005.11401)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [MoE综述](https://arxiv.org/abs/2101.03961)

### 协议规范
- [MCP协议](https://spec.modelcontextprotocol.io)
- [A2A协议](https://github.com/google/A2A)

### 框架文档
- [LangChain](https://python.langchain.com)
- [vLLM](https://docs.vllm.ai)
- [SGLang](https://sgl-project.github.io)

---

## 十、面试注意事项

### ✅ 要做到
1. **用数据说话**：准确率95%、幻觉率5%、成本-64%
2. **讲清权衡**：为什么选这个方案，放弃了什么
3. **展示深度**：能回答追问，知道底层原理
4. **承认边界**：幻觉不可能完全消除，量化有精度损失

### ❌ 要避免
1. **不要背概念**：面试官想听你的理解，不是教科书
2. **不要夸大**：实测数据比理论值更有说服力
3. **不要回避问题**：不知道就说不知道，不要编
4. **不要只说好处**：每个技术都有代价，要讲清楚

---

**最后更新**: 2026-07-13
**版本**: v2.9.2
**作者**: DeepRAG项目组
