# deep-rag Multi-Agent 优化与评测方案 v2.1

> **目标**：优化现有 5 个 Agent 协作 + 增加 Multi-Agent 评测系统  
> **时间**：2026-06-05  
> **预计完成**：2-3 小时

---

## 🎯 核心目标

### B. 优化现有Agent协作

1. **增强通信机制**
   - Agent 间消息传递（Message Passing）
   - 共享状态管理（Shared State）
   - 事件总线（Event Bus）

2. **优化决策路由**
   - 动态路由器（根据查询类型选择Agent）
   - 置信度评分（每个Agent返回置信度）
   - 智能降级（失败时自动切换策略）

3. **失败重试机制**
   - 指数退避重试（Exponential Backoff）
   - 熔断器模式（Circuit Breaker）
   - 降级策略（Fallback）

### D. Multi-Agent评测系统

1. **协作效率指标**
   - Agent 调用次数
   - 总耗时 vs 串行耗时（并行度）
   - 消息传递次数

2. **通信成本指标**
   - Token 消耗（每个 Agent）
   - API 调用次数
   - 平均响应时间

3. **决策准确率指标**
   - Agent 决策正确率
   - 路由准确率
   - 降级触发率

---

## 📐 系统架构设计

### 当前架构（v2.0）

```
用户查询
    ↓
QueryAgent (分析查询)
    ↓
GraderAgent (评分文档)
    ↓
GeneratorAgent (生成答案)
    ↓
FactCheckerAgent (事实校验)
    ↓
ConflictDetectorAgent (冲突检测)
    ↓
输出答案
```

### 优化后架构（v2.1）

```
用户查询
    ↓
QueryAgent (分析查询 + 置信度)
    ↓
SmartRouter (智能路由 + 降级策略) ← [新增]
    ↓
[并行执行]
    ├─ GraderAgent (评分 + 置信度)
    ├─ RetrievalAgent (检索 + 置信度)
    └─ [共享状态] MessageBus ← [新增]
    ↓
GeneratorAgent (生成 + 置信度)
    ↓
[并行验证]
    ├─ FactCheckerAgent (事实校验)
    └─ ConflictDetectorAgent (冲突检测)
    ↓
MetricsCollector (收集指标) ← [新增]
    ↓
输出答案 + 评测报告
```

---

## 🛠️ 实现计划

### Phase 1: 增强通信机制（30 分钟）

**文件**：`src/agents/coordination/message_bus.py`（新建）

**核心功能**：
```python
class MessageBus:
    """Agent 间消息总线"""
    
    def publish(self, event: str, data: dict):
        """发布事件"""
        
    def subscribe(self, event: str, handler: callable):
        """订阅事件"""
        
    def get_shared_state(self, key: str):
        """获取共享状态"""
        
    def set_shared_state(self, key: str, value: any):
        """设置共享状态"""
```

---

### Phase 2: 优化决策路由（45 分钟）

**文件**：`src/agents/coordination/smart_router.py`（新建）

**核心功能**：
```python
class SmartRouter:
    """智能路由器（带置信度和降级）"""
    
    def route(self, query: str, confidence: float) -> str:
        """根据查询类型和置信度决定路由"""
```

**路由策略**：
1. 高置信度查询（>0.8）→ 快速流程
2. 中等置信度（0.5-0.8）→ 标准流程
3. 低置信度（<0.5）→ 增强流程

---

### Phase 3: 失败重试机制（30 分钟）

**文件**：`src/agents/coordination/retry_handler.py`（新建）

---

### Phase 4: Multi-Agent 评测系统（60 分钟）

**文件**：`src/evaluation/multi_agent_metrics.py`（新建）

**核心指标**：

#### 1. 协作效率指标
- 总调用Agent数
- 总耗时 vs 串行耗时（并行度）
- 消息传递次数

#### 2. 通信成本指标
- Token 消耗（每个 Agent）
- API 调用次数
- 成本（美元）

#### 3. 决策准确率指标
- Agent 决策正确率
- 路由准确率
- 降级触发率

---

## 📊 评测报告示例

```
================================================================================
Multi-Agent 评测报告
================================================================================

【协作效率】
  并行度: 1.96x ⭐
  总耗时: 2.3秒
  消息传递: 12次

【通信成本】
  总Token: 1,250
  成本: $0.0125

【决策准确率】
  路由准确率: 92% ✅
  Agent平均准确率: 90.2%

【各Agent表现】
  QueryAgent: 95% (0.3秒, 200 tokens)
  GraderAgent: 88% (0.5秒, 300 tokens)
  GeneratorAgent: 90% (0.6秒, 500 tokens)
  FactCheckerAgent: 93% (0.4秒, 150 tokens)
  ConflictDetectorAgent: 85% (0.5秒, 100 tokens)
================================================================================
```

---

## 🚀 实施步骤

### Step 1: 创建文件结构（10 分钟）
### Step 2: 实现核心功能（2 小时）
### Step 3: 编写测试（30 分钟）
### Step 4: 集成验证（30 分钟）

---

## 💡 面试价值提升

### 优化前（v2.0）
> "我实现了一个 RAG 系统，有 5 个 Agent 协作"

### 优化后（v2.1）⭐⭐⭐⭐⭐
> "我实现了企业级 Multi-Agent RAG 系统，包括："
> 
> **1. 智能协作机制**：消息总线、智能路由、失败重试
> **2. 完整评测体系**：协作效率、通信成本、决策准确率
> **3. 工程化实践**：20+ 测试用例、覆盖率 >90%

---

**预计完成时间**：2-3 小时
