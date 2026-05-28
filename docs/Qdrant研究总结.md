# Qdrant 深度研究总结

**研究时间**：2026-05-24  
**版本**：qdrant-client 1.18.0  
**模式**：in-memory（无需 Docker）

---

## 1. 核心发现

### 1.1 关键突破

✅ **无需 Docker 即可运行真实 Qdrant**
- `:memory:` 模式：纯内存，适合测试和小型应用
- `path="./local_qdrant"` 模式：本地持久化，适合开发环境
- 生产环境才需要 Docker 部署

✅ **Query API 替代已废弃的 search()**
- `search()` 方法在 v1.10+ 已废弃
- 新 API：`query_points()` + `Prefetch` + `FusionQuery`
- 支持真正的混合检索（dense + sparse + RRF）

✅ **真实混合检索实现**
- 之前的 `hybrid_search` 是假实现（只用了 dense）
- 现在：Prefetch 并行检索 → RRF 融合 → 返回融合结果
- 面试中的关键技术点

---

## 2. 技术实现

### 2.1 混合检索（Hybrid Search）

**核心代码**：
```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion

results = client.query_points(
    collection_name="my_collection",
    prefetch=[
        # 阶段1：稠密向量检索（语义相似）
        Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
        # 阶段2：稀疏向量检索（关键词匹配）
        Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2),
    ],
    # 阶段3：RRF 融合
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
)
```

**工作流程**：
1. **Prefetch 阶段**：并行检索 dense 和 sparse 向量（各取 top_k * 2）
2. **Fusion 阶段**：RRF (Reciprocal Rank Fusion) 融合两个结果集
3. **返回**：融合后的 top_k 结果

**RRF 公式**：
```
score(doc) = Σ 1 / (k + rank_i)
```
- `k = 60`（默认常数）
- `rank_i`：文档在第 i 个检索结果中的排名

---

### 2.2 稀疏向量生成（BM25 风格）

**实现方式**：
```python
import jieba
from collections import Counter
from qdrant_client.models import SparseVector

def generate_sparse_vector(text: str):
    # 1. 中文分词
    tokens = list(jieba.cut(text))
    
    # 2. 词频统计
    word_counts = Counter(tokens)
    
    # 3. 转换为稀疏向量
    indices = []
    values = []
    for word, count in word_counts.items():
        idx = hash(word) % 10000  # 词的 hash 作为索引
        indices.append(idx)
        values.append(float(count))
    
    return SparseVector(indices=indices, values=values)
```

**特点**：
- 基于词频（TF），简化版 BM25
- 中文分词用 jieba
- 索引空间：10000 维（hash 取模）

---

### 2.3 元数据过滤

**代码示例**：
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# 构建过滤器
query_filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="psychology")),
        FieldCondition(key="source", match=MatchValue(value="mbti.md")),
    ]
)

# 检索时应用过滤
results = client.query_points(
    collection_name="my_collection",
    query=dense_vector,
    query_filter=query_filter,
    limit=5,
)
```

---

## 3. 测试覆盖

### 3.1 Mock 单元测试（10 个）

**文件**：`tests/test_qdrant_retriever.py`

| 测试 | 覆盖点 |
|------|--------|
| test_create_collection | 集合创建 |
| test_create_collection_idempotent | 幂等性 |
| test_add_documents | 批量插入 |
| test_generate_sparse_vector | 稀疏向量生成 |
| test_search_basic | 基础检索 |
| test_search_with_filters | 元数据过滤 |
| test_search_empty | 空结果处理 |
| test_hybrid_search | 混合检索 |
| test_delete_collection | 集合删除 |
| test_qdrant_unavailable_raises | 依赖检查 |

**状态**：✅ 10/10 通过

---

### 3.2 真实集成测试（2 个）

**文件**：`tests/test_qdrant_real_integration.py`

| 测试 | 覆盖点 |
|------|--------|
| test_real_hybrid_search_chinese | 中文混合检索 + RRF 融合 |
| test_real_metadata_filtering | 元数据过滤 |

**特点**：
- 使用真实 Qdrant 客户端（`:memory:` 模式）
- 中文文档 + jieba 分词
- 验证 RRF 融合结果

**状态**：✅ 2/2 通过

---

## 4. 面试话术

### 4.1 Qdrant 是什么

**错误回答**：
> "Qdrant 是一个向量数据库，可以把文本转成向量。"

**正确回答**：
> "Qdrant 是一个高性能向量数据库，**它不负责生成向量**，只负责存储、索引和检索向量。向量生成由外部的 embedding 模型（如 BGE、sentence-transformers）完成。Qdrant 的核心优势是：
> 1. **Rust 实现**：内存安全、低占用、无 GC 停顿
> 2. **HNSW 索引**：近似最近邻搜索，速度快
> 3. **混合检索**：支持 dense + sparse 向量 + RRF 融合
> 4. **强大过滤**：支持复杂的元数据过滤
> 5. **易部署**：Docker 一键启动，或 in-memory 模式开发"

---

### 4.2 你真实跑过 Qdrant 吗？

**错误回答**：
> "我写了接口，单元测试都通过了。"

**正确回答**：
> "我不仅写了接口和 mock 单元测试，还用 **in-memory 模式跑了真实集成测试**。测试覆盖：
> 1. **中文混合检索**：jieba 分词 → sparse vector → RRF 融合
> 2. **元数据过滤**：验证 Filter + FieldCondition
> 3. **Query API**：用最新的 `query_points()` + `Prefetch` 替代已废弃的 `search()`
> 
> 虽然生产环境我还没部署 Docker 版，但核心逻辑已经在 in-memory 模式下验证通过，迁移到 Docker 只需改一行代码（`:memory:` → `host:port`）。"

---

### 4.3 Hybrid Search 怎么实现的？

**错误回答**：
> "我调用了 Qdrant 的 hybrid_search 方法。"

**正确回答**：
> "Qdrant 没有单独的 `hybrid_search()` 方法，混合检索是通过 **Query API** 实现的：
> 
> 1. **Prefetch 阶段**：并行检索 dense 和 sparse 向量
>    - Dense：语义相似（cosine）
>    - Sparse：关键词匹配（BM25 风格）
> 2. **Fusion 阶段**：RRF (Reciprocal Rank Fusion) 融合
>    - 公式：`score = Σ 1/(k + rank_i)`，k=60
>    - 自动平衡两种检索结果
> 3. **返回**：融合后的 top_k 结果
> 
> 我之前的实现有个 bug：只调用了 dense 检索，sparse 向量虽然生成了但没参与融合。后来我重写了，用 `Prefetch` + `FusionQuery` 实现了真正的混合检索。"

---

### 4.4 为什么选 Qdrant 而不是其他向量库？

**对比表**：

| 数据库 | 优势 | 劣势 | 适用场景 |
|--------|------|------|---------|
| **Qdrant** | 性能最强、过滤强、Rust 内存低 | Hybrid 需配置 sparse | 高性能 Hybrid RAG |
| **pgvector** | 统一 SQL、ACID 事务、轻量 | 纯向量性能不是最顶尖 | 小中型 RAG + 已有 Postgres |
| **Milvus** | 亿级规模、GPU 加速 | 部署重、小项目杀鸡用牛刀 | 超大规模 RAG |
| **Weaviate** | Hybrid 最开箱即用、GraphQL | 学习曲线、资源比 Qdrant 高 | 复杂 Hybrid + 开发体验优先 |
| **ChromaDB** | 嵌入式、Python 友好 | 性能一般、不适合生产 | 快速原型、小型 RAG |

**我的选择**：
- **开发/测试**：Qdrant in-memory（无需 Docker，快速迭代）
- **生产**：Qdrant Docker（性能最强，过滤强大）
- **如果已有 Postgres**：pgvector（统一数据库，降低维护成本）

---

## 5. 遗留问题与下一步

### 5.1 已完成 ✅

- [x] 安装 qdrant-client 1.18.0
- [x] 重写 `hybrid_search`（真实 RRF 融合）
- [x] 升级 `search()` 到 Query API
- [x] Mock 单元测试（10/10 通过）
- [x] 真实集成测试（2/2 通过）
- [x] 中文分词 + 稀疏向量生成

### 5.2 待完成 ⏳

- [ ] Docker 部署验证（需重启电脑激活 Hyper-V）
- [ ] BGE-zh 中文 embedding 集成（替换随机向量）
- [ ] 性能基准测试（vs ChromaDB / LanceDB）
- [ ] 接入 deep-rag 主 Pipeline（`graph.py`）
- [ ] 写技术博客《Qdrant Hybrid Search 实战》

---

## 6. 关键代码位置

| 文件 | 说明 |
|------|------|
| `src/retrieval/qdrant_retriever.py` | Qdrant 检索器（真实混合检索） |
| `tests/test_qdrant_retriever.py` | Mock 单元测试（10 个） |
| `tests/test_qdrant_real_integration.py` | 真实集成测试（2 个） |
| `src/config.py` | Qdrant 配置（host/port/collection） |

---

## 7. 参考资料

- [Qdrant 官方文档](https://qdrant.tech/documentation/)
- [Query API 文档](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [Hybrid Search 最佳实践](https://qdrant.tech/articles/hybrid-search/)
- [RRF 融合算法](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

**最后更新**：2026-05-24  
**维护者**：wzy
