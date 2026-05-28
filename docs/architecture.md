# 架构设计文档

## 核心技术决策

### 1. 为什么用Corrective RAG而不是直接生成

**问题**：传统RAG检索到垃圾文档也照样生成——40%的生产RAG系统在检索阶段就失败了。

**方案**：在检索和生成之间插入一个文档评分层（doc_grader），逐文档判断relevant/ambiguous/irrelevant。只有relevant的文档才进入生成。

**面试追问**：
> Q: 评分本身也需要调LLM，不是更慢了吗？
> A: 离线模式用关键词覆盖率评分（零成本），在线模式用LLM但prompt很短（单文档+问题），延迟<500ms。相比"生成一个错误答案再被用户投诉"的成本，值得。

### 2. 为什么有两个纠错循环

**循环1: Corrective RAG**（检索层纠错）
- 触发条件：文档评分全部irrelevant
- 动作：改写查询→重新检索
- 上限：max_retries次

**循环2: Self-RAG**（生成层纠错）
- 触发条件：事实校验hallucination_score >= 0.3
- 动作：重新生成（LangGraph条件边回到generate节点）
- 上限：max_retries次

两个循环在不同层解决不同问题——检索层保证"找到对的文档"，生成层保证"基于文档说对的话"。

### 3. 为什么用RRF而不是简单加权

**Reciprocal Rank Fusion** 的优势：
- 不需要归一化不同检索器的分数（BM25分数和向量距离量级不同）
- 对异常值鲁棒（一个检索器给出异常高分不会主导结果）
- 被微软Bing和多个学术论文验证有效

公式：`RRF_score = sum(1/(k+rank))` across all lists, k=60

### 4. Web Fallback的设计

**原则**：承认知识边界比瞎编好。

当知识库检索+查询改写都失败后，触发外部搜索作为兜底。但Web结果进入生成时也要经过同样的事实校验。

生产环境可接Tavily/Serper API，当前用占位实现。

### 5. 多源冲突解决

**真实场景**：企业知识库中同一个主题可能有多个版本的文档（如v3.6说相关度0.075，v5.3说0.9158），直接忽略矛盾会误导用户。

**方案**：
1. 检测矛盾（同一关键词不同数值/说法）
2. 列出各方证据和来源
3. 基于证据强度给置信度排序
4. 不替用户做选择，而是展示双方

### 6. 评估指标设计

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| precision | relevant/total_retrieved | 检索准确率 |
| faithfulness | 1 - hallucination_score | 生成忠实度 |
| citation_density | citations/answer_length | 引用密度 |
| completeness | 基于答案长度 | 回答完整度 |
| overall | 加权综合 | 总体质量(0-100) |
