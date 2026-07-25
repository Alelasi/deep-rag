# DeepRAG 多维度实测报告

- 时间：2026-07-18T10:32:00
- 样本数：39
- 等级：**B**
- 准确率(pass)：**97.4%**（38/39）
- 综合分 mean/median：7.037 / 7.4（满分 10）
- 说明：关键词+规则启发式评分，含系统 hallucination/fact_check 字段；非人工金标，也非官方 RAGAS 全量；勿直接写简历 95%。

## 核心指标

| 指标 | 值 |
|---|---:|
| 准确率_pass | 97.4% |
| 综合分_mean | 7.037 |
| 关键词命中率_mean | 0.876 |
| 准确性_mean | 8.761 |
| 完整性_mean | 6.534 |
| 相关性_mean | 3.103 |
| 引用质量_mean | 9.0 |
| 速度分_mean | 4.872 |
| 反幻觉_mean | 6.576 |
| 事实校验通过率 | 69.2% |
| no_knowledge率 | 10.3% |
| 拒识题正确率 | 100.0% |
| 有引用率 | 100.0% |
| 错误率 | 0.0% |
| 空答率 | 0.0% |
| 延迟_mean_s | 13.095 |
| 延迟_p50_s | 12.453 |
| 延迟_p90_s | 20.865 |
| 延迟_max_s | 40.43 |
| 3秒内占比 | 7.7% |
| 10秒内占比 | 41.0% |
| 20秒内占比 | 84.6% |
| 系统幻觉分_mean | 0.2706 |
| 平均检索文档数 | 4.62 |
| 平均相关文档数 | 4.26 |
| 平均答案长度 | 305.0 |
| mock_web率 | 0.0% |

## 分类 pass / 综合分

| 类别 | pass | 综合分 |
|---|---:|---:|
| Agent | 100.0% | 7.91 |
| MBTI | 100.0% | 7.366 |
| RAG | 100.0% | 6.212 |
| cross | 100.0% | 7.635 |
| enneagram | 100.0% | 7.18 |
| finance | 50.0% | 5.35 |
| refuse | 100.0% | 8.21 |
| social | 100.0% | 6.83 |
| thesis | 100.0% | 7.203 |

## 难度

| 难度 | pass | 综合分 |
|---|---:|---:|
| easy | 100.0% | 7.448 |
| hard | 80.0% | 6.272 |
| medium | 100.0% | 6.815 |

## 知识库

| collection | pass |
|---|---:|
| proj_psychology | 100.0% |
| proj_social | 100.0% |
| proj_thesis | 100.0% |
| proj_work | 93.8% |

## 分题明细

| id | ok | total | acc | lat_s | nk | fc | q |
|---|---|---:|---:|---:|---|---|---|
| mbti_01 | True | 7.47 | 10.0 | 12.916 | False | True | 什么是MBTI？ |
| mbti_02 | True | 7.49 | 10.0 | 15.375 | False | False | INTJ的主导功能是什么？ |
| mbti_03 | True | 7.68 | 10.0 | 6.333 | False | True | INTJ功能堆栈顺序是什么？ |
| mbti_04 | True | 7.33 | 10.0 | 20.007 | False | True | 八个认知功能有哪些？ |
| mbti_05 | True | 6.98 | 10.0 | 20.452 | False | False | Te和Ti有什么区别？ |
| mbti_06 | True | 7.75 | 10.0 | 7.493 | False | True | Fe和Fi有什么不同？ |
| mbti_07 | True | 7.28 | 10.0 | 17.347 | False | True | Si是什么认知功能？ |
| mbti_08 | True | 7.14 | 10.0 | 19.58 | False | False | Ne外倾直觉的特点是什么？ |
| mbti_09 | True | 7.34 | 10.0 | 12.399 | False | True | Se外倾感觉关注什么？ |
| mbti_10 | True | 7.06 | 10.0 | 22.518 | False | True | INTP的主导功能通常是什么？ |
| mbti_11 | True | 7.44 | 10.0 | 8.435 | False | True | 什么是内倾直觉Ni？ |
| mbti_12 | True | 7.44 | 10.0 | 8.179 | False | True | ENFP的主导功能是什么？ |
| mbti_13 | True | 7.14 | 10.0 | 40.43 | False | False | MBTI的四个维度分别是什么？ |
| mbti_14 | True | 7.58 | 10.0 | 9.008 | False | True | 什么是阴影功能？ |
| mbti_15 | True | 7.27 | 10.0 | 12.411 | False | False | 九型人格有哪九型？ |
| mbti_16 | True | 7.09 | 10.0 | 12.84 | False | False | 九型中完美主义者是哪一型？ |
| work_01 | True | 7.67 | 10.0 | 8.916 | False | True | 什么是RAG？ |
| work_02 | True | 5.6 | 5.0 | 13.82 | False | False | 什么是混合检索？ |
| work_03 | True | 7.66 | 10.0 | 13.221 | False | True | 什么是Self-RAG？ |
| work_04 | True | 6.2 | 5.0 | 6.568 | False | True | 什么是Corrective RAG？ |
| work_05 | True | 7.4 | 10.0 | 15.886 | False | False | DeepRAG项目用了什么向量数据库？ |
| work_06 | True | 5.68 | 5.0 | 14.645 | False | True | 什么是Reranker精排？ |
| work_07 | True | 7.98 | 10.0 | 18.935 | False | False | 什么是Agentic RAG？ |
| work_08 | True | 7.84 | 10.0 | 7.597 | False | True | 什么是LangGraph？ |
| work_09 | True | 5.57 | 5.0 | 18.637 | False | True | RAG系统如何降低幻觉？ |
| work_10 | True | 2.6 | 0.0 | 10.153 | True | False | chunk_size一般怎么选？ |
| work_11 | True | 7.62 | 10.0 | 13.021 | False | True | 什么是BM25算法？ |
| work_12 | True | 6.12 | 5.0 | 7.625 | False | True | 什么是向量检索？ |
| fg_01 | True | 7.76 | 10.0 | 9.313 | False | True | 什么是恐惧贪婪指数？ |
| fg_02 | False | 2.94 | 0.0 | 25.393 | False | False | 恐贪指数算法大致怎么计算？ |
| refuse_01 | True | 8.21 | 10.0 | 0.002 | True | True | 本知识库里有没有介绍量子纠缠实验设备型号？ |
| refuse_02 | True | 8.21 | 10.0 | 0.001 | True | True | 请给出不存在的公司DeepRAG宇宙总部地址门牌号 |
| refuse_03 | True | 8.21 | 10.0 | 0.001 | True | True | INTJ昨天中午在火星吃了什么？ |
| cross_01 | True | 7.36 | 10.0 | 12.453 | False | True | 请用知识库内容说明MBTI与九型如何结合使用 |
| cross_02 | True | 7.91 | 10.0 | 7.005 | False | True | DeepRAG里的Self-RAG和幻觉检测大概怎么做？ |
| social_01 | True | 6.83 | 10.0 | 24.793 | False | False | 什么是制度经济学？ |
| thesis_01 | True | 7.66 | 10.0 | 8.367 | False | True | 类别不平衡常见处理方法有哪些？ |
| thesis_02 | True | 6.52 | 6.67 | 9.774 | False | True | 混淆矩阵是什么？ |
| thesis_03 | True | 7.43 | 10.0 | 18.864 | False | True | 什么是F1分数？ |

## 失败样例（最多 12）

- **fg_02** `恐贪指数算法大致怎么计算？` total=2.94 kw=0.0 refuse=False preview='\n1. 【直接回答】恐贪指数算法以RSI(14)为基础，结合动态情绪因子调整，其中速度因子根据5日或10日跌幅超过阈值进行恐慌值修正[1]。\n\n2. 【详细解释】基础恐贪指数通过计算14日相对强弱指数（RSI）实现，RSI公式为平均收盘价上'