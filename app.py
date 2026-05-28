"""DeepRAG — Streamlit 演示界面"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import streamlit as st
from pathlib import Path

from src.graph import query as rag_query, get_indexer

st.set_page_config(page_title="DeepRAG", page_icon="🔍", layout="wide")

# === 侧边栏 ===
with st.sidebar:
    st.title("🔍 DeepRAG")
    st.markdown("**自纠错多源知识Agent**")
    st.markdown("Corrective RAG + Self-RAG")
    st.markdown("---")

    st.subheader("知识库管理")
    collection = st.text_input("集合名称", value="demo_kb")
    doc_dir = st.text_input("文档目录", value="data/sample_docs")

    if st.button("📥 索引文档"):
        indexer = get_indexer(collection)
        count = indexer.index_directory(doc_dir)
        st.success(f"已索引 {count} 个文档块")
        st.session_state["indexed"] = True

    st.markdown("---")
    st.subheader("参数设置")
    max_retries = st.slider("最大重试次数", 1, 5, 2)

# === 主界面 ===
st.title("🔍 DeepRAG 知识问答")

tab1, tab2, tab3 = st.tabs(["💬 问答", "🏗️ 架构", "📊 评估指标"])

with tab1:
    question = st.text_input("请输入问题", value="INTJ的主导功能是什么？")

    if st.button("🚀 查询", type="primary"):
        if not st.session_state.get("indexed"):
            indexer = get_indexer(collection)
            count = indexer.index_directory(doc_dir)
            st.info(f"自动索引了 {count} 个文档块")

        with st.spinner("Pipeline 运行中..."):
            result = rag_query(question, collection_name=collection, max_retries=max_retries)

        # 答案
        st.subheader("答案")
        st.markdown(result.get("answer", "未生成答案"))

        # 指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("幻觉评分", f"{result.get('hallucination_score', 0):.2f}")
        col2.metric("引用数", len(result.get("citations", [])))
        col3.metric("相关文档", result.get("relevant_count", 0))
        col4.metric("冲突数", len(result.get("conflicts", [])))

        # 引用详情
        citations = result.get("citations", [])
        if citations:
            st.subheader("📎 引用来源")
            for c in citations:
                st.caption(f"[{c.get('source', '?')} p.{c.get('page', '?')}] {c.get('text', '')[:200]}")

        # 冲突
        conflicts = result.get("conflicts", [])
        if conflicts:
            st.subheader("⚠️ 多源冲突")
            for cf in conflicts:
                st.warning(f"**{cf['topic']}**: {cf.get('resolution', '')}")

        # Pipeline历史
        with st.expander("📜 Pipeline执行历史"):
            for h in result.get("history", []):
                st.text(f"  → {h}")

        # 事实校验
        unsupported = result.get("unsupported_claims", [])
        if unsupported:
            st.error(f"未被文档支持的断言: {unsupported}")

with tab2:
    st.header("🏗️ 7层Pipeline架构")
    st.markdown("""
    ```
    用户提问
        ↓
    [1. Query分析] → 判断类型 + 改写查询
        ↓
    [2. 混合检索] → BM25(关键词) + 向量(语义) + RRF融合
        ↓
    [3. 文档评分] → Corrective RAG: 逐文档评分(relevant/ambiguous/irrelevant)
        ↓
        ├── 有relevant → [5. 生成]
        ├── 无relevant + 可重试 → [4. Query改写] → 回到[2]（纠错循环）
        └── 重试耗尽 → [4b. Web搜索兜底] → [5. 生成]
        ↓
    [5. 答案生成] → 带引用标注 [来源:文件, 第N块]
        ↓
    [6. 事实校验] → Self-RAG: 逐句比对源文档，检测幻觉
        ↓
        ├── 通过(score<0.3) → [7. 冲突检测]
        └── 不通过 → 回到[5]重新生成（Self-RAG循环）
        ↓
    [7. 冲突解决] → 标注分歧点 + 各方证据 + 置信度排序
        ↓
    输出: 答案 + 引用 + 幻觉评分 + 冲突报告
    ```

    **技术决策**：
    - **为什么有两个循环？** Corrective RAG纠正检索质量，Self-RAG纠正生成质量，解决不同层的问题
    - **为什么用RRF融合？** 不需要归一化不同检索器的分数（BM25分数和向量距离量级不同），对异常值鲁棒
    - **为什么有Web Fallback？** 承认知识边界比瞎编好，知识库没答案时搜索兜底
    - **为什么检测多源冲突？** 企业知识库同一主题可能有多版本文档，直接忽略矛盾会误导用户
    """)

with tab3:
    st.header("📊 RAG评估指标")
    st.markdown("""
    | 指标 | 计算方式 | 含义 |
    |------|----------|------|
    | **precision** | relevant / total_retrieved | 检索准确率 |
    | **faithfulness** | 1 - hallucination_score | 生成忠实度 |
    | **citation_density** | citations / answer_length | 引用密度 |
    | **completeness** | 基于答案长度和覆盖度 | 回答完整度 |
    | **overall** | 加权综合 | 总体质量(0-100) |
    """)

    st.markdown("""
    **vs 竞品**：
    - Dify/MaxKB/FastGPT：检索到就用（无文档评分），一次生成（无自纠错），无冲突检测
    - DeepRAG：Corrective RAG文档评分 + Self-RAG事实校验 + 查询改写重试 + 多源冲突检测
    """)
