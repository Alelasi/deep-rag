"""
DeepRAG Web UI - Streamlit 交互界面

功能：
1. 文档上传与解析
2. 实时问答（流式输出）
3. 多轮对话（会话管理）
4. 检索结果展示
"""

import streamlit as st
import requests
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import time

# 页面配置
st.set_page_config(
    page_title="DeepRAG - 企业级知识库问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 配置
API_BASE_URL = "http://localhost:8000"

# 会话状态初始化
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


def upload_document(file) -> Dict:
    """上传文档到知识库"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def query_rag(question: str, session_id: str, stream: bool = True):
    """查询 RAG 系统（支持流式输出）"""
    try:
        payload = {
            "question": question,
            "session_id": session_id,
            "stream": stream
        }

        if stream:
            # SSE 流式输出
            response = requests.post(
                f"{API_BASE_URL}/api/query/stream",
                json=payload,
                stream=True,
                timeout=60
            )
            return response
        else:
            # 普通请求
            response = requests.post(
                f"{API_BASE_URL}/api/query",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_session_history(session_id: str) -> List[Dict]:
    """获取会话历史"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/sessions/{session_id}/history",
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("history", [])
    except Exception:
        return []


def clear_session(session_id: str):
    """清空会话历史"""
    try:
        requests.delete(
            f"{API_BASE_URL}/api/sessions/{session_id}",
            timeout=10
        )
        return True
    except Exception:
        return False


# ===== 侧边栏 =====
with st.sidebar:
    st.title("📚 知识库管理")

    # 文档上传
    st.subheader("上传文档")
    uploaded_file = st.file_uploader(
        "支持 PDF、Markdown、TXT",
        type=["pdf", "md", "txt"],
        help="上传文档后将自动解析并加入知识库"
    )

    if uploaded_file:
        if st.button("开始上传", type="primary"):
            with st.spinner("正在上传并解析文档..."):
                result = upload_document(uploaded_file)
                if result.get("success"):
                    st.success(f"✅ {uploaded_file.name} 上传成功！")
                    st.session_state.uploaded_docs.append({
                        "name": uploaded_file.name,
                        "size": uploaded_file.size,
                        "doc_id": result.get("doc_id")
                    })
                else:
                    st.error(f"❌ 上传失败：{result.get('error')}")

    # 已上传文档列表
    st.subheader("已上传文档")
    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.text(f"📄 {doc['name']} ({doc['size']/1024:.1f} KB)")
    else:
        st.info("暂无文档，请先上传")

    st.divider()

    # 会话管理
    st.subheader("⚙️ 会话设置")
    st.text(f"会话 ID: {st.session_state.session_id[:8]}...")

    if st.button("清空对话历史"):
        clear_session(st.session_state.session_id)
        st.session_state.chat_history = []
        st.success("✅ 对话历史已清空")

    if st.button("新建会话"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.success("✅ 新会话已创建")

    st.divider()

    # 检索配置
    st.subheader("🔧 检索配置")
    retrieval_mode = st.selectbox(
        "检索模式",
        ["混合检索（推荐）", "向量检索", "关键词检索"],
        help="混合检索 = BM25 + 向量检索 + RRF 融合"
    )

    top_k = st.slider("返回文档数", 1, 10, 5)
    use_reranker = st.checkbox("启用 Reranker 重排序", value=True)


# ===== 主界面 =====
st.title("🤖 DeepRAG - 企业级知识库问答系统")
st.caption("基于 Agentic RAG + 混合检索 + Self-RAG 自我反思")

# 显示对话历史
chat_container = st.container()

with chat_container:
    for chat in st.session_state.chat_history:
        # 用户问题
        with st.chat_message("user"):
            st.write(chat["question"])

        # 系统回答
        with st.chat_message("assistant"):
            st.write(chat["answer"])

            # 显示检索来源
            if "sources" in chat and chat["sources"]:
                with st.expander("📚 查看检索来源"):
                    for i, source in enumerate(chat["sources"], 1):
                        st.markdown(f"**[{i}] {source['title']}**")
                        st.text(f"相似度: {source['score']:.3f}")
                        st.caption(source['content'][:200] + "...")
                        st.divider()

# 输入框
question = st.chat_input("请输入您的问题...")

if question:
    # 显示用户问题
    with st.chat_message("user"):
        st.write(question)

    # 显示 AI 回答（流式输出）
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources = []

        # 流式获取回答
        try:
            response = query_rag(question, st.session_state.session_id, stream=True)

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = json.loads(line[6:])

                        if data.get("type") == "token":
                            # 逐字显示
                            full_response += data.get("content", "")
                            response_placeholder.markdown(full_response + "▌")

                        elif data.get("type") == "sources":
                            # 保存来源
                            sources = data.get("sources", [])

                        elif data.get("type") == "done":
                            # 完成
                            response_placeholder.markdown(full_response)
                            break

            # 显示检索来源
            if sources:
                with st.expander("📚 查看检索来源"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**[{i}] {source['title']}**")
                        st.text(f"相似度: {source['score']:.3f}")
                        st.caption(source['content'][:200] + "...")
                        st.divider()

            # 保存到历史
            st.session_state.chat_history.append({
                "question": question,
                "answer": full_response,
                "sources": sources
            })

        except Exception as e:
            st.error(f"❌ 查询失败：{str(e)}")
            st.info("💡 提示：请确保后端服务已启动（python -m uvicorn src.api:app）")

# 底部统计信息
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("已上传文档", len(st.session_state.uploaded_docs))
with col2:
    st.metric("对话轮次", len(st.session_state.chat_history))
with col3:
    st.metric("会话时长", f"{len(st.session_state.chat_history) * 2} 分钟（估算）")
