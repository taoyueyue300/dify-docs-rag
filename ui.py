"""
Streamlit前端界面 - Dify文档RAG助手
启动: streamlit run ui.py
"""
import streamlit as st
from chain import RAGChain

st.set_page_config(page_title="Dify文档助手", page_icon="📚", layout="wide")
st.title("📚 Dify 文档智能问答助手")
st.caption("基于 RAG (混合检索 + Reranker) 构建")


@st.cache_resource
def load_chain():
    return RAGChain()


chain = load_chain()

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 引用来源"):
                for s in msg["sources"]:
                    st.code(s, language=None)

# 用户输入
if prompt := st.chat_input("输入你的问题，例如：如何本地部署Dify？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            result = chain.query(prompt)
        st.markdown(result["answer"])
        with st.expander(f"📄 引用来源 ({result['retrieved_chunks']} 个文档片段)"):
            for s in result["sources"]:
                st.code(s, language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
