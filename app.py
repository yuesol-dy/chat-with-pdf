import streamlit as st
import tempfile
import os
from utils import extract_text_from_pdf, split_text_into_chunks
from rag_pipeline import create_vector_store, load_vector_store, build_qa_chain

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with PDF – 本地文档智能问答")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ----- 侧边栏：上传文件 -----
with st.sidebar:
    st.header("📤 上传 PDF 文件")
    uploaded_files = st.file_uploader(
        "选择 PDF 文件（可多选）",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("处理文件"):
            all_chunks = []
            for f in uploaded_files:
                # 将上传文件写入临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getbuffer())
                    tmp_path = tmp.name
                # 提取文本并切块
                text = extract_text_from_pdf(tmp_path)
                os.unlink(tmp_path)  # 删除临时文件
                chunks = split_text_into_chunks(text)
                all_chunks.extend(chunks)
            if all_chunks:
                st.session_state.vectorstore = create_vector_store(all_chunks)
                st.success(f"✅ 已处理 {len(uploaded_files)} 个文件，创建 {len(all_chunks)} 个文本块")
            else:
                st.warning("⚠️ 未能从文件中提取到文字")
    # 如果本地已有向量库，自动加载
    if st.session_state.vectorstore is None and os.path.exists("./chroma_db"):
        st.session_state.vectorstore = load_vector_store()
        st.sidebar.info("📂 已加载本地向量数据库")

# ----- 对话区域 -----
st.divider()
# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 输入框
if prompt := st.chat_input("请输入你的问题"):
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 生成回答
    if st.session_state.vectorstore is None:
        response = "请先上传 PDF 文件并点击「处理文件」。"
    else:
        qa_chain = build_qa_chain(st.session_state.vectorstore)
        with st.spinner("🤔 思考中..."):
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]
    # 记录助手回复
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)