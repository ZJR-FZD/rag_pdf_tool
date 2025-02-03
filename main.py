import streamlit as st
from utils import rag_tool

from langchain.memory import ConversationBufferMemory

st.title("智能PDF问答工具")

with st.sidebar:
    api_key = st.text_input("请输入你的OpenAI API密钥：")
    st.markdown("[获取OpenAI API密钥](https://2233.ai/api)")

    uploaded_file = st.file_uploader("请上传你的PDF文件：",type="pdf")


# 初始化会话状态
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
    st.session_state.messages = [
        {
            "role":"ai","content":"你好，我是PDF分析小助手，上传文件向我提问吧！"
        }
    ]

# 显示历史对话消息（初始显示）
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 获取用户输入
question = st.chat_input("对PDF的内容进行提问",disabled=not uploaded_file)
if question:
    if not api_key:
        st.info("请输入你的OpenAI API密钥")
        st.stop()

    # 合法后就显示
    st.session_state.messages.append(
        {"role":"human","content":question}
    )
    st.chat_message("human").write(question)

    # 获取AI的回复
    with st.spinner("AI正在思考中，请稍等···"):
        result = rag_tool(api_key=api_key,memory=st.session_state.memory,uploaded_file=uploaded_file,question=question)
    st.session_state.messages.append(
        {"role":"ai","content":result["answer"]}
    )
    st.chat_message("ai").write(result["answer"])