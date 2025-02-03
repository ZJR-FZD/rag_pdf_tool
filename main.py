import streamlit as st
from utils import rag_tool

from langchain.memory import ConversationBufferMemory

st.title("智能PDF问答工具")

with st.sidebar:
    api_key = st.text_input("请输入你的OpenAI API密钥：",type="password")
    st.markdown("[获取OpenAI API密钥](https://2233.ai/api)")

    # 上传文件
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
    st.session_state.documents = []

# 显示历史对话消息（初始显示）和历史资料
num = 0
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    if message["role"]=="ai":
        if message["content"]!="你好，我是PDF分析小助手，上传文件向我提问吧！": #注意超范围问题，第一句ai消息没有相应的相关资料
            with st.expander("相关资料"):
                st.write(st.session_state.documents[num][0].page_content) 
            num += 1

# 获取用户输入
question = st.chat_input("对PDF的内容进行提问")
if question:
    if not api_key:
        st.info("请输入你的OpenAI API密钥")
        st.stop()
    if not uploaded_file:
        st.info("请先上传文件！")
        st.stop()

    # 合法后就显示
    st.session_state.messages.append(
        {"role":"human","content":question}
    )
    st.chat_message("human").write(question)

    # 获取AI的回复
    with st.spinner("AI正在思考中，请稍等···"):
        result = rag_tool(api_key=api_key,memory=st.session_state.memory,uploaded_file=uploaded_file,question=question)

    answer = result["answer"]
    st.session_state.messages.append(
        {"role":"ai","content":answer}
    )
    st.chat_message("ai").write(answer)

    relavant_docs = result["source_documents"]
    st.session_state.documents.append(relavant_docs)
    with st.expander("相关资料"):
        st.write(relavant_docs[0].page_content) #Document(id='1abe1f48-fd94-445f-b3ff-02b1b5f12f29', metadata={'source': './京剧介绍.txt'},page_content='***') 这种里面的元素是对象的属性，不是键值