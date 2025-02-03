from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os

# 加载
loader = TextLoader("./京剧介绍.txt",encoding="utf-8") #加载器，可能要指定编码格式
docs = loader.load()
# 分割
text_splitter = RecursiveCharacterTextSplitter() #分割器
texts = text_splitter.split_documents(docs)
# 嵌入模型
embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),
                                    base_url="https://api.gptsapi.net/v1") #注意：指定模型都要指定api_key，不是官方的还要加上base_url
# 嵌入并储存
db = FAISS.from_documents(texts,embeddings_model)
# 检索
retriever = db.as_retriever() #检索器


# 模型
model = ChatOpenAI(model="gpt-3.5-turbo",
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url="https://api.gptsapi.net/v1")

# 记忆
memory = ConversationBufferMemory(return_messages=True,
                                  memory_key="chat_history",
                                  output_key="answer")


# 创建对话链
chain = ConversationalRetrievalChain.from_llm(
    llm = model,
    retriever = retriever,
    memory = memory,
    return_source_documents=True
)

result = chain.invoke(
    {
        "chat_history":memory,
        "question":"介绍京剧中的旦角"
    }
)

print(result)
print(result["answer"]) #不是result.answer,'dict' object has no attribute 'answer'
print(result["source_documents"])