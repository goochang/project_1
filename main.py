from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 모델 초기화
model = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o-mini")

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("resource/ai_trand.pdf")

# 페이지 별 문서 로드
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",           # 각 청크를 구분하기 위한 기준 문자열
    chunk_size=100,             # 각 청크의 최대 길이
    chunk_overlap=10,           # 인접 청크 사이에 중복으로 포함될 문자수
    length_function=len,        # 청크의 길이를 계산하는 함수
    is_separator_regex=False,
)

splits = text_splitter.split_documents(docs)
# print(splits)
# recursive_text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10,
#     length_function=len,
#     is_separator_regex=False,
# )
# splits = recursive_text_splitter.split_documents(docs)
# print(splits)

# OpenAI 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
# 문서 리스트를 텍스트로 변환하는 단계 추가
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

# RAG 체인에서 각 단계마다 DebugPassThrough 추가
rag_chain_debug = {
    "context": retriever,                    # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
}  | DebugPassThrough() | ContextToText()|   contextual_prompt | model


while True: 
    print("========================")
    query = input("질문을 입력하세요: ")
    if query == "quit":
        break

    response = rag_chain_debug.invoke(query)
    print("Final Response:")
    print(response.content)