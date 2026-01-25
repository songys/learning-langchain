from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# --- 블로그 게시물 인덱싱 ---

urls = [
    "https://blog.langchain.dev/top-5-langgraph-agents-in-production-2024/",
    "https://blog.langchain.dev/langchain-state-of-ai-2024/",
    "https://blog.langchain.dev/introducing-ambient-agents/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 벡터DB에 추가
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# 관련 문서 검색
results = retriever.invoke(
    "2024년에 프로덕션 환경에서 사용된 LangGraph 에이전트 2개는 무엇인가요?")

print("결과: \n", results)


# --- 관련성 평가 ---

# 데이터 모델
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 체크를 위한 평가."""

    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'"
    )


# 구조화된 출력이 있는 LLM
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 프롬프트
system = """당신은 사용자 질문에 대한 검색된 문서의 관련성을 평가하는 채점자입니다.
문서에 질문과 관련된 키워드나 의미가 포함되어 있다면 관련성이 있다고 평가하세요.
문서가 질문과 관련이 있는지 여부를 나타내는 'yes' 또는 'no'로 평가해 주세요."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human",
         "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# --- 검색된 문서 평가 ---

question = "2024년에 프로덕션에서 사용된 LangGraph 에이전트 2개는 무엇인가요?"

# 예시로 retrieval_grader.invoke({"question": question, "document": doc_txt})
docs = retriever.invoke(question)

doc_txt = docs[0].page_content

result = retrieval_grader.invoke({"question": question, "document": doc_txt})

print("\n\n평가 결과: \n", result)
