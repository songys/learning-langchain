from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain


connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

# 문서를 로드 후 분할
raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# 문서에 대한 임베딩 생성
embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

# 벡터 저장소에서 2개의 관련 문서 검색
retriever = db.as_retriever(search_kwargs={'k': 2})

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'

# 관련 문서 받아오기
docs = retriever.invoke(query)

print(docs[0].page_content)

prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트:{context}

질문: {question}
'''
)

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
llm_chain = prompt | llm

# 관련 문서를 사용한 답변
result = llm_chain.invoke({'context': docs, 'question': query})

print(result)
print('\n\n')

# 이번에는 효율성을 위해 로직을 캡슐화한 후 재실행.

# 데코레이터 @chain는 이 함수를 LangChain runnable로 변환하여
# LangChain의 chain 작업과 파이프라인과 호환되게 함

print('이번에는 효율성을 위해 로직을 캡슐화한 후 재실행\n')


@chain
def qa(input):
    # 관련 문서 검색
    docs = retriever.invoke(input)
    # 프롬프트 포매팅
    formatted = prompt.invoke({'context': docs, 'question': input})
    # 답변 생성
    answer = llm.invoke(formatted)
    return answer
    # return {"answer": answer, "docs": docs}

# 실행
result = qa.invoke(query)
print(result.content)
