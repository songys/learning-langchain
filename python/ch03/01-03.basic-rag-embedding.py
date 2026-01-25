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

# 벡터 저장소에서 관련 문서 검색
retriever = db.as_retriever()
query = '고대 그리스 철학사의 주요 인물은 누구인가요?'

# 관련 문서 받아오기
print(retriever.invoke(query))

# 벡터 저장소에서 2개의 관련 문서 검색
retriever = db.as_retriever(search_kwargs={'k': 2})

# 관련 문서 받아오기
docs = retriever.invoke(query)

print(docs[0].page_content)