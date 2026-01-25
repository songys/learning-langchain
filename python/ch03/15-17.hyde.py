from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser

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

# 벡터 저장소에서 5개의 관련 문서 검색
retriever = db.as_retriever(search_kwargs={'k': 5})

prompt_hyde = ChatPromptTemplate.from_template(
    '''
질문에 답할 구절을 영문으로 작성해 주세요.
질문: {question}

구절:''')

generate_doc = (prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser())

'''
위에서 생성한 가상 문서를 검색의 입력으로 사용하여 임베딩을 생성하고 벡터 저장소에서 유사한 문서를 검색
'''
retrieval_chain = generate_doc | retriever

query = '고대 그리스 철학사에서 잘 알려지지 않은 철학자는 누구인가요?'

prompt = ChatPromptTemplate.from_template(
    '''
다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트:{context}

질문: {question}
'''
)

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)


@chain
def qa(input):
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke({'context': docs, 'question': input})
    answer = llm.invoke(formatted)
    return answer


print('HyDE 실행\n')
result = qa.invoke(query)
print('\n\n')
print(result.content)
