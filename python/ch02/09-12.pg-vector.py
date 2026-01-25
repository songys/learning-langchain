'''
도커 실행 명령어
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
'''


from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid


# 도커 연결 설정
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

results = db.similarity_search('query', k=4)

print(results)

print('문서를 벡터 저장소에 저장')
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [
        Document(
            page_content='there are cats in the pond',
            metadata={'location': 'pond', 'topic': 'animals'},
        ),
        Document(
            page_content='ducks are also found in the pond',
            metadata={'location': 'pond', 'topic': 'animals'},
        ),
    ],
    ids=ids,
)

print('문서 저장 성공.\n  전체 문서 수:',
      len(db.get_by_ids(ids)))

print('id로 문서 삭제\n  ', ids[1])
db.delete({'ids': ids})

print('문서 삭제 성공.\n  전체 문서 수:',
      len(db.get_by_ids(ids)))
