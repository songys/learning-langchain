import os
import dotenv

dotenv.load_dotenv()

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client
from langchain_core.documents import Document

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()


vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# 문서 예시
document1 = Document(
    page_content="The powerhouse of the cell is the mitochondria",
    metadata={"source": "https://example.com"}
)

document2 = Document(
    page_content="Buildings are made out of brick", 
    metadata={"source": "https://example.com"}
)

documents = [document1, document2]

# 데이터베이스에 데이터 저장
vector_store.add_documents(documents, ids=["1", "2"])

## 유사도 검색 테스트

query = "biology"
matched_docs = vector_store.similarity_search(query)

print(matched_docs[0].page_content)
