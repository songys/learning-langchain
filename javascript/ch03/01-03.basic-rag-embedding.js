import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';
import { RunnableLambda } from '@langchain/core/runnables';
const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';

// 문서를 로드 후 분할
const loader = new TextLoader('./test.txt');
const raw_docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splitDocs = await splitter.splitDocuments(raw_docs);

// 각 청크를 임베딩하고 벡터 저장소에 삽입
const model = new OpenAIEmbeddings();

const db = await PGVectorStore.fromDocuments(splitDocs, model, {
  postgresConnectionOptions: {
    connectionString,
  },
});

// 벡터 저장소에서 2개의 관련 문서 검색
const retriever = db.asRetriever({ k: 2 });

const query =
  '고대 그리스 철학사의 주요 인물은 누구인가요?';

// 관련 문서 받아오기
const docs = await retriever.invoke(query);

console.log(
  `유사도 검색 결과:\n ${docs[0].pageContent}\n\n`
);
