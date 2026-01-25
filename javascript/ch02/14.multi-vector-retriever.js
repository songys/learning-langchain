import * as uuid from 'uuid';
import { MultiVectorRetriever } from 'langchain/retrievers/multi_vector';
import { OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { InMemoryStore } from '@langchain/core/stores';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { Document } from '@langchain/core/documents';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';

const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';
const collectionName = 'summaries';

const textLoader = new TextLoader('./test.txt');
const parentDocuments = await textLoader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 10000,
  chunkOverlap: 20,
});
const docs = await splitter.splitDocuments(parentDocuments);

const prompt = PromptTemplate.fromTemplate(
  `다음 문서의 요약을 생성하세요:\n\n{doc}`
);

const llm = new ChatOpenAI({ modelName: 'gpt-4o-mini' });

const chain = RunnableSequence.from([
  { doc: (doc) => doc.pageContent },
  prompt,
  llm,
  new StringOutputParser(),
]);

const summaries = await chain.batch(docs, {
  maxConcurrency: 5,
});

const idKey = 'doc_id';
const docIds = docs.map((_) => uuid.v4());
// 각 요약은 doc_id를 통해 원본 문서와 연결
const summaryDocs = summaries.map((summary, i) => {
  const summaryDoc = new Document({
    pageContent: summary,
    metadata: {
      [idKey]: docIds[i],
    },
  });
  return summaryDoc;
});

// 기존 청크를 저장할 바이트스토어
const byteStore = new InMemoryStore();

// 요약을 저장할 벡터 저장소
const vectorStore = await PGVectorStore.fromDocuments(
  docs,
  new OpenAIEmbeddings(),
  {
    postgresConnectionOptions: {
      connectionString,
    },
  }
);

const retriever = new MultiVectorRetriever({
  vectorstore: vectorStore,
  byteStore,
  idKey,
});

const keyValuePairs = docs.map((originalDoc, i) => [docIds[i], originalDoc]);

// retriever로 청크를 문서저장소에 추가
await retriever.docstore.mset(keyValuePairs);

// 벡터 저장소가 요약을 검색
const vectorstoreResult = await retriever.vectorstore.similaritySearch(
  'chapter on philosophy',
  2
);
console.log(`summary: ${vectorstoreResult[0].pageContent}`);
console.log(
  `summary retrieved length: ${vectorstoreResult[0].pageContent.length}`
);

// retriever는 더 큰 원본 문서 청크를 반환
const retrieverResult = await retriever.invoke('chapter on philosophy');
console.log(
  `multi-vector retrieved chunk length: ${retrieverResult[0].pageContent.length}`
);
