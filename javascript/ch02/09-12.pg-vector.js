/** 
 도커 실행 명령어
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
*/

import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { v4 as uuidv4 } from 'uuid';

// 도커 연결 설정
const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';

// 문서를 로드 후 분할
const loader = new TextLoader('./test.txt');
const raw_docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await splitter.splitDocuments(raw_docs);

// 문서에 대한 임베딩 생성
const model = new OpenAIEmbeddings();
const db = await PGVectorStore.fromDocuments(docs, model, {
  postgresConnectionOptions: {
    connectionString,
  },
});

console.log('벡터 저장소 생성 완료');

const results = await db.similaritySearch('query', 4);

console.log(`유사도 검색 결과: ${JSON.stringify(results)}`);

console.log('문서를 벡터 저장소에 저장');

const ids = [uuidv4(), uuidv4()];

await db.addDocuments(
  [
    {
      pageContent: 'there are cats in the pond',
      metadata: { location: 'pond', topic: 'animals' },
    },
    {
      pageContent: 'ducks are also found in the pond',
      metadata: { location: 'pond', topic: 'animals' },
    },
  ],
  { ids }
);

console.log('문서 저장 성공');

await db.delete({ ids: [ids[1]] });

console.log('문서 삭제 성공');
