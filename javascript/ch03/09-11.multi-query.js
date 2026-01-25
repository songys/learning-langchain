/** 
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. Run the following command to start the postgres container:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
3. Use the connection string below for the postgres container
*/

import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';
import { RunnableLambda } from '@langchain/core/runnables';

const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';

const loader = new TextLoader('./test.txt');
const raw_docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splitDocs = await splitter.splitDocuments(raw_docs);

const model = new OpenAIEmbeddings();

const db = await PGVectorStore.fromDocuments(splitDocs, model, {
  postgresConnectionOptions: {
    connectionString,
  },
});

// 벡터 저장소에서 5개의 관련 문서 검색
const retriever = db.asRetriever({ k: 5 });

const llm = new ChatOpenAI({ temperature: 0, modelName: 'gpt-4o-mini' });

const perspectivesPrompt = ChatPromptTemplate.fromTemplate(
  `당신은 AI 언어 모델 어시스턴트입니다. 주어진 사용자 질문의 다섯 가지 버전을 생성하여 벡터 데이터베이스에서 관련 문서를 검색하세요. 사용자 질문에 대한 다양한 관점을 생성함으로써 사용자가 거리 기반 유사도 검색의 한계를 극복할 수 있도록 돕는 것이 목표입니다. 이러한 대체 질문을 개행으로 구분하여 제공하세요. 원래 질문: {question}`
);

const queryGen = perspectivesPrompt.pipe(llm).pipe((message) => {
  return message.content.split('\n');
});


const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe((documentLists) => {
    const dedupedDocs = {};
    documentLists.flat().forEach((doc) => {
      dedupedDocs[doc.pageContent] = doc;
    });
    return Object.values(dedupedDocs);
  });

const prompt = ChatPromptTemplate.fromTemplate(
  '다음 컨텍스트만 사용해 질문에 답변하세요.\n 컨텍스트: {context}\n\n질문: {question}'
);

console.log('다중 쿼리 검색\n');
const multiQueryQa = RunnableLambda.from(async (input) => {
  // 관련 문서 검색
  const docs = await retrievalChain.invoke({ question: input });
  // 프롬프트 포매팅
  const formatted = await prompt.invoke({ context: docs, question: input });
  // 답변 생성
  const answer = await llm.invoke(formatted);
  return answer;
});

const result = await multiQueryQa.invoke(
  '고대 그리스 철학사의 주요 인물은 누구인가요?'
);

console.log(result);
