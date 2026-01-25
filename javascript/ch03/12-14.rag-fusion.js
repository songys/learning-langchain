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
//  문서를 로드 후 분할
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
  `하나의 입력 쿼리를 기반으로 여러 개의 검색 쿼리를 생성하는 유용한 어시스턴트입니다. \n 다음과 관련된 여러 검색 쿼리를 영문으로 생성합니다: {question} \n 출력(쿼리 4개):`
);
const queryGen = perspectivesPrompt.pipe(llm).pipe((message) => {
  return message.content.split('\n');
});

function reciprocalRankFusion(results, k = 60) {

  const fusedScores = {};
  const documents = {};
  results.forEach((docs) => {
    docs.forEach((doc, rank) => {
      const key = doc.pageContent;
      // 문서가 아직 본 적 없으면
      // - 점수를 0으로 초기화
      // - 나중에 사용하기 위해 저장
      if (!(key in fusedScores)) {
        fusedScores[key] = 0;
        documents[key] = 0;
      }
      // RRF 공식을 사용하여 문서의 점수 업데이트
      // 1 / (rank + k)
      fusedScores[key] += 1 / (rank + k);
    });
  });
  // 결합된 점수에 따라 문서를 내림차순으로 정렬하여 최종 재정렬된 결과 가져오기
  const sorted = Object.entries(fusedScores).sort((a, b) => b[1] - a[1]);
  // 각 키에 대한 해당 문서 검색
  return sorted.map(([key]) => documents[key]);
}

const prompt = ChatPromptTemplate.fromTemplate(
  '다음 컨텍스트만 사용해 질문에 답변하세요.\n 컨텍스트: {context}\n\n질문: {question}'
);

const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe(reciprocalRankFusion);

console.log('RAG 융합 실행\n');
const ragFusion = RunnableLambda.from(async (input) => {
  // 관련 문서 검색
  const docs = await retrievalChain.invoke({ question: input });
  // 프롬프트 포매팅
  const formatted = await prompt.invoke({ context: docs, question: input });
  // 답변 생성
  const answer = await llm.invoke(formatted);
  return answer;
});

const result = await ragFusion.invoke(
  '고대 그리스 철학사의 주요 인물은 누구인가요?'
);

console.log(result);
