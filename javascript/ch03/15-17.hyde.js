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
// 문서를 로드 후 분할
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

/**
 * 관련 문서를 LLM에 제공하여 사용자의 질문에 답변
 */
const llm = new ChatOpenAI({ temperature: 0, modelName: 'gpt-4o-mini' });

const hydePrompt = ChatPromptTemplate.fromTemplate(
  `질문에 답할 구절을 영문으로 작성해 주세요.\n 질문: {question} \n 구절:`
);

const generatedDoc = hydePrompt.pipe(llm).pipe((msg) => msg.content);

const retrievalChain = generatedDoc.pipe(retriever);

const prompt = ChatPromptTemplate.fromTemplate(
  '다음 컨텍스트만 사용해 질문에 답변하세요.\n 컨텍스트: {context}\n\n질문: {question}'
);

console.log('hyde 실행\n');
const hydeQa = RunnableLambda.from(async (input) => {
  // 관련 문서 검색
  const docs = await retrievalChain.invoke({ question: input });
  // 프롬프트 포매팅
  const formatted = await prompt.invoke({ context: docs, question: input });
  // 답변 생성
  const answer = await llm.invoke(formatted);
  return answer;
});

const result = await hydeQa.invoke(
  '고대 그리스 철학사에서 잘 알려지지 않은 철학자는 누구인가요?'
);

console.log(result);
