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

// 문서에 대한 임베딩 생성
const model = new OpenAIEmbeddings();

const db = await PGVectorStore.fromDocuments(splitDocs, model, {
  postgresConnectionOptions: {
    connectionString,
  },
});

// 관련 문서 2개를 검색하는 검색기
const retriever = db.asRetriever({ k: 2 });

/**
 * 관련 질문을 하기 전에 관련 없는 정보로 시작하는 쿼리
 */
const query =
  '일어나서 이를 닦고 뉴스를 읽었어요. 그러다 전자레인지에 음식을 넣어둔 걸 깜빡했네요. 고대 그리스 철학사의 주요 인물은 누구인가요?';
/**
 * 관련 문서를 컨텍스트로 제공하여 사용자의 질문에 답변
 */
const prompt = ChatPromptTemplate.fromTemplate(
  '다음 컨텍스트만 사용해 질문에 답변하세요.\n 컨텍스트: {context}\n\n질문: {question}'
);

const llm = new ChatOpenAI({ temperature: 0, modelName: 'gpt-4o-mini' });

const qa = RunnableLambda.from(async (input) => {
  // 관련 문서 검색
  const docs = await retriever.invoke(input);
  // 프롬프트 포매팅
  const formatted = await prompt.invoke({ context: docs, question: input });
  // 답변 생성
  const answer = await llm.invoke(formatted);
  return { answer, docs };
});

const result = await qa.invoke(query);
console.log(result);
console.log('\n\n정확도를 높이기 위해 쿼리를 재작성\n\n');

const rewritePrompt = ChatPromptTemplate.fromTemplate(
  `웹 검색 엔진이 주어진 질문에 답할 수 있도록 더 나은 영문 검색어를 제공하세요. 쿼리는 '**'로 끝내세요.\n\n 질문: {question} 답변:`
);

const rewriter = rewritePrompt.pipe(llm).pipe((message) => {
  return message.content.replaceAll('"', '').replaceAll('**');
});

const rewriterQA = RunnableLambda.from(async (input) => {
  const newQuery = await rewriter.invoke({ question: input }); 
  const docs = await retriever.invoke(newQuery); 
  const formatted = await prompt.invoke({ context: docs, question: input }); 
  const answer = await llm.invoke(formatted);
  return answer;
});

const finalResult = await rewriterQA.invoke(query);
console.log(finalResult);
