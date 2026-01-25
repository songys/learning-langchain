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

/**
 * 검색된 문서를 LLM에 제공하여 사용자의 질문에 답변
 */
const prompt = ChatPromptTemplate.fromTemplate(
  '다음 컨텍스트만 사용해 질문에 답변하세요.\n 컨텍스트: {context}\n\n질문: {question}'
);

const llm = new ChatOpenAI({ temperature: 0, modelName: 'gpt-4o-mini' });
const chain = prompt.pipe(llm);

const result = await chain.invoke({
  context: docs,
  question: query,
});

console.log(result);
console.log('\n\n');

// 이번에는 효율성을 위해 로직을 캡슐화한 후 재실행.

console.log(
  '이번에는 효율성을 위해 로직을 캡슐화한 후 재실행\n'
);
const qa = RunnableLambda.from(async (input) => {
  // 관련 문서 검색
  const docs = await retriever.invoke(input);
  // 프롬프트 포매팅
  const formatted = await prompt.invoke({ context: docs, question: input });
  // 답변 생성
  const answer = await llm.invoke(formatted);
  return answer;
});

const finalResult = await qa.invoke(query);
console.log(finalResult);
