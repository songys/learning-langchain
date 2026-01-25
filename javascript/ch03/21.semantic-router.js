import { cosineSimilarity } from '@langchain/core/utils/math';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';

const physicsTemplate = `당신은 매우 똑똑한 물리학 교수입니다. 
    당신은 물리학에 대한 질문에 간결하고 쉽게 이해할 수 있는 방식으로 대답하는 데 뛰어납니다.
    당신이 질문에 대한 답을 모를 때는 모른다고 인정합니다.
    다음 질문에 답하세요.: {query}`;

const mathTemplate = `당신은 매우 뛰어난 수학자입니다. 당신은 수학 문제에 답하는 데 뛰어납니다.
    당신은 어려운 문제를 구성 요소로 분해하고 구성 요소를 해결한 다음
    함께 모아 더 넓은 질문에 대답합니다.
    다음 질문에 답하세요.: {query}`;

const embeddings = new OpenAIEmbeddings();

const promptTemplates = [physicsTemplate, mathTemplate];

const promptEmbeddings = await embeddings.embedDocuments(promptTemplates);

const promptRouter = RunnableLambda.from(async (query) => {
  // 질문 임베딩
  const queryEmbedding = await embeddings.embedQuery(query);
  // 유사도 계산
  const similarities = cosineSimilarity([queryEmbedding], promptEmbeddings)[0];
  // 입력 질문에 가장 유사한 프롬프트 선택
  const mostSimilar =
    similarities[0] > similarities[1] ? promptTemplates[0] : promptTemplates[1];
  console.log(
    `Using ${mostSimilar === promptTemplates[0] ? 'PHYSICS' : 'MATH'}`
  );
  return PromptTemplate.fromTemplate(mostSimilar).invoke({ query });
});

const semanticRouter = promptRouter.pipe(
  new ChatOpenAI({ modelName: 'gpt-4o-mini', temperature: 0 })
);

const result = await semanticRouter.invoke('블랙홀이란 무엇인가요?');
console.log('\n의미론적 라우팅 결과: ', result);
