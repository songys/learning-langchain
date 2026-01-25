import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const routeQuery = z
  .object({
    datasource: z
      .enum(['python_docs', 'js_docs'])
      .describe(
        'Given a user question, choose which datasource would be most relevant for answering their question'
      ),
  })
  .describe('Route a user query to the most relevant datasource.');

const llm = new ChatOpenAI({ model: 'gpt-4o-mini', temperature: 0 });

const structuredLlm = llm.withStructuredOutput(routeQuery, {
  name: 'RouteQuery',
});

const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    `당신은 사용자 질문을 적절한 데이터 소스로 라우팅하는 전문가입니다. 질문이 지목하는 프로그래밍 언어에 따라 해당 데이터 소스로 라우팅하세요.`,
  ],
  ['human', '{question}'],
]);

const router = prompt.pipe(structuredLlm);

const question = `이 코드가 안 돌아가는 이유를 설명해주세요: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
prompt.invoke("french") `;

const result = await router.invoke({ question });

console.log('Routing to: ', result);

const chooseRoute = (result) => {
  if (result.datasource.toLowerCase().includes('python_docs')) {
    return 'chain for python_docs';
  } else {
    return 'chain for js_docs';
  }
};

const fullChain = router.pipe(chooseRoute);

const finalResult = await fullChain.invoke({ question });

console.log('Choose route: ', finalResult);
