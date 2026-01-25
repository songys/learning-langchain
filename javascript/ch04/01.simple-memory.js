import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';

const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    '당신은 친절한 어시스턴트입니다. 모든 질문에 최선을 다해 답하세요.',
  ],
  ['placeholder', '{messages}'],
]);
const model = new ChatOpenAI({modelName: 'gpt-4o-mini'});
const chain = prompt.pipe(model);

const response = await chain.invoke({
  messages: [
    [
      'human',
      '다음 한국어 문장을 프랑스어로 번역하세요.: 나는 프로그래밍을 좋아해요.',
    ],
    ['ai', 'J\'adore programmer.'],
    ['human', '뭐라고 말했죠?'],
  ],
});

console.log(response.content);
