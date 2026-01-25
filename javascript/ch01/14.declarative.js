import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

// 구성 요소

const template = ChatPromptTemplate.fromMessages([
  ['system', '당신은 친절한 어시스턴트입니다.'],
  ['human', '{question}'],
]);

const model = new ChatOpenAI({
  model: 'gpt-3.5-turbo',
  streaming: true, // 스트리밍 활성화
});

// 함수에서 구성 요소 결합

const chatbot = template.pipe(model);

// 사용한다

const response = await chatbot.invoke({
  question: '거대 언어 모델은 어디서 제공하나요?',
});

console.log(response);