import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';

// 구성 요소

const template = ChatPromptTemplate.fromMessages([
  ['system', '당신은 친절한 어시스턴트입니다.'],
  ['human', '{question}'],
]);

const model = new ChatOpenAI({
  model: 'gpt-3.5-turbo',
});

// 함수로 결합한다
// RunnableLambda로 작성한 함수에 Runnable 인터페이스를 추가한다

const chatbot = RunnableLambda.from(async (values) => {
  const prompt = await template.invoke(values);
  return await model.invoke(prompt);
});

// 사용한다

const response = await chatbot.invoke({
  question: '거대 언어 모델은 어디서 제공하나요?',
});
console.log(response);
