import {
  AIMessage,
  SystemMessage,
  HumanMessage,
} from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
  END,
} from '@langchain/langgraph';

const model = new ChatOpenAI({model: 'gpt-4o-mini'});

const annotation = Annotation.Root({
  messages: Annotation({ reducer: messagesStateReducer, default: () => [] }),
});

const generatePrompt = new SystemMessage(
  `당신은 훌륭한 3단락 에세이를 작성하는 임무를 가진 에세이 어시스턴트입니다.
사용자의 요청에 맞춰 최상의 에세이를 작성하세요.
사용자가 비평을 제공하면, 이전 시도에 대한 수정 버전을 응답하세요.`
);

async function generate(state) {
  const answer = await model.invoke([generatePrompt, ...state.messages]);
  return { messages: [answer] };
}

const reflectionPrompt = new SystemMessage(
  `당신은 에세이 제출물을 평가하는 교사입니다. 사용자의 제출물에 대해 비평과 추천을 생성하세요.
길이, 깊이, 스타일 등과 같은 구체적인 요구사항을 포함한 자세한 추천을 제공하세요.`
);

async function reflect(state) {
  // 메시지들을 반전시켜 LLM이 자신의 출력을 성찰하도록 합니다.
  const clsMap = {
    ai: HumanMessage,
    human: AIMessage,
  };
  // 첫 번째 메시지는 원래 사용자의 요청입니다. 모든 노드에서 동일하게 유지합니다.
  const translated = [
    reflectionPrompt,
    state.messages[0],
    ...state.messages
      .slice(1)
      .map((msg) => new clsMap[msg._getType()](msg.content)),
  ];
  const answer = await model.invoke(translated);
  // 이 출력 결과를 생성기(generator)에 대한 사용자 피드백으로 취급합니다.
    return {'messages': [HumanMessage(content=answer.content)]}
  return { messages: [new HumanMessage({ content: answer.content })] };
}

function shouldContinue(state) {
  if (state.messages.length > 6) {
    // 3회 반복 후, 각 반복마다 2개의 메시지가 쌓이면 종료합니다.
    return END;
  } else {
    return 'reflect';
  }
}

const builder = new StateGraph(annotation)
  .addNode('generate', generate)
  .addNode('reflect', reflect)
  .addEdge(START, 'generate')
  .addConditionalEdges('generate', shouldContinue)
  .addEdge('reflect', 'generate');

const graph = builder.compile();

// 예시
const initialState = {
  messages: [
    new HumanMessage(
      '오늘날 \'어린 왕자\'가 왜 중요한지에 대해 에세이를 작성하세요.'
    ),
  ],
};

for await (const output of await graph.stream(initialState)) {
  const messageType = output.generate ? 'generate' : 'reflect';
  console.log(
    '\nNew message:',
    output[messageType].messages[
      output[messageType].messages.length - 1
    ].content.slice(0, 100),
    '...'
  );
}
