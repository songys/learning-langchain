import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
  END,
} from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';
import * as fs from 'fs';

const State = {
  /**
   * State는 세 가지를 정의한다.
   * 1. 그래프 상태의 구조 (어떤 "채널"이 읽기/쓰기가 가능한지)
   * 2. 상태 채널의 기본값
   * 3. 상태 채널의 리듀서. 리듀서는 상태 업데이트 방법을 표현하는 함수를 말한다.
   *    아래에서는 새 메시지를 메시지 배열에 추가한다.
   */

  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => [],
  }),
};

let builder = new StateGraph(State);

const model = new ChatOpenAI({model: 'gpt-4o-mini'});

async function chatbot(state) {
  const answer = await model.invoke(state.messages);
  return { messages: answer };
}

builder = builder.addNode('chatbot', chatbot);

builder = builder.addEdge(START, 'chatbot').addEdge('chatbot', END);

let graph = builder.compile();

// 그래프 이미지 저장
const image = await graph.getGraph().drawMermaidPng();
const arrayBuffer = await image.arrayBuffer();
const buffer = new Uint8Array(arrayBuffer);

fs.writeFileSync('LLM_call_architecture.png', buffer);

// 그래프 실행
const input = { messages: [new HumanMessage('안녕하세요!')] };
for await (const chunk of await graph.stream(input)) {
  console.log(chunk);
}
