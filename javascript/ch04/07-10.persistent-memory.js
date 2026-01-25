import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
  END,
} from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';

const State = {
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
builder = builder.addEdge(START, 'chatbot');
builder = builder.addEdge('chatbot', END);

// 영속성 추가
const graph = builder.compile({ checkpointer: new MemorySaver() });

// 스레드 설정
const thread1 = { configurable: { thread_id: '1' } };

// 영속성 추가 후 그래프 실행
const result_1 = await graph.invoke(
  {
    messages: [new HumanMessage('안녕하세요, 저는 민혁입니다!')],
  },
  thread1,
);

const result_2 = await graph.invoke(
  {
    messages: [new HumanMessage('제 이름이 뭐죠?')],
  },
  thread1,
);

// 상태 확인
await graph.getState(thread1);

// 상태 업데이트
await graph.updateState(thread1, {messages: [new HumanMessage('저는 LLM이 좋아요!')]});
