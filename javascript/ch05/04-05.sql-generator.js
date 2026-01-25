import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
  END,
} from '@langchain/langgraph';

// SQL 쿼리 생성용
const modelLowTemp = new ChatOpenAI({ model:'gpt-4o-mini', temperature: 0.1 });
//  자연어 출력 생성용
const modelHighTemp = new ChatOpenAI({ model:'gpt-4o-mini', temperature: 0.7 });

const annotation = Annotation.Root({
  messages: Annotation({ reducer: messagesStateReducer, default: () => [] }),
  user_query: Annotation(),
  sql_query: Annotation(),
  sql_explanation: Annotation(),
});

const generatePrompt = new SystemMessage(
  '당신은 친절한 데이터 분석가입니다. 사용자의 질문을 바탕으로 SQL 쿼리를 작성하세요.',
);

async function generateSql(state) {
  const userMessage = new HumanMessage(state.user_query);
  const messages = [generatePrompt, ...state.messages, userMessage];
  const res = await modelLowTemp.invoke(messages);
  return {
    sql_query: res.content,
    // 대화 기록 업데이트
    messages: [userMessage, res],
  };
}

const explainPrompt = new SystemMessage(
  '당신은 친절한 데이터 분석가입니다. 사용자에게 SQL 쿼리를 간단하게 설명하세요.',
);

async function explainSql(state) {
  const messages = [explainPrompt, ...state.messages];
  const res = await modelHighTemp.invoke(messages);
  return {
    sql_explanation: res.content,
    // 대화 기록 업데이트
    messages: res,
  };
}

const builder = new StateGraph(annotation)
  .addNode('generate_sql', generateSql)
  .addNode('explain_sql', explainSql)
  .addEdge(START, 'generate_sql')
  .addEdge('generate_sql', 'explain_sql')
  .addEdge('explain_sql', END);

const graph = builder.compile();

// 예시
const result = await graph.invoke({
  user_query: '각 품목의 판매량을 구해주세요',
});
console.log(result);
