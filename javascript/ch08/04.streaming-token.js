import { DuckDuckGoSearch } from '@langchain/community/tools/duckduckgo_search';
import { Calculator } from '@langchain/community/tools/calculator';
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
} from '@langchain/langgraph';
import { ToolNode, toolsCondition } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';
import { MemorySaver } from '@langchain/langgraph';

const search = new DuckDuckGoSearch();
const calculator = new Calculator();
const tools = [search, calculator];
const model = new ChatOpenAI({
  model: 'gpt-4o-mini',
  temperature: 0.1,
}).bindTools(tools);

const annotation = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => [],
  }),
});

async function modelNode(state) {
  const res = await model.invoke(state.messages);
  return { messages: res };
}

const builder = new StateGraph(annotation)
  .addNode('model', modelNode)
  .addNode('tools', new ToolNode(tools))
  .addEdge(START, 'model')
  .addConditionalEdges('model', toolsCondition)
  .addEdge('tools', 'model');


const controller = new AbortController();

const input = {
  messages: [
    new HumanMessage(
      '미국의 제30대 대통령이 사망했을 때 몇 살이었나요?',
    ),
  ],
};

const config = { configurable: { thread_id: '1' } };
const graph = builder.compile({ checkpointer: new MemorySaver() });

try {
  const output = await graph.streamEvents(input, {...config, version: "v2"});

  for await (const { event, data } of output) {
    if (event === "on_chat_model_stream") {
      const msg = data.chunk;
      if (msg.content) {
        console.log(msg.content);
      }
    }
  }
  
} catch (e) {
  console.log(e);
}
