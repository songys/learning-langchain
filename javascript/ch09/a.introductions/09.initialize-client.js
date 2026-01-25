import { Client } from "@langchain/langgraph-sdk";

// langgraph up 호출 시 기본 포트를 변경한 경우에만 url 인자를 get_client()에 전달
const client = new Client({apiUrl:"http://127.0.0.1:2024"});
// "chatbot_graph"라는 이름으로 배포된 그래프 사용
const assistantId = "chatbot_graph";
const thread = await client.threads.create();

const input = {
  messages: [{ "role": "user", "content": "미국의 제30대 대통령이 사망했을 때 몇 살이었나요?"}]
}

const streamResponse = client.runs.stream(
  thread["thread_id"],
  assistantId,
  {
    input: input,
    streamMode: "updates",
  }
);
for await (const chunk of streamResponse) {
  console.log(`Receiving new event of type: ${chunk.event}...`);
  console.log(chunk.data);
  console.log("\n\n");
}
