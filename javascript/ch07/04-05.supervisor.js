import { ChatOpenAI } from "@langchain/openai";
import {
  StateGraph,
  Annotation,
  MessagesAnnotation,
  START,
  END,
} from "@langchain/langgraph";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";


const SupervisorDecision = z.object({
  next: z.enum(["researcher", "coder", "FINISH"]),
});

// 모델 초기화
const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
const modelWithStructuredOutput =
  model.withStructuredOutput(SupervisorDecision);

// 사용 가능한 에이전트
const agents = ["researcher", "coder"];

// 시스템 프롬프트 정의
const systemPromptPart1 = `당신은 다음 서브에이전트 사이의 대화를 관리하는 슈퍼바이저입니다. 서브에이전트: ${agents.join(
  ", ",
)}. 다음으로 행동할 서브에이전트를 지목하세요. 각 서브에이전트는 임무를 수행하고 결과와 상태를 응답합니다. 실행할 서브에이전트가 없거나 작업이 완료되면, FINISH로 응답하세요.`;

const systemPromptPart2 = `위 대화를 바탕으로, 다음으로 행동할 서브에이전트는 누구입니까? 아니면 FINISH 해야 합니까? 서브에이전트: ${agents.join(
  ", ",
)}, FINISH`;

// 슈퍼바이저 정의
const supervisor = async (state) => {
  const messages = [
    new SystemMessage(systemPromptPart1),
    ...state.messages,
    new SystemMessage(systemPromptPart2),
  ];

  const result = await modelWithStructuredOutput.invoke(messages);
  return {
    messages: state.messages,
    next: result.next,
  };
};

// 에이전트 상태 정의
const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  next: Annotation("researcher" | "coder" | "FINISH"),
});

// 에이전트 함수 정의
const researcher = async (state) => {
  // 실제 구현에서는 이 함수가 리서치 작업을 수행합니다.
  // 여기서는 임의로 관련 데이터를 찾는 척 합니다.
  const response = {
    role: 'assistant',
    content: '관련 데이터를 찾는 중입니다... 잠시만 기다려주세요.'
  };

  // 임의의 데이터 생성
  const fakeData = {
    data: '전세계 인구 데이터: [미국: 331M, 중국: 1.4B, 인도: 1.3B]'
  };

  response.content += `\n찾은 데이터: ${fakeData.data}`;

  return {
    messages: [...state.messages, response],
  };
};

const coder = async (state) => {
  // 실제 구현에서는 이 함수가 코드를 작성합니다.
  // 여기서는 임의로 코드를 작성하는 척 합니다.
  const response = {
    role: 'assistant', 
    content: '코드를 작성 중입니다... 잠시만 기다려주세요.'
  };

  // 임의의 코드 생성
  const fakeCode = `
def visualize_population(data):
    import matplotlib.pyplot as plt

    countries = list(data.keys())
    population = list(data.values())

    plt.bar(countries, population)
    plt.xlabel('Country')
    plt.ylabel('Population')
    plt.title('World Population by Country')
    plt.show()

data = {'USA': 331, 'China': 1400, 'India': 1300}
visualize_population(data)
`;

  response.content += `\n작성된 코드:\n${fakeCode}`;

  return {
    messages: [...state.messages, response],
  };
};

// 그래프 구축
const graph = new StateGraph(StateAnnotation)
  .addNode("supervisor", supervisor)
  .addNode("researcher", researcher)
  .addNode("coder", coder)
  .addEdge(START, "supervisor")
  // 슈퍼바이저의 결정에 따라 에이전트 중 하나로 라우팅하거나 종료합니다.
  .addConditionalEdges("supervisor", async (state) =>
    state.next === "FINISH" ? END : state.next,
  )
  .addEdge("researcher", "supervisor")
  .addEdge("coder", "supervisor")
  .compile();

// 예시

const initialState = {
  messages: [
    new HumanMessage(
      "전세계 인구를 국적을 기준으로 시각화 해주세요."
    ),
  ],
  next: "supervisor",
};

let graphStream = graph.stream(initialState);

for await (const output of await graphStream) {
  const currentOutput = output.supervisor || output;
  
  if (currentOutput.messages) {
    console.log("\nMessages:");
    // 마지막 메시지만 출력
    if (currentOutput.messages.length > 0) {
      const lastMsg = currentOutput.messages[currentOutput.messages.length - 1];
      try {
        // 메시지 타입 확인
        let role = 'unknown';
        if (lastMsg.type === 'constructor') {
          role = lastMsg.id[2]; // langchain_core/messages/HumanMessage에서 HumanMessage 추출
        } else {
          role = lastMsg.type || lastMsg._getType?.() || lastMsg.role;
        }
        console.log(`Content: ${lastMsg.kwargs?.content || lastMsg.content}`);
      } catch (e) {
        
      }
    }
  }
  
  if (currentOutput.next) {
    console.log(`\nNext Step: ${currentOutput.next}`);
  }
}
