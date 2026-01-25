import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
  END,
} from '@langchain/langgraph';

const embeddings = new OpenAIEmbeddings();
// SQL 쿼리 생성용
const modelLowTemp = new ChatOpenAI({ model:'gpt-4o-mini', temperature: 0.1 });
// 자연어 출력 생성용
const modelHighTemp = new ChatOpenAI({ model:'gpt-4o-mini', temperature: 0.7 });

const annotation = Annotation.Root({
  messages: Annotation({ reducer: messagesStateReducer, default: () => [] }),
  user_query: Annotation(),
  domain: Annotation(),
  documents: Annotation(),
  answer: Annotation(),
});

// 테스트를 위한 샘플 문서
const sampleDocs = [
  { pageContent: `# 환자 의료 기록
- 이름: 홍길동
- 성별: 남
- 나이: 30세
## 진료 내역
- 2021년 1월 13일: 감기로 인한 발열로 병원 방문
- 2022년 3월 15일: 비염 진단으로 약 처방
- 2022년 5월 20일: 피부과 진료 및 약 처방
- 2022년 6월 10일: 발열 및 기침으로 코로나 검사 및 음성 판정
- 2022년 7월 2일: 코로나 진단 검사 및 양성 판정`,
    metadata: { domain: 'records' } 
  },
  {
    pageContent: `# 보험 FAQ
- Q: 과거에 병력이 있는데 가입가능한가요?
- A: 치료기간, 현재 상태, 후유증 여부, 연령 등에 따라 다르므로 가입 가능 여부는 가까운 대리점에서 상담 받으세요.

- Q: 보험금 청구 방법은 어떻게 되나요??
- A: 보험금 청구는 보험금 청구서 작성 후 가까운 대리점을 통해 진행 가능합니다.

- Q: 코로나 19도 보험 적용이 되나요?
- A: 코로나 19는 보험 대상에 포함되지 않습니다.

- Q: 보험금 지급이 거부되는 경우는 어떤 경우인가요?
- A: 보험금 지급 거부 사유는 보험 계약서에 명시되어 있습니다. 자세한 사항은 계약서를 참조하세요
    `,
    metadata: { domain: 'insurance' },
  },
];

// 벡터 저장소 초기화
const medicalRecordsStore = await MemoryVectorStore.fromDocuments(
  sampleDocs,
  embeddings,
);
const medicalRecordsRetriever = medicalRecordsStore.asRetriever();

const insuranceFaqsStore = await MemoryVectorStore.fromDocuments(
  sampleDocs,
  embeddings,
);
const insuranceFaqsRetriever = insuranceFaqsStore.asRetriever();

const routerPrompt = new SystemMessage(
  `사용자 문의를 어느 도메인으로 라우팅할지 결정하세요. 선택할 수 있는 두 가지 도메인은 다음과 같습니다.
- records: 진단, 치료, 처방과 같은 환자의 의료 기록을 포함합니다.  
- insurance: 보험 정책, 청구, 보장에 대한 자주 묻는 질문을 포함합니다.  

도메인 이름만 출력하세요.`,
);

async function routerNode(state) {
  const userMessage = new HumanMessage(state.user_query);
  const messages = [routerPrompt, ...state.messages, userMessage];
  const res = await modelLowTemp.invoke(messages);
  return {
    domain: res.content,
    // 대화 기록 업데이트
    messages: [userMessage, res],
  };
}

function pickRetriever(state) {
  if (state.domain === 'records') {
    return 'retrieve_medical_records';
  } else {
    return 'retrieve_insurance_faqs';
  }
}

async function retrieveMedicalRecords(state) {
  const documents = await medicalRecordsRetriever.invoke(state.user_query);
  return {
    documents,
  };
}

async function retrieveInsuranceFaqs(state) {
  const documents = await insuranceFaqsRetriever.invoke(state.user_query);
  return {
    documents,
  };
}

const medicalRecordsPrompt = new SystemMessage(
  '당신은 유능한 의료 챗봇입니다. 진단, 치료, 처방과 같은 환자의 의료 기록을 기반으로 질문에 답하세요.',
);

const insuranceFaqsPrompt = new SystemMessage(
  '당신은 유능한 의료 보험 챗봇입니다. 보험 정책, 청구 및 보장에 대한 자주 묻는 질문에 답하세요.',
);

async function generateAnswer(state) {
  const prompt =
    state.domain === 'records' ? medicalRecordsPrompt : insuranceFaqsPrompt;
  const messages = [
    prompt,
    ...state.messages,
    new HumanMessage(`Documents: ${state.documents}`),
  ];
  const res = await modelHighTemp.invoke(messages);
  return {
    answer: res.content,
    // 대화 기록 업데이트
    messages: res,
  };
}

const builder = new StateGraph(annotation)
  .addNode('router', routerNode)
  .addNode('retrieve_medical_records', retrieveMedicalRecords)
  .addNode('retrieve_insurance_faqs', retrieveInsuranceFaqs)
  .addNode('generate_answer', generateAnswer)
  .addEdge(START, 'router')
  .addConditionalEdges('router', pickRetriever)
  .addEdge('retrieve_medical_records', 'generate_answer')
  .addEdge('retrieve_insurance_faqs', 'generate_answer')
  .addEdge('generate_answer', END);

const graph = builder.compile();

// 예시
const input = {
  user_query: '코로나 19도 보험 적용이 되나요?',
};

for await (const chunk of await graph.stream(input)) {
  console.log(chunk);
}
