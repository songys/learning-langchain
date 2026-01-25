import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  filterMessages,
} from '@langchain/core/messages';

const messages = [
  new SystemMessage({ content: '당신은 친절한 어시스턴트입니다', id: '1' }),
  new HumanMessage({ content: '예시 입력', id: '2', name: 'example_user' }),
  new AIMessage({ content: '예시 출력', id: '3', name: 'example_assistant'}),
  new HumanMessage({ content: '실제 입력', id: '4', name: 'bob' }),
  new AIMessage({ content: '실제 출력', id: '5', name: 'alice' }),
];

// 사용자 메시지만 필터링
const filterByHumanMessages = filterMessages(messages, {
  includeTypes: ['human'],
});
console.log(`사용자 메시지: ${JSON.stringify(filterByHumanMessages)}`);

// 특정 이름의 메시지 제외
const filterByExcludedNames = filterMessages(messages, {
  excludeNames: ['example_user', 'example_assistant'],
});
console.log(
  `\nn이름에 example이 포함되지 않은 메시지: ${JSON.stringify(filterByExcludedNames)}`
);

// 유형과 ID로 필터링
const filterByTypesAndIDs = filterMessages(messages, {
  includeTypes: ['human', 'ai'],
  excludeIds: ['3'],
});
console.log(
  `\n특정 유형과 ID로 필터링한 메시지: ${JSON.stringify(filterByTypesAndIDs)}`
);

// 선언형 구성
const model = new ChatOpenAI()
const filter = filterMessages({
  excludeNames: ["example_user", "example_assistant"]
})
const chain = filter.pipe(model)
