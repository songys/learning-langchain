import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  mergeMessageRuns,
} from '@langchain/core/messages';

const messages = [
  new SystemMessage('당신은 친절한 어시스턴트입니다.'),
  new SystemMessage('항상 농담으로 대답하세요.'),
  new HumanMessage({
    content: [{ type: 'text', text: '어떤 피자가 제일 맛있나요?' }],
  }),
  new HumanMessage('어떤 햄버거가 가장 맛있나요?'),
  new AIMessage(
    '나는 항상 너만 "고르곤졸라"'
  ),
  new AIMessage(
    '너가 "버거" 싶어'
  ),
];

// 연속된 메시지를 병합
const mergedMessages = mergeMessageRuns(messages);
console.log(mergedMessages);

// 선언형 구성
const model = new ChatOpenAI()
const merger = mergeMessageRuns()
const chain = merger.pipe(model)

