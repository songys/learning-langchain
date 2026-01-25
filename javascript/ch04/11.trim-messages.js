import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  trimMessages,
} from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

const messages = [
  new SystemMessage('당신은 친절한 어시스턴트입니다.'),
  new HumanMessage('안녕하세요! 나는 민혁입니다.'),
  new AIMessage('안녕하세요!'),
  new HumanMessage('바닐라 아이스크림을 좋아해요.'),
  new AIMessage('좋네요!'),
  new HumanMessage('2 + 2는 얼마죠?'),
  new AIMessage('4입니다.'),
  new HumanMessage('고마워요.'),
  new AIMessage('천만에요!'),
  new HumanMessage('즐거운가요?'),
  new AIMessage('예!'),
];

const trimmer = trimMessages({
  maxTokens: 65,
  strategy: 'last',
  tokenCounter: new ChatOpenAI({ modelName: 'gpt-4o' }),
  includeSystem: true,
  allowPartial: false,
  startOn: 'human',
});

const trimmed = await trimmer.invoke(messages);
console.log(trimmed);
