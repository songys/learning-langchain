import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

const answerSchema = z
  .object({
    answer: z.string().describe('사용자의 질문에 대한 답변'),
    justification: z.string().describe('답변에 대한 근거'),
  })
  .describe(
    '사용자의 질문에 대한 답변과 그에 대한 근거(justification)를 함께 제공하세요.'
  );

const model = new ChatOpenAI({
  model: 'gpt-4o-mini',
  temperature: 0,
}).withStructuredOutput(answerSchema);

const response = await model.invoke(
  '1 킬로그램의 벽돌과 1 킬로그램의 깃털 중 어느 쪽이 더 무겁나요?'
);
console.log(response);
