import { z } from 'zod';
import { ChatOpenAI } from '@langchain/openai';

const joke = z.object({
  setup: z.string().describe('농담의 설정'),
  punchline: z.string().describe('농담의 포인트'),
});

let model = new ChatOpenAI({
  model: 'gpt-4o-mini',
  temperature: 0,
});

model = model.withStructuredOutput(joke);

const result = await model.invoke('고양이에 대한 농담을 만들어 주세요.');
console.log(result);
