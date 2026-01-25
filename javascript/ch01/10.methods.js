import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI();

const response = await model.invoke('반가워요!');
console.log(response);
// Hi!

const completions = await model.batch(['반가워요!', '잘 있어요!']);
// ['Hi!', 'See you!']

for await (const token of await model.stream('잘 있어요!')) {
  console.log(token);
  // Good
  // bye
  // !
}
