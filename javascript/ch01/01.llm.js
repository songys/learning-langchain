import { OpenAI } from '@langchain/openai';

const model = new OpenAI({ model: 'gpt-3.5-turbo-instruct' });

const response = await model.invoke('하늘이');
console.log(response);
