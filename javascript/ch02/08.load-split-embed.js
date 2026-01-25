import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';

const loader = new TextLoader('./test.txt');
const docs = await loader.load();

// 문서 분할
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const chunks = await splitter.splitDocuments(docs);

console.log(chunks);

// 임베딩 생성
const model = new OpenAIEmbeddings();
const embeddings = await model.embedDocuments(chunks.map((c) => c.pageContent));

console.log(embeddings);
