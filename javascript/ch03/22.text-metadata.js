import { ChatOpenAI } from '@langchain/openai';
import { SelfQueryRetriever } from 'langchain/retrievers/self_query';
import { FunctionalTranslator } from '@langchain/core/structured_query';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document } from 'langchain/document';
import { AttributeInfo } from 'langchain/chains/query_constructor';
import { OpenAIEmbeddings } from '@langchain/openai';

const docs = [
  new Document({
    pageContent:
      '과학자들이 공룡을 되살리고 대혼란이 일어난다.',
    metadata: {
      year: 1993,
      rating: 7.7,
      genre: 'SF',
      length: 122,
    },
  }),
  new Document({
    pageContent:
      '레오나르도 디카프리오가 꿈속의 꿈속의 꿈속의 꿈속에 빠진다.',
    metadata: {
      year: 2010,
      director: '크리스토퍼 놀란',
      rating: 8.2,
      length: 148,
    },
  }),
  new Document({
    pageContent:
      '심리학자인 형사가 꿈속의 꿈속의 꿈속의 꿈속의 꿈속에 빠진다. 인셉션이 이 발상을 차용했다.',
    metadata: { year: 2006, director: '곤 사토시', rating: 8.6 },
  }),
  new Document({
    pageContent:
      '평범한 체형의 매우 건강하고 순수한 매력을 지닌 여성들을 남성들이 동경한다.',
    metadata: {
      year: 2019,
      director: '그레타 거윅',
      rating: 8.3,
      length: 135,
    },
  }),
  new Document({
    pageContent: '장난감들이 살아 움직이며 신나는 시간을 보낸다',
    metadata: { year: 1995, genre: '애니메이션', length: 77 },
  }),
  new Document({
    pageContent: '세 남자가 구역으로 들어가고, 세 남자가 구역 밖으로 나온다.',
    metadata: {
      year: 1979,
      director: '안드레이 타르코프스키',
      genre: '스릴러',
      rating: 9.9,
    },
  }),
];

const llm = new ChatOpenAI({ modelName: 'gpt-4o-mini', temperature: 0 });

const embeddings = new OpenAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);


const fields = [
  {
    name: 'genre',
    description: '영화 장르',
    type: 'string or array of strings',
  },
  {
    name: 'year',
    description: '영화 개봉 연도',
    type: 'number',
  },
  {
    name: 'director',
    description: '영화 감독',
    type: 'string',
  },
  {
    name: 'rating',
    description: '영화 평점 1-10점',
    type: 'number',
  },
  {
    name: 'length',
    description: '영화 상영 시간',
    type: 'number',
  },
];

const attributeInfos = fields.map(
  (field) => new AttributeInfo(field.name, field.description, field.type)
);
const description = '영화에 대한 간략한 정보';
const selfQueryRetriever = SelfQueryRetriever.fromLLM({
  llm,
  vectorStore,
  description,
  attributeInfo: attributeInfos,

  structuredQueryTranslator: new FunctionalTranslator(),
});

const result = await selfQueryRetriever.invoke(
  '평점이 높은(8.5점 이상) SF영화는 무엇인가요?'
);
console.log(result);
