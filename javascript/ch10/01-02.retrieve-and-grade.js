import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { z } from 'zod';
import { ChatOpenAI } from '@langchain/openai';

const urls = [
  'https://blog.langchain.dev/top-5-langgraph-agents-in-production-2024/',
  'https://blog.langchain.dev/langchain-state-of-ai-2024/',
  'https://blog.langchain.dev/introducing-ambient-agents/',
];

// URL에서 문서 로드
const loadDocs = async (urls) => {
  const docs = [];
  for (const url of urls) {
    const loader = new CheerioWebBaseLoader(url);
    const loadedDocs = await loader.load();
    docs.push(...loadedDocs);
  }
  return docs;
};

async function main() {
  const docsList = await loadDocs(urls);

  // 텍스트 분할기 초기화
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 0,
  });

  // 문서를 작은 청크로 분할
  const docSplits = await textSplitter.splitDocuments(docsList);

  // 벡터 데이터베이스에 추가
  const vectorstore = await MemoryVectorStore.fromDocuments(
    docSplits,
    new OpenAIEmbeddings()
  );

  const retriever = vectorstore.asRetriever(); // `retriever` 객체를 이제 쿼리에 사용할 수 있습니다

  const question = '2024년에 프로덕션에서 사용된 LangGraph 에이전트 2개는 무엇인가?';

  const docs = await retriever.invoke(question);

  console.log('검색된 문서: \n', docs[0].page_content);

  // Zod를 사용하여 스키마 정의
  const GradeDocumentsSchema = z.object({
    binary_score: z
      .string()
      .describe("문서가 질문과 관련이 있으면 'yes', 없으면 'no'"),
  });

  // Zod 스키마를 사용하여 구조화된 출력이 있는 LLM 초기화
  const llm = new ChatOpenAI({ model: 'gpt-4o-mini', temperature: 0 });
  const structuredLLMGrader = llm.withStructuredOutput(GradeDocumentsSchema);

  // 시스템 메시지와 프롬프트 템플릿
  const systemMessage = `당신은 사용자 질문에 대한 검색된 문서의 관련성을 평가하는 채점자입니다. 문서에 질문과 관련된 키워드나 의미가 포함되어 있다면 관련성이 있다고 평가하세요. 문서가 질문과 관련이 있는지 여부를 나타내는 'yes' 또는 'no'로 평가해 주세요.`;
  const gradePrompt = ChatPromptTemplate.fromMessages([
    { role: 'system', content: systemMessage },
    {
      role: 'human',
      content:
        '검색된 문서: \n\n {document} \n\n 사용자 질문: {question}',
    },
  ]);

  // 프롬프트와 구조화된 출력 결합
  const retrievalGrader = gradePrompt.pipe(structuredLLMGrader);

  // 검색된 문서 평가
  const results = await retrievalGrader.invoke({
    question,
    document: docs[0].page_content,
  });

  console.log('\n\n평가 결과: \n', results);
}

// 메인 함수 실행
main()