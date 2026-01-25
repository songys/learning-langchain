import {
    SupabaseVectorStore
  } from "@langchain/community/vectorstores/supabase";
  import { OpenAIEmbeddings } from "@langchain/openai";
  
  import { Document } from "@langchain/core/documents";
  import { createClient } from "@supabase/supabase-js";
  import dotenv from 'dotenv';
  dotenv.config();

  const embeddings = new OpenAIEmbeddings();
  
  const supabaseClient = createClient(
    process.env.SUPABASE_URL as string,
    process.env.SUPABASE_SERVICE_ROLE_KEY as string
  );
  const vectorStore = new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: "documents",
    queryName: "match_documents",
  });
  
  // 문서 예시
  
  const document1: Document = {
    pageContent: "The powerhouse of the cell is the mitochondria",
    metadata: { source: "https://example.com" },
  };
  
  const document2: Document = {
    pageContent: "Buildings are made out of brick", 
    metadata: { source: "https://example.com" },
  };
  
  const documents = [document1, document2];
  
  // 데이터베이스에 데이터 저장
  
  await vectorStore.addDocuments(documents, { ids: ["1", "2"] });
  
  // 벡터 저장소에 쿼리 전송
  
  const filter = { source: "https://example.com" };
  
  const similaritySearchResults = await vectorStore.similaritySearch(
    "biology",
    2,
    filter
  );
  
  for (const doc of similaritySearchResults) {
    console.log(`* ${doc.pageContent} [${JSON.stringify(doc.metadata, null)}]`);
  }