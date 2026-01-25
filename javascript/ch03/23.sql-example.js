/*
이번 예시에서는 sqlite를 통해 Chinook_Sqlite.sql를 사용한다. 실습 실행 전에 sqlite와 sql 파일을 다운받아야 한다.

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

`Chinook.db`를 코드를 실행할 디렉터리에 옮긴다.
*/

import { ChatOpenAI } from '@langchain/openai';
import { createSqlQueryChain } from 'langchain/chains/sql_db';
import { SqlDatabase } from 'langchain/sql_db';
import { DataSource } from 'typeorm';
import { QuerySqlTool } from 'langchain/tools/sql';

const datasource = new DataSource({
  type: 'sqlite',
  database: 'Chinook.db',
});
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});

const llm = new ChatOpenAI({ modelName: 'gpt-4o', temperature: 0 });
// 질문을 SQL 쿼리로 변환
const writeQuery = await createSqlQueryChain({ llm, db, dialect: 'sqlite' });
// SQL 쿼리 실행
const executeQuery = new QuerySqlTool(db);
// 체인 구성
const chain = writeQuery.pipe(executeQuery);

const result = await chain.invoke({
  question: '직원(employee)은 모두 몇 명인가요?',
});
console.log(result);
