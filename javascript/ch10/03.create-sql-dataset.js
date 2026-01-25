import { Client } from 'langsmith';
const client = new Client();

const exampleInputs = [
  [
    "어느 나라의 고객이 가장 많이 지출했나요? 그리고 얼마를 지출했나요?",
    "가장 많이 지출한 나라는 미국으로, 총 지출액은 $523.06입니다",
  ],
  [
    "2013년에 가장 많이 판매된된 트랙은 무엇인가요?", 
    "2013년에 가장 많이 판매된된 트랙은 Hot Girl입니다.",
  ],
  [
    "Led Zeppelin 아티스트는 몇 개의 앨범을 발매했나요?",
    "Led Zeppelin은 14개의 앨범을 발매했습니다",
  ],
  [
    "'Big Ones' 앨범의 총 가격은 얼마인가요?",
    "'Big Ones' 앨범의 총 가격은 14.85입니다",
  ],
  [
    "2009년에 어떤 영업 담당자가 가장 많은 매출을 올렸나요?",
    "Steve Johnson이 2009년에 가장 많은 매출을 올렸습니다",
  ],
];

const datasetName = 'sql-agent-response';

if (!(await client.hasDataset({ datasetName }))) {
  client.createDataset(datasetName);

  // 일괄 생성을 위한 입력, 출력 및 메타데이터 준비
  const inputs = exampleInputs.map(([inputPrompt]) => ({
    question: inputPrompt,
  }));

  const outputs = exampleInputs.map(([, outputAnswer]) => ({
    answer: outputAnswer,
  }));

  await client.createExamples({
    inputs,
    outputs,
    datasetId: dataset.id,
  });
}
