# 9장 AI 애플리케이션 배포

러닝 랭체인 9장의 RAG AI 챗봇 에이전트 배포 실습에 필요한 코드를 담은 디렉터리입니다. 이번 실습에서 제공하는 에이전트는 인제스천 과정을 통해 문서를 이해하고, 이해한 내용을 바탕으로 질문에 답합니다.

## 구성

이번 실습에서 배포할 랭그래프 애플리케이션은 두 가지 주요 요소 구성되어 있습니다.

1.  **인제스천 그래프:** 문서를 로드하고, 임베딩하고, 인덱싱합니다.
2.  **검색 그래프:** 인덱싱된 문서를 기반으로 질문에 답합니다.

**참고:** 이번 예시의 완전한 AI 애플리케이션(프런트엔드 및 백엔드)은 [여기](https://github.com/mayooear/ai-pdf-chatbot-langchain/tree/main)에서 확인하세요.

## 준비 사항

코드를 실행하기 전에 다음 사항을 확인하세요.

1.  **learning-langchain 저장소 루트에 환경 변수 설정:** learning-langchain 저장소의 루트에 `.env` 파일에 필요한 환경 변수를 설정하세요. 필요한 변수 목록은 `.env.example`을 참조하세요.

2.  **슈파베이스 계정 및 Supabase API 키:**
    *   [supabase.com](https://supabase.com)에 가입하세요.
    *   계정이 생성되면 새 프로젝트를 만든 다음 [Project Settings] 섹션으로 이동하세요.
    *   [Data API] 섹션으로 이동해 키를 확인하세요.
    *   프로젝트 URL과 `service_role` 키를 복사하여 `.env` 파일에 `SUPABASE_URL` 및 `SUPABASE_SERVICE_ROLE_KEY` 값으로 추가하세요.

3.  **도커:** 자바스크립트 구현에서 Chroma 벡터 데이터베이스를 실행하려면 도커가 필요합니다. 도커 설치 및 설정은 [README](../../README.md#docker-setup-and-usage)의 지침을 따르세요. 도커 설정이 완료되면 다음 명령어를 실행하여 Chroma 서버를 시작하세요.

```bash 
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma

```

8000번 포트에서 Chroma 서버가 시작합니다.

## 저장소 구성

저장소는 다음과 같이 구성됩니다.

```
ch09 # 자바스크립트 구현
├── src # 소스 코드
│ ├── ingestion_graph # 인제스천 그래프 컴포넌트
│ ├── retrieval_graph # 검색 그래프 컴포넌트 
│ ├── shared # 공유 컴포넌트
│ ├── configuration.ts # 설정 파일
│ ├── graph.ts # 그래프 정의 파일
│ ├── state.ts # 상태 정의 파일
│ └── utils.ts # 유틸리티 함수
├── demo.ts # 데모 스크립트
├── langgraph.json # 랭그래프 설정 파일
├── package.json # 자바스크립트 의존성
└── tsconfig.json # 타입스크립트 설정
```

## 환경 설정

1.  **가상 환경 설정:**

    ```bash
    cd javascript/ch09
    npm install
    ```

## 애플리케이션 실행

### JavaScript

1.  **데모 스크립트 실행**

    ```bash
    node demo.ts
    ```

## 로컬 개발 서버

현재 위치에서 로컬 개발 서버를 실행할 수 있습니다.

```bash
# Using npm script
npm run langgraph:dev
```

2024 포트에서 로컬 개발 서버가 시작되고 디버깅 및 추적을 위한 langsmith UI가 연결됩니다.

## 애플리케이션 배포

LangGraph 에이전트를 클라우드 서비스에 배포하려면 이 [가이드](https://langchain-ai.github.io/langgraph/cloud/quick_start/?h=studio#deploy-to-langgraph-cloud)를 따라 LangGraph의 클라우드를 사용하거나 이 [가이드](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/)를 따라 직접 호스팅할 수 있습니다.

### 랭그래프 CLI 사용

1.  **랭그래프 설정:**

    *   `langgraph.json` 파일이 그래프의 진입점을 올바르게 가리키도록 구성되어 있는지 확인하세요.

2.  **애플리케이션 배포:**

    *   저장소의 루트에서 다음 명령어를 실행

    ```bash
    npx @langchain/langgraph-cli deploy -c javascript/ch09/langgraph.json
    ```

    ```

    *   Follow the prompts to deploy your application.

*   프롬프트를 따라 배포 과정을 완료하세요.

## 배포된 애플리케이션과 상호작용하기

배포가 완료되면 LangGraph SDK를 사용하여 애플리케이션과 상호작용할 수 있습니다. `demo.ts` 파일에서 스레드를 생성하고 배포된 그래프를 호출하는 방법에 대한 예제를 확인할 수 있습니다.

