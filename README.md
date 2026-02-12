# 러닝 랭체인
<img align="right" src="./cover.png" width="300px">

### 랭체인부터 랭그래프, RAG, AI 에이전트, 그리고 MCP까지<br>직접 만들며 익히는 생성 AI 애플리케이션 개발의 모든 것

* 지은이 : 메이오 오신, 누노 캄포스 
* 옮긴이 : 강민혁
* ISBN :  979-11-6921-378-3  93000
* 발행일 : 2025년 5월 14일
* 페이지수 : 400쪽
* 정가 : 28,000원
* ISBN: 979-11-6921-378-3
* [원서 실습 코드 저장소](https://github.com/langchain-ai/learning-langchain)


**구매 링크**
* [교보문고](https://product.kyobobook.co.kr/detail/S000216453776)
* [예스24](https://www.yes24.com/product/goods/146327472)
* [알라딘](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=363882755)


LLM 애플리케이션, 어디서부터 시작해야 할지 막막한가요?
챗GPT 이후의 시대, 검색 증강 생성(RAG), 멀티 에이전트, 랭그래프, MCP 같은 용어들이 쏟아지지만
정작 무엇부터 어떻게 시작해야 할지 고민이라면,
이 책이 지금 꼭 필요한 이유입니다.

이 책은 LLM 기반 애플리케이션 개발의 효율을 극대화하는 랭체인과, 복잡한 아키텍처 설계를 가능하게 하는 랭그래프를 중심으로, 기초 개념부터 실전 배포·운영까지 전 과정을 체계적으로 안내합니다. 

직접 구현하며 익히는 실습 중심의 구성으로, 다양한 예제로 AI 애플리케이션을 즉시 개발에 적용할 수 있는 실전 역량을 길러줍니다. 특히 부록에서는 AI 에이전트와 외부 시스템을 표준 방식으로 연결하는 최신 기술, MCP를 상세히 소개하여, 빠르게 진화하는 생성 AI 기술 흐름을 반영했습니다. 

LLM을 활용한 애플리케이션 개발이 처음인 독자에게는 출발점을, 이미 경험이 있는 개발자에게는 한 단계 더 나아갈 수 있는 실전 감각을 제공합니다. 빠르게 변화하는 생성 AI 환경 속에서, 이 책은 랭체인의 핵심 개념부터 실전 적용까지 단계별로 안내합니다. 최신 AI 기술을 활용한 개발 역량을 키워보세요.



## 설치 방법

이 저장소의 코드를 실행하기 위해서는 파이썬이나 자바스크립트 환경이 필요합니다.

**파이썬**

1.  필요한 경우 `.python-version` 파일에 명시된 Python 버전을 설치합니다. (예: pyenv 사용)
2.  가상 환경을 생성하고 활성화합니다. (권장)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\\Scripts\\activate  # Windows
    ```
3.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

**자바스크립트**

1.  Node.js 및 npm (또는 yarn)을 설치합니다.
2.  프로젝트 루트에서 의존성을 설치합니다.
    ```bash
    npm install
    # 또는 yarn install
    ```
3.  환경 변수 설정이 필요한 경우 `.env.example` 파일을 복사하여 `.env` 파일을 만들고 내용을 채웁니다.


## 사용법

각 실습 코드는 해당 디렉토리 내에서 실행할 수 있습니다.

**파이썬 예제**

```bash
export OPENAI_API_KEY=<오픈AI-API-키>
cd python/ch00  # ch00는 해당 챕터 번호
python 00.xxxxx.py
```

**자바스크립트 예제**

```bash
export OPENAI_API_KEY=<오픈AI-API-키>
cd javascript/ch00 # ch00는 해당 챕터 번호
node 00.xxxxx.js
```

자세한 내용은 도서와 소스 코드를 참고하세요.

---

## 번역서 파일에 Jupyter 노트북 (Colab 실행 가능) 파일을 추가했습니다.

각 챕터의 Python 코드와 **Jupyter 노트북(.ipynb)** 을 제공합니다.

### 주요 특징

| 특징 | 설명 |
|------|------|
| **설명** | 개념 설명 추가 |
| **Ollama 변환** | OpenAI API 대신 무료 로컬 LLM(Ollama)을 사용하도록 변경 |
| **Colab 호환** | Google Colab에서 바로 실행 가능한 환경 설정 셀 포함 |

### 코드 변경 예시 (OpenAI → Ollama)

```python
# 원본 (OpenAI API 필요)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
model = ChatOpenAI(model='gpt-4o-mini')
embeddings = OpenAIEmbeddings()

# 변경 (무료, 로컬 실행)
from langchain_ollama import ChatOllama, OllamaEmbeddings
model = ChatOllama(model='llama3.2')
embeddings = OllamaEmbeddings(model='nomic-embed-text')
```

### 사용 가능한 Ollama 모델 목록

노트북에서 사용하는 `llama3.2` 대신 다른 모델로 교체할 수 있습니다.

#### 채팅 모델 (ChatOllama)

| 모델 | 크기 | VRAM | 특징 |
|------|------|------|------|
| `llama3.2` | 3B | 4GB | 기본 모델, 가벼움, Colab 무료 티어 가능 |
| `llama3.2:1b` | 1B | 2GB | 초경량, 저사양 환경용 |
| `llama3.1` | 8B | 8GB | 더 높은 성능, 일반 GPU 필요 |
| `llama3.1:70b` | 70B | 40GB+ | 최고 성능, 고사양 GPU 필수 (A100 등) |
| `gemma2` | 9B | 8GB | Google 모델, 균형 잡힌 성능 |
| `gemma2:2b` | 2B | 3GB | 경량 버전 |
| `mistral` | 7B | 6GB | 빠른 추론 속도 |
| `qwen2.5` | 7B | 6GB | 중국어/영어 이중 언어 |
| `phi3` | 3.8B | 4GB | Microsoft 모델, 소형이지만 우수한 성능 |

#### 한국어 특화 모델

| 모델 | 크기 | VRAM | 특징 |
|------|------|------|------|
| `gemma2-ko` | 2B | 3GB | 한국어 미세조정, 경량 |
| `qwen2.5:7b` | 7B | 6GB | 한국어 포함 다국어 지원 우수 |
| `exaone` | 8B | 8GB | LG AI Research, 한국어 특화 |
| `solar` | 10.7B | 10GB | Upstage, 한국어 성능 우수 |

#### 임베딩 모델 (OllamaEmbeddings)

| 모델 | 차원 | 특징 |
|------|------|------|
| `nomic-embed-text` | 768 | 기본 모델, 범용 |
| `mxbai-embed-large` | 1024 | 고성능 임베딩 |
| `bge-m3` | 1024 | 다국어 지원, 한국어 포함 |
| `multilingual-e5-large` | 1024 | 다국어 특화 |

#### 모델 변경 방법

```python
# 예시: 한국어 특화 모델 사용
model = ChatOllama(model='exaone')  # 또는 'solar', 'qwen2.5'

# 예시: 고성능 모델 사용 (고사양 GPU 필요)
model = ChatOllama(model='llama3.1:70b')

# 예시: 다국어 임베딩 사용
embeddings = OllamaEmbeddings(model='bge-m3')
```

#### 모델 설치

```bash
# Ollama에서 모델 다운로드
ollama pull llama3.2      # 기본 모델
ollama pull exaone        # 한국어 특화
ollama pull bge-m3        # 다국어 임베딩
```

> **참고**: Colab 무료 티어(T4 GPU, 15GB VRAM)에서는 8B 이하 모델을 권장합니다. 70B 이상 모델은 Colab Pro+ 또는 로컬 고사양 GPU가 필요합니다.

### 챕터별 노트북 (Colab에서 바로 열기)

#### Ch01 - LangChain 기초, 프롬프트, 체인

| # | 노트북 | Colab |
|---|--------|-------|
| 01 | LLM 기초 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/01.llm.ipynb) |
| 02 | Chat 모델 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/02.chat.ipynb) |
| 03 | System 메시지 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/03.system.ipynb) |
| 04 | 프롬프트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/04.prompt.ipynb) |
| 05 | 프롬프트+모델 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/05.prompt-model.ipynb) |
| 06 | Chat 프롬프트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/06.chat-prompt.ipynb) |
| 07 | Chat 프롬프트+모델 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/07.chat-prompt-model.ipynb) |
| 08 | 구조화된 출력 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/08.structured.ipynb) |
| 09 | CSV 처리 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/09.csv.ipynb) |
| 10 | 메서드 활용 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/10.methods.ipynb) |
| 11 | 명령형 방식 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/11.imperative.ipynb) |
| 12 | 스트리밍 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/12.stream.ipynb) |
| 13 | 비동기 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/13.async.ipynb) |
| 14 | 선언형 방식 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/14.declarative.ipynb) |
| 15 | 선언형+스트리밍 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/15.declarative-stream.ipynb) |
| - | Ch01 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch01/Learning_LangChain_Ch01.ipynb) |

#### Ch02 - RAG 기초, 벡터 DB

| # | 노트북 | Colab |
|---|--------|-------|
| 01 | 텍스트 로더 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/01.text-loader.ipynb) |
| 02 | 웹 로더 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/02.web-loader.ipynb) |
| 03 | PDF 로더 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/03.pdf-loader.ipynb) |
| 04 | 재귀적 텍스트 분할 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/04.rec-text-splitter.ipynb) |
| 05 | 코드 분할 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/05.rec-text-splitter-code.ipynb) |
| 06 | 마크다운 분할 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/06.markdown-splitter.ipynb) |
| 07 | 임베딩 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/07.embeddings.ipynb) |
| 08 | 로드+분할+임베딩 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/08.load-split-embed.ipynb) |
| 09-12 | PGVector | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/09-12.pg-vector.ipynb) |
| 13 | 레코드 매니저 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/13.record-manager.ipynb) |
| 14 | 멀티벡터 리트리버 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/14.multi-vector-retriever.ipynb) |
| 15 | RAG ColBERT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/15.rag-colbert.ipynb) |
| - | Ch02 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch02/Learning_LangChain_Ch02.ipynb) |

#### Ch03 - 고급 RAG, Agentic RAG

| # | 노트북 | Colab |
|---|--------|-------|
| 01-03 | RAG 기초 임베딩 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/01-03.basic-rag-embedding.ipynb) |
| 04-06 | RAG 기초 답변 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/04-06.basic-rag-answer.ipynb) |
| 07-08 | 쿼리 재작성 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/07-08.rewrite.ipynb) |
| 09-11 | 멀티 쿼리 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/09-11.multi-query.ipynb) |
| 12-14 | RAG Fusion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/12-14.rag-fusion.ipynb) |
| 15-17 | HyDE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/15-17.hyde.ipynb) |
| 18-20 | 라우터 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/18-20.router.ipynb) |
| 21 | 시맨틱 라우터 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/21.semantic-router.ipynb) |
| 22 | 텍스트 메타데이터 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/22.text-metadata.ipynb) |
| 23 | SQL 예제 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/23.sql-example.ipynb) |
| - | Ch03 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch03/Learning_LangChain_Ch03.ipynb) |

#### Ch04 - 메모리 관리

| # | 노트북 | Colab |
|---|--------|-------|
| 01 | 간단한 메모리 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch04/01.simple-memory.ipynb) |
| 02-06 | 상태 그래프 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch04/02-06.state-graph.ipynb) |
| 07-10 | 영속 메모리 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch04/07-10.persistent-memory.ipynb) |
| 11 | 메시지 자르기 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch04/11.trim-messages.ipynb) |
| 12-14 | 메시지 필터링 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch04/12-14.filter-messages.ipynb) |
| 15-16 | 메시지 병합 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch04/15-16.merge-messages.ipynb) |
| - | Ch04 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch04/Learning_LangChain_Ch04.ipynb) |

#### Ch05 - LangGraph 기본 챗봇

| # | 노트북 | Colab |
|---|--------|-------|
| 01-03 | 챗봇 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch05/01-03.chatbot.ipynb) |
| 04-05 | SQL 생성기 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch05/04-05.sql-generator.ipynb) |
| 06-07 | 멀티 RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch05/06-07.multi-rag.ipynb) |
| - | Ch05 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch05/Learning_LangChain_Ch05.ipynb) |

#### Ch06 - 에이전트와 도구

| # | 노트북 | Colab |
|---|--------|-------|
| 01-02 | 기본 에이전트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch06/01-02.basic-agent.ipynb) |
| 03-04 | 도구 우선 실행 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch06/03-04.force-first-tool.ipynb) |
| 05-06 | 다중 도구 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch06/05-06.many-tools.ipynb) |
| - | Ch06 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch06/Learning_LangChain_Ch06.ipynb) |

#### Ch07 - 고급 패턴

| # | 노트북 | Colab |
|---|--------|-------|
| 01 | 리플렉션 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch07/01.reflection.ipynb) |
| 02 | 서브그래프 (직접) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch07/02.subgraph-direct.ipynb) |
| 03 | 서브그래프 (함수) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch07/03.subgraph-function.ipynb) |
| 04-05 | 슈퍼바이저 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch07/04-05.supervisor.ipynb) |
| - | Ch07 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch07/Learning_LangChain_Ch07.ipynb) |

#### Ch08 - 고급 기능

| # | 노트북 | Colab |
|---|--------|-------|
| 01-02 | 구조화된 출력 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch08/01-02.structured-output.ipynb) |
| 03 | 스트리밍 출력 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch08/03.streaming-output.ipynb) |
| 04 | 토큰 스트리밍 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch08/04.streaming-token.ipynb) |
| 05-06 | 인터럽트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch08/05-06.interrupt.ipynb) |
| 07-11 | 상태 관리 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch08/07-11.state-management.ipynb) |
| - | Ch08 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch08/Learning_LangChain_Ch08.ipynb) |

#### Ch09 - LangGraph Cloud 배포 (노트북 미제공)

#### Ch10 - 평가

| # | 노트북 | Colab |
|---|--------|-------|
| 01-02 | 검색 및 관련성 평가 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch10/01-02.retrieve_and_grade.ipynb) |
| 03-06 | LangSmith 평가 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch10/03-06.evaluation.ipynb) |
| - | Ch10 통합 노트북 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songys/learning-langchain/blob/main/python/ch10/Learning_LangChain_Ch10.ipynb) |

### ch09에 노트북이 없는 이유

ch09는 **LangGraph Cloud에 배포하기 위한 프로젝트 구조**입니다:
- 외부 서비스 의존성 (Supabase, LangSmith)
- `langgraph.json`, `pyproject.toml` 등 프로젝트 설정 파일 기반
- 로컬 서버 실행 필요 (`langgraph dev`)

Colab에서 단독 실행이 어려우므로 로컬 환경에서 `python/ch09/README.md`를 참조하여 실습하세요.

### ch10 일부 파일에 노트북이 없는 이유

ch10에는 다음 노트북이 제공됩니다:
- `01-02.retrieve_and_grade.ipynb` - 검색 및 관련성 평가
- `03-06.evaluation.ipynb` - LangSmith 평가 개념 설명

나머지 파일들(`agent_sql_graph.py`, `rag_graph.py` 등)은 **LangSmith 연동이 필수**입니다. LangSmith는 LangChain의 평가/모니터링 플랫폼으로, API 키와 프로젝트 설정이 필요하여 Colab에서 단독 실행이 어렵습니다.

### 권장 학습 방법

| 챕터 | 학습 방법 |
|------|----------|
| ch01~ch08 | Google Colab에서 노트북 실행 |
| ch09 | 로컬 환경에서 배포 실습 |
| ch10 (노트북) | Colab에서 개념 학습 |
| ch10 (평가 스크립트) | LangSmith 계정 생성 후 로컬에서 실행 |
