# One-click 실행(macOS) 안내

본 저장소는 컴퓨터 비전공자(언어학 연구자)도 실험을 재현할 수 있도록,
명령어 입력을 최소화한 "원클릭 실행 메뉴(셸 스크립트)"를 제공한다.

---

## 1. 준비물(최소 요건)

- macOS (Apple Silicon 또는 Intel)
- Python 3.11 (설치 권장)
- Terminal (macOS 기본 탑재)
- (DeepSeek 실험 시) Ollama 설치 권장

> 실험에 필요한 Python 패키지는 메뉴에서 자동 설치된다.

---

## 2. 실행 순서(가장 쉬운 루트)

1) 저장소 다운로드(ZIP) 후 압축 해제
2) 저장소 루트에 `.env.example`을 복사하여 `.env` 생성
3) `.env` 파일에 필요한 키 값을 입력
4) Terminal에서 `bash menu.sh` 실행
5) 메뉴에서 번호 선택

---

## 3. 메뉴 기능 설명

### (A) 환경 설치/갱신
- [1] GPT-5 Cloud API 환경 설치/갱신
- [2] DeepSeek-R1 로컬 환경 설치/갱신

> 처음 실행하는 Mac이라면 반드시 (A)를 먼저 수행한다.

### (B) 실험 실행
- [3] GPT-5 No-RAG 실행
- [4] GPT-5 Plain-RAG 실행 (로컬 저장)
- [5] GPT-5 Plain-RAG 실행 (LangSmith 추적)
- [6] DeepSeek-R1 No-RAG 실행
- [7] DeepSeek-R1 Plain-RAG 실행

---

## 4. DeepSeek 실험(Ollama) 관련 안내

DeepSeek-R1 실험은 로컬 실행 기반이며, 다음이 필요할 수 있다.

- Ollama 설치 (`brew install ollama` 또는 [공식 사이트](https://ollama.com)에서 다운로드)
- 모델 다운로드:
  - `deepseek-r1:14b`
  - `nomic-embed-text` (Plain-RAG에서 임베딩용)

본 저장소는 실행 전 점검(doctor) 단계에서
모델이 없으면 자동으로 경고 메시지를 표시한다.

---

## 5. 자주 발생하는 문제(FAQ)

### Q1. `menu.sh` 실행 시 "permission denied" 오류가 나와요.
- 실행 권한을 부여한다: `chmod +x menu.sh`
- 또는 `bash menu.sh`로 직접 실행한다.

### Q2. `.env`를 만들었는데 키가 없다고 나와요.
- `.env.example` → `.env`로 "복사"가 되었는지 확인한다.
- OpenAI / LangSmith 키 값이 공백이 아닌지 확인한다.

### Q3. DeepSeek가 "모델이 없다"고 나와요.
- doctor 메시지대로 `ollama pull ...`을 진행한다.
- 최초 1회 다운로드 시간이 걸릴 수 있다.

### Q4. Python 3.11이 설치되어 있지 않아요.
- Homebrew를 이용하여 설치한다: `brew install python@3.11`
- 또는 [python.org](https://www.python.org/downloads/)에서 macOS용 설치 파일을 다운로드한다.

---

## 6. Windows 버전 안내

Windows 환경에서 실행하려면 아래 저장소를 참고한다.

- [LLM_pipeline_practice (Windows 버전)](https://github.com/karmalet/LLM_pipeline_practice)

