# 사용 기술과 구현 상세

이 문서는 시스템을 만드는 데 쓴 기술(언어, 라이브러리, 알고리즘, 인프라)을 영역별로 정리하고,
각 기술이 코드 어디에서 어떻게 쓰였는지와 그렇게 선택한 이유를 설명합니다.
구조 전체는 [architecture.md](architecture.md), 단계별 경과는 [work-log.md](work-log.md)를 참고하세요.

규모: 패키지 `alphaagent` 약 6,500줄, 테스트 약 1,200줄(86개).

## 1. 기술 스택 한눈에 보기

| 영역 | 기술 | 개발 환경 버전 | 사용 위치 |
|---|---|---|---|
| 언어 | Python | 3.9.10 (배포 이미지 3.11) | 전체 |
| 수치 계산 | NumPy, pandas | 1.26.4, 2.3.3 | 지표, 백테스트, 패널 데이터 |
| 벡터 DB | FAISS (faiss-cpu) | 1.13.0 | `vectorstore/faiss_store.py` |
| 임베딩 | sentence-transformers, PyTorch, BGE-M3, FinBERT | 5.1.2, 2.2.2 (CPU) | `embeddings/sentence.py` |
| HTTP 클라이언트 | httpx | 0.28.1 | SEC·RSS 수집, vLLM 클라이언트 |
| RSS 파싱 | feedparser | 6.0.12 | `ingestion/news.py` |
| 웹 API | FastAPI, Pydantic v2, Uvicorn | 0.128.8, 2.13.5, 0.39.0 | `retrieval/api.py`, `ops/api.py` |
| LLM | Anthropic Python SDK (Claude) | 0.125.0 | `llm/anthropic_client.py` |
| 로컬 LLM | vLLM OpenAI 호환 서버 (HTTP) | – | `llm/vllm_client.py` |
| 백테스트 | 자체 벡터화 엔진, Backtrader | 1.9.78.123 | `backtest/` |
| 메시징 | Apache Kafka, confluent-kafka, Strimzi | Kafka 3.7 | `streaming/`, `deploy/k8s/kafka.yaml` |
| 컨테이너 | Docker, Docker Compose | – | `deploy/` |
| 오케스트레이션 | Kubernetes, Kustomize | – | `deploy/k8s/` |
| 테스트 | pytest, httpx MockTransport, FastAPI TestClient | 8.4.2 | `tests/` |
| 표준 라이브러리 | `ast`, `subprocess`, `hashlib`, `html.parser`, `threading`, `dataclasses`, `argparse`, `pickle`, `json` | – | 샌드박스, 감사, 수집 등 |

선택적 의존성은 `pyproject.toml`의 extras로 나눴습니다: `embeddings`, `llm`, `backtest`, `streaming`, `market`, `dev`.
핵심 기능은 extras 없이도 동작하고, 무거운 라이브러리(torch, anthropic, confluent-kafka, yfinance)는 쓰는 시점에만 import합니다.

## 2. 문서 수집

### SEC EDGAR (`ingestion/sec.py`)
- **기술**: EDGAR 공개 JSON API (`company_tickers.json`, `data.sec.gov/submissions/CIK##########.json`, `Archives/edgar/data/...`), httpx
- **구현**: 종목 → CIK 변환 → 최근 제출 목록에서 원하는 양식(10-K/10-Q/8-K) 필터 → 본문 HTML 다운로드 → 텍스트 변환.
  문서 날짜는 보고 기간 종료일이 아니라 **제출일**로 둡니다(시장이 처음 읽을 수 있었던 날).
- **SEC 정책 준수**: 연락처 이메일이 든 User-Agent가 없으면 생성 단계에서 거부하고, 요청 간 최소 0.12초 간격(초당 10회 이하)을 둡니다.

### 뉴스 (`ingestion/news.py`)
- **기술**: RSS/Atom + feedparser, 야후 파이낸스 종목별 헤드라인 RSS, 로컬 JSONL
- **구현**: httpx로 받은 바이트를 feedparser로 파싱, `published_parsed`를 UTC 날짜로 변환. 날짜 없는 항목은 버립니다.

### HTML → 텍스트 (`ingestion/html_text.py`)
- **기술**: 표준 라이브러리 `html.parser.HTMLParser`
- **구현**: `script/style/head`와 XBRL 머리말(`ix:header`)은 건너뛰고, 블록 태그는 줄바꿈으로 바꾼 뒤 공백을 정리합니다. 외부 의존성(BeautifulSoup 등) 없이 EDGAR 공시를 처리하려고 직접 만들었습니다.

### 청크 분할 (`ingestion/chunking.py`)
- **알고리즘**: 문단 단위로 채우다가 크기(기본 1,200자)를 넘으면 새 청크. 너무 긴 문단은 문장(정규식 `(?<=[.!?])\s+`)으로, 그래도 길면 고정 길이로 자릅니다.
  다음 청크 앞에 이전 청크 끝 200자를 단어 경계에서 붙여 문맥이 끊기지 않게 합니다.
- **메타데이터 머리말**: 각 청크 앞에 `[TICKER | 유형 | 날짜] 제목`을 붙여, 본문에 회사명이 없어도 임베딩이 어느 회사·시점의 글인지 반영하게 했습니다.

### 중복 제거
- `doc_id = sha1(ticker, doc_type, date, source)` 앞 16자리, `chunk_id = sha1(doc_id, 위치)`.
  같은 문서를 몇 번 받아도 저장소에 한 번만 들어가며, 이 성질이 Kafka 최소 1회 전달과 맞물려 중복 없는 실시간 수집을 가능하게 합니다.

## 3. 임베딩

| 임베더 | 모델 | 차원 | 용도 |
|---|---|---|---|
| `bge_m3()` | `BAAI/bge-m3` | 1024 | 기본. 다국어·장문(최대 8,192토큰, 기본 1,024로 제한) 지원 |
| `finbert()` | `ProsusAI/finbert` + 평균 풀링 | 768 | 금융 도메인 대안 |
| `HashingEmbedder` | 없음 | 512 (설정 가능) | 오프라인·테스트 |

- **sentence-transformers**: `SentenceTransformer.encode(normalize_embeddings=True)`로 L2 정규화 벡터를 얻습니다. 정규화 덕분에 내적 = 코사인 유사도가 되어 FAISS 내적 인덱스를 그대로 씁니다.
- **해싱 임베더**: 소문자 토큰과 바이그램을 BLAKE2b로 해싱해 차원 인덱스와 부호(±1)를 정하는 feature hashing. 모델 다운로드 없이 결정적으로 동작해 CI와 데모에 씁니다.
- 원문 예제는 문서마다 "Represent this financial text..." 지시문을 붙였지만, BGE-M3 밀집 검색은 지시문이 필요 없어 넣지 않았습니다.

## 4. 벡터 데이터베이스: FAISS (`vectorstore/faiss_store.py`)

- **인덱스**: `IndexIDMap2(IndexFlatIP(dim))`. 정확한(brute-force) 내적 검색에 사용자 지정 64비트 ID를 붙인 구성입니다. 수백만 청크 이하에서는 근사 인덱스보다 단순하고 정확합니다.
- **필터 검색**: 메타데이터(종목·유형·날짜)를 파이썬 쪽 사전에 두고, 조건에 맞는 ID를 먼저 모아 `faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))`로 넘깁니다.
  FAISS가 허용된 ID 안에서만 top-k를 찾으므로 필터를 걸어도 결과 수가 줄지 않습니다. 종목별 ID 집합을 따로 유지해 후보를 빠르게 좁힙니다.
- **삭제**: `remove_ids`와 메타데이터 정리.
- **영속화**: `faiss.serialize_index` → 바이트를 파이썬으로 기록. FAISS C++ 파일 쓰기가 Windows의 비ASCII 경로(사용자 폴더명)에서 실패할 수 있어 이렇게 했습니다.
  메타데이터는 JSONL. 두 파일 모두 `.tmp`에 쓴 뒤 `os.replace`로 원자적으로 교체해, 다른 프로세스가 반쯤 쓰인 인덱스를 읽지 않습니다.
- **교체 가능성**: `VectorStore` 추상 클래스(`add / search / has_document / delete_document / ticker_counts`)만 맞추면 Pinecone·Weaviate로 바꿀 수 있습니다.

## 5. 검색 API (`retrieval/`)

- **FastAPI + Pydantic v2**: 요청 모델(`SearchRequest`, `DocumentIn`)에 범위 검증(k 1~100 등)을 두고, 응답은 `response_model=List[Hit]`로 고정합니다.
- **엔드포인트**: `/health`, `/tickers`, `GET·POST /search`, `/tickers/{ticker}/search`, `POST /documents`.
- **동시성**: 검색과 색인 추가를 `threading.Lock`으로 직렬화(FAISS 인덱스는 동시 쓰기에 안전하지 않음).
- **자동 재로드**: `ReloadingRetriever`가 최대 30초마다 메타데이터 파일 수정 시각을 보고, 컨슈머가 인덱스를 갱신하면 다시 읽습니다.
- **지연 생성**: 모듈 수준 `__getattr__`로 `app`을 요청 시점에 만들어 `uvicorn alphaagent.retrieval.api:app`이 설정을 읽고 앱을 구성하게 했습니다.
- **LLM용 컨텍스트**: `Retriever.context()`가 결과를 `[n] (종목, 유형, 날짜, 출처)` 형식으로 번호를 매겨 묶어, 에이전트가 근거를 `[n]`으로 인용할 수 있게 합니다.

## 6. LLM 백엔드 (`llm/`)

### 공통 인터페이스
`LLMClient.complete(prompt, system, max_tokens) -> LLMResponse(text, model, input_tokens, output_tokens, stop_reason)`.
호출 수와 토큰을 누적하고, 에이전트는 이 인터페이스만 알기 때문에 공급자를 바꿔도 에이전트 코드는 그대로입니다.

### Claude (`anthropic_client.py`)
- **Anthropic Python SDK** 사용, 기본 모델 `claude-opus-5-5`
- **스트리밍**: `client.beta.messages.stream(...).get_final_message()`로 긴 생성에서 HTTP 타임아웃을 피합니다.
- **추론 강도**: `output_config={"effort": "high"}`. 이 모델은 사고(thinking)를 끌 수 없고 effort로 깊이를 조절하므로 `thinking`·`temperature`는 보내지 않습니다.
- **거절 대비**: 베타 `server-side-fallback-2026-07-01` + `fallbacks="default"`로 정책 거절 시 서버가 대체 모델로 재시도합니다. 그래도 `stop_reason == "refusal"`이면 `LLMError`로 올립니다.
- **인증**: `ANTHROPIC_API_KEY` 또는 `ant auth login` 프로필을 SDK가 자동으로 읽습니다. 코드에 키를 넣지 않습니다.

### vLLM (`vllm_client.py`)
- 로컬 오픈 모델(예: Llama-3-70B-Instruct)을 `vllm serve`로 띄운 OpenAI 호환 `/v1/chat/completions` 엔드포인트에 httpx로 요청합니다. 별도 SDK 없이 HTTP만 씁니다.

### 모의 백엔드 (`mock.py`, `mock_handlers.py`)
- 프롬프트 첫 줄 `TASK_KIND: ideation|implementation|repair|evaluation|text_scoring|decomposition|synthesis|portfolio_review|risk_review|vote`로 작업을 구분하고, 작업별 처리기가 형식에 맞는 응답을 돌려줍니다.
- 처리기는 실제로 작동하는 팩터 코드 7종, 어휘 기반 문서 채점기, 계획·종합·검토·투표 규칙을 담고 있어 API 키 없이 전체 시스템이 끝까지 돌아갑니다.
- `ScriptedLLM`은 정해진 응답 목록이나 함수를 돌려줘, 잘못된 JSON·위험한 코드·과한 배율 같은 경계 상황을 테스트합니다.

### 회복성·감사 래퍼 (`ops/`)
- `ResilientLLM`: 회로 차단기로 감싸고, 열려 있으면 대체 클라이언트(예: 로컬 vLLM)로 넘깁니다.
- `AuditedLLM`: 모든 호출의 모델, 작업 종류, 프롬프트·응답 SHA-256, 토큰, 지연 시간을 감사 기록에 남깁니다. 본문은 기본적으로 저장하지 않습니다(유료 데이터가 섞일 수 있음).
- 운영 CLI는 `공급자 → ResilientLLM → AuditedLLM` 순서로 조립합니다.

## 7. 에이전트 기법 (`agents/`)

| 기법 | 구현 |
|---|---|
| **RAG** | 매 호출마다 작업 내용으로 벡터 검색(종목·기준일 필터) → 번호 붙은 근거를 프롬프트에 삽입 |
| **Chain-of-Thought** | 프롬프트 끝에 4단계 지시(근거 확인 → 정량 추론 → 누수·근거 없는 주장 점검 → 답과 신뢰도), `## Reasoning` / `## Answer` / `[Confidence: x]` 형식 |
| **구조화 출력** | 아이디어·평가·계획·투표는 ```json 블록으로 받음. 파서는 마지막 json 블록 → 괄호 균형이 맞는 첫 `{...}`/`[...]` 순으로 시도(문자열 안 괄호 처리) |
| **코드 추출** | 마지막 ```python 블록 → 태그 없는 블록 → 원문 순 |
| **신뢰도 추출** | `[Confidence: 0.72]`의 마지막 값, 1보다 크면 백분율로 해석, 0~1로 자름 |
| **공유 메모리** | 라운드 번호가 붙은 `MemoryEntry` 목록, 역할·종류·라운드로 조회, 최근 항목 요약을 프롬프트에 포함, JSONL 영속화, 스레드 안전 |
| **메시지 프로토콜** | `AgentMessage(sender, recipient, type, content, payload, correlation_id)`, 유형 TASK/RESULT/CRITIQUE/VOTE/ALERT/INFO, `*` 브로드캐스트, 받은편지함(pull) + 구독(push), 모든 메시지를 메모리에 기록 |
| **피드백 루프** | Alpha-GPT 방식. 지난 팩터의 이름·식·상태·OOS IC·잔차 IC·가장 비슷한 알려진 팩터·오류를 다음 아이디어 프롬프트에 넣음 |
| **자기 수정** | 샌드박스·편향 검사 오류 메시지를 구현 에이전트에 돌려줘 코드 수정 (기본 2회) |
| **계층 구조** | FinCon 방식 Manager → 분석가, `Strategy = Manager(∪ Analyst_i(Task_i))` |

에이전트 역할: Ideation, Implementation, Evaluation, Manager, Analyst(감성·펀더멘털·기술·리스크), Portfolio, Risk.

## 8. 생성 코드 샌드박스 (`sandbox/`)

LLM이 쓴 파이썬 코드를 안전하게 실행하기 위한 다층 방어입니다.

| 층 | 기술 | 내용 |
|---|---|---|
| 정적 정책 | `ast` 모듈 | import, `open/exec/eval/compile/getattr/__import__` 등 이름, dunder 속성, 파일·네트워크·코드 실행 관련 pandas/numpy 메서드(`read_*`, `to_*`, `query`, `eval` 등), `global`, async·generator, 모듈 최상위 실행문 금지. `compute_alpha(df)` 정의 필수, 2만 자 제한 |
| 프로세스 격리 | `subprocess` + `python -I` | 별도 인터프리터. `-I`는 사용자 site-packages·환경 변수·스크립트 경로를 sys.path에서 뺌. 임시 디렉터리를 작업 폴더로 사용 |
| 실행 환경 | 축소 builtins | `abs, len, range, sum, zip` 등 안전한 내장 함수만 넣은 `__builtins__`, `pd`·`np`만 제공. 데이터셋마다 새 네임스페이스 |
| 자원 제한 | `subprocess.run(timeout=)`, `resource.setrlimit(RLIMIT_AS)` | 시간 제한(기본 60초×데이터셋 수), POSIX에서 메모리 상한 |
| 결과 회수 | `np.save/np.load(allow_pickle=False)` + JSON | 자식이 만든 pickle은 절대 열지 않음. 입력은 부모가 만든 pickle만 자식이 읽음 |
| 출력 검증 | pandas | Series 여부, 인덱스 정합(필요 시 reindex), inf → NaN, 커버리지 30% 이상, 날짜 대부분에서 종목 간 값이 달라야 함 |
| 묶음 실행 | `run_many` | 전체 데이터 + 절단 데이터 여러 개를 한 프로세스에서 실행해 기동 비용(Windows 약 3초)을 한 번만 지불 |

운영 환경에서는 여기에 쿠버네티스 보안 설정(비루트, 읽기 전용 FS, 권한 제거, NetworkPolicy)이 더해집니다.

## 9. 예측(미래 정보) 편향 탐지 (`validation/lookahead.py`)

- **정적 검사 (AST)**: `shift/diff/pct_change`에 음수 기간, `bfill/backfill`, `fillna(method="bfill")`, `rolling(center=True)`는 오류. 
  rolling/expanding/groupby 없이 열 전체에 `mean/std/...`를 쓰면 경고(전체 표본 통계 의심).
- **절단 테스트 (동적)**: 기준일을 여러 개(표본의 45~85% 지점) 골라, 그 이후 데이터를 지운 패널로 다시 계산했을 때
  기준일 직전 30일의 값이 전체 데이터로 계산한 값과 같은지 비교합니다(허용 오차 1e-8, NaN 위치까지 비교).
  미래 정보를 쓰는 코드는 형태와 관계없이 이 테스트에 걸립니다. 정적 검사로는 경고에 그치는 "전체 표본 평균으로 정규화" 누수를 이 테스트가 확정합니다.

## 10. 팩터 평가와 통계 (`backtest/metrics.py`, `validation/`)

### 평가 지표
| 지표 | 계산 방법 |
|---|---|
| IC | 날짜별 횡단면 피어슨 상관(알파, h일 선행 수익률)의 평균. 날짜별 편차를 `groupby.transform("mean")`으로 빼고 공분산·분산을 한 번에 집계하는 **벡터화** 구현 |
| Rank IC | 날짜별 순위로 바꾼 뒤 같은 계산(스피어만) |
| ICIR, t값 | IC 평균 / 표준편차, × √날짜 수 |
| 롱숏 포트폴리오 | 날짜별 순위 → 평균 차감 → 절대값 합 1로 정규화(달러 중립, 총노출 1) |
| 샤프·연수익·MDD | 일별 포트폴리오 수익 기준, 252일 연율화 |
| 회전율 | 일별 비중 변화 절대값 합의 평균 |

### 검증 기법
| 기법 | 구현 |
|---|---|
| IS/OOS 분할 | 기간 앞 70% / 뒤 30%. 팩터 부호는 IS IC로만 정함 |
| 워크포워드 | 126일 학습 / 21일 검증을 굴리며 구간마다 부호 결정 후 OOS IC 측정. 학습 라벨이 검증 시작 전에 끝나도록 예측 기간만큼 **퍼지** |
| 스패닝 테스트 | 팩터 일별 수익을 알려진 팩터 포트폴리오 수익에 OLS 회귀, 절편의 t값을 **Newey-West HAC**(지연 5, Bartlett 가중) 표준오차로 계산 |
| 알려진 팩터 | 모멘텀(120일, 최근 5일 제외), 단기 반전(5일), 저변동성(20일), 규모(log 시총), 유동성(log 20일 평균 거래대금). 날짜별 z-score, ±5 절단 |
| 직교화 | 날짜별로 알파 z-score를 알려진 팩터 z-score에 OLS(`np.linalg.lstsq`) 회귀해 잔차를 구함. 평균 상관, 평균 R², 잔차 IC 산출 |
| 중복도 | 이미 채택된 팩터와의 날짜별 순위 상관 평균 |
| 시점 기준 재실행 | 기준일마다 그 날까지의 데이터로 파이프라인 전체 실행, 지금까지 채택된 팩터를 다음 21일 데이터로 채점 |

## 11. 백테스트 (`backtest/`)

- **자체 벡터화 엔진** (`engine.py`): 알파 → 순위 비중 → (선택) 리스크 오버레이 → `execution_lag`일 지연 → 다음날 수익. 
  비용 = 회전율 × bps. 리밸런싱 주기 지정 가능. pandas 행렬 연산만 써서 수백 종목 × 수백 일도 빠르게 돕니다.
  t일 신호가 t+1일 종가에 체결되고 t+2일 수익을 얻는 시점 규칙을 지키며, 테스트로 "당일 수익을 알파로 쓰면 성과가 없어야 한다"를 확인합니다.
- **Backtrader 어댑터** (`backtrader_adapter.py`): 같은 목표 비중을 Backtrader의 `order_target_percent`와 브로커 시뮬레이션(현금, 수수료)으로 재생해 엔진 결과를 교차 확인합니다.

## 12. 텍스트 신호 (`features/text_signals.py`)

- **LLM 문서 채점**: 문서마다 한 번 LLM에 보내 `sentiment(-1~1)`, `guidance(-1~1)`, `risk(0~1)`, 한 줄 근거를 JSON으로 받습니다. 범위를 벗어난 값은 잘라냅니다.
- **캐시**: 결과를 `doc_id` 키로 JSONL에 저장해 같은 문서를 다시 호출하지 않습니다(비용 절감).
- **시점 정렬**: `np.searchsorted(..., side="right")`로 발행일 **다음 거래일**부터 신호를 씁니다(장 마감 후 발표 대비).
- **감쇠**: 같은 날 여러 문서는 평균, 이후 `0.5^(경과일/반감기)`로 감쇠(기본 반감기 10일), 60거래일이 지나면 0. 문서가 없는 종목은 0(중립).
- 결과는 가격 패널과 같은 `(date, ticker)` 인덱스라 팩터 코드가 `df["sentiment"]`처럼 바로 씁니다.

## 13. 리스크와 포트폴리오

### 리스크 오버레이 (`risk/overlay.py`)
날짜마다: 섹터 내 평균 차감(섹터 중립) → 반복(총노출에 맞춰 배율 → 종목 상한 자르기 → 섹터 총노출 상한 → 섹터별로 **큰 쪽만 줄여** 순노출 0 맞춤)
→ 마지막에 `min(목표총노출/현재총노출, 상한/최대비중)` 균일 배율. 균일 배율은 중립성을 깨지 않고, 줄이기만 하므로 상한을 넘지 않습니다.

### 포트폴리오 에이전트 (`agents/portfolio.py`)
- 팩터 결합: OOS ICIR(음수는 0)에 비례한 가중치, Manager의 강조 배수(0~2) 반영, 날짜별 z-score 가중합
- 변동성 타기팅: 전일까지 60일 실현 변동성으로 목표 연 10%에 맞춰 배율(최대 2배)
- LLM 검토: 0~1 배율만 허용, 거부 시 0

### 리스크 에이전트 (`agents/risk_agent.py`)
- 지표: 과거 250일 **역사적 VaR95 / CVaR95**, 20일 실현 변동성, 낙폭, 총·순노출, 최대 비중, **HHI**(집중도), 섹터 순노출
- 조치: 위반 없음 → 유지 / 위반 → 위반 정도에 비례한 축소(변동성·VaR·총노출 비율 중 최솟값, 낙폭 위반은 0.5) / 낙폭 15% 초과 → 중단
- 위반 시 버스에 ALERT 브로드캐스트

### 앙상블 결정 (`agents/ensemble.py`)
- 투표: 팩터는 날짜별 상위 20% 매수 / 하위 20% 매도 / 나머지 보유, LLM 분석가는 근거 문서로 투표
- 결정: `ŷ = argmax_y Σ w_i·1[vote_i = y]`, 1위 동점이면 보유
- 가중치 학습: 횡단면 초과수익 기준 적중률을 지수 이동평균(학습률 0.05)으로 반영, 최소 0.02, 합 1. t일 투표는 t+h일 수익이 확정된 뒤에만 반영

## 14. 실시간 수집: Kafka (`streaming/`)

- **브로커 추상화**: `MessageBroker(publish / poll / commit)`. 구현은 `InMemoryBroker`(토픽별 로그 + 그룹별 커밋 오프셋, Kafka 의미론 재현)와 `KafkaBroker`(confluent-kafka).
- **프로듀서**: 멱등 프로듀서(`enable.idempotence=True`, `acks=all`), 키는 `doc_id`(같은 문서는 같은 파티션)
- **컨슈머**: 자동 커밋을 끄고 배치 처리(색인 + 저장 + 채점 발행)가 끝난 뒤 커밋 → **최소 1회 전달**. 문서 ID 중복 제거와 합쳐 결과적으로 정확히 한 번 색인
- **토픽**: `docs.raw`(30일 보존, 인덱스 재구축 시 재생), `docs.indexed`, `signals.text`(compact, 종목 키), `agents.messages`, `ops.alerts`(1년 보존)
- **폴러**: 소스별 예외를 격리해 한 피드가 죽어도 나머지는 계속
- **분산 에이전트 통신**: `BrokerMessageBus`가 메시지를 Kafka에 복제하고, 다른 파드는 `pull_remote`로 받아 자기 받은편지함에 넣습니다(자기 메시지는 무시)
- **Strimzi**: KRaft 모드(ZooKeeper 없음) 3노드, 복제 3, `min.insync.replicas=2`

## 15. 운영 통제 (`ops/`)

| 기술·패턴 | 구현 |
|---|---|
| **회로 차단기 패턴** | closed → (연속 실패 N회) → open → (대기 후) half_open → 성공 시 closed / 실패 시 다시 open. 시계 주입으로 테스트 가능, 상태 전이를 감사 기록 |
| **거래 차단기** | 일간 손실, 낙폭, 입력 데이터 지연, 리스크 중단 요청 시 **래치**(사람이 리셋할 때까지 정지). 회전율 초과는 부분 이동 |
| **킬 스위치** | 다음 스텝에서 전량 청산, 운영자 이름·사유 필수 |
| **사람 승인(human-in-the-loop)** | 종목당 비중 변화가 임계값을 넘으면 승인 요청 생성, 승인 전까지 기존 비중 유지, 승인되면 다음 스텝 반영. 이미 결정된 요청은 다시 결정 불가 |
| **영속 상태** | 킬 스위치·일시정지·배율·승인 내역을 JSON 파일로 저장, 재시작 후 복원 |
| **해시 체인 감사 추적** | 레코드 = {seq, ts, event, actor, details, prev_hash}, `hash = SHA-256(prev_hash + 정렬된 JSON)`. 첫 레코드의 prev_hash는 0×64. 재시작 시 마지막 해시부터 이어 씀. `verify()`가 처음부터 다시 계산해 끊긴 위치 보고 |
| **규제 보고서** | 기간별 이벤트 집계, 알고리즘(팩터) 목록과 코드 해시·승인자, LLM 사용량(모델·작업별), 리스크 조치, 차단기·킬 스위치·사람 개입 이력. JSON과 Markdown |
| **운영자 API** | FastAPI `APIRouter(prefix="/ops")`, 모든 변경에 `X-Operator` 헤더 필수(없으면 401), 중복 승인 409 |
| **모의 운용 루프** | 손익 실현 → 승인분 반영 → 킬 스위치 → 팩터 재계산(샌드박스) → 포트폴리오 → 리스크 → 운영자 배율 → 거래 차단기 → 승인 대기 → 주문 기록 |

## 16. 모니터링 (`monitoring/`)

- **롤링 IC**(20일), **IC 감쇠 곡선**(1·2·3·5·10·20일 뒤의 하루 수익과의 IC), **반감기**(IC가 1일 값의 절반이 되는 지점을 선형 보간)
- **상태 판정**: 최근 60일·20일 IC가 모두 0 이하면 dead, 60일 IC / 전체 IC < 0.5면 decaying, 그 외 healthy
- **대시보드**: 외부 스크립트나 CDN 없이 인라인 SVG로 그린 단일 HTML. 누적 수익, 롤링 IC, 감쇠 곡선, 낙폭 차트와 상태 표. CSS 변수와 `prefers-color-scheme`으로 다크 모드 지원

## 17. 배포 (`deploy/`)

| 기술 | 구현 |
|---|---|
| **Docker** | `python:3.11-slim`, librdkafka, CPU용 PyTorch, uid 10001 비루트 사용자, `/data`·`/models` 볼륨, `/health` 헬스체크, `alphaagent` 엔트리포인트 하나로 API·컨슈머·폴러 겸용 |
| **Docker Compose** | Kafka(KRaft 단일 노드) + api + ingest-consumer + pollers, 비밀값은 `.env` |
| **Kubernetes** | Namespace(Pod Security `restricted`), ConfigMap, RWX PVC, api Deployment + Service + **HPA**(CPU 70%, 2~8), 컨슈머 Deployment(1개, Recreate, 인덱스 단일 작성자), 폴러, 야간 연구 **CronJob**(평일 22:30 뉴욕 시간, 동시 실행 금지) |
| **보안 설정** | `runAsNonRoot`, `readOnlyRootFilesystem`, `capabilities.drop: [ALL]`, `allowPrivilegeEscalation: false`, seccomp `RuntimeDefault`, 쓰기는 PVC와 `emptyDir` `/tmp`만 |
| **NetworkPolicy** | 기본 전면 차단 → DNS, Kafka 9092, 사설 대역을 뺀 외부 443, 인그레스 컨트롤러 → API 8000만 허용 |
| **Kustomize** | `kubectl apply -k deploy/k8s`, 이미지 태그를 `kustomization.yaml`에서 관리 |
| **비밀 관리** | 매니페스트에는 예시만, 실제 값은 `kubectl create secret` 또는 External/Sealed Secrets |

## 18. 테스트와 검증 방법 (`tests/`)

| 기법 | 사용 예 |
|---|---|
| **pytest 픽스처** | 샘플 코퍼스를 색인한 검색기, 합성 데이터셋을 모듈 단위로 재사용 |
| **httpx MockTransport** | SEC EDGAR·RSS·vLLM 서버 응답을 흉내 내 네트워크 없이 수집기 검증 |
| **FastAPI TestClient** | 검색 API, 운영자 API 전 엔드포인트 |
| **가짜 SDK 객체** | Claude 요청 인자(모델, effort, fallback 베타, 금지 파라미터 미전송)와 거절 처리 검증 |
| **심어 둔 신호** | 합성 가격에 단기 반전, 합성 문서 톤에 발행 후 드리프트를 심어 "찾아야 할 것은 찾고, 잡음에서는 못 찾는지" 확인 |
| **반례 테스트** | 미래를 보는 코드, 전체 표본 정규화, 당일 수익 알파, 완벽한 예지 알파, 위장한 알려진 팩터, 무작위 투표자 |
| **장애 주입** | 소비 후 커밋 전 중단(rewind), 중복 전달, 실패하는 소스, 연속 API 오류, 감사 기록 변조 |
| **가짜 시계** | 회로 차단기 대기 시간 |

합성 데이터의 종목은 모두 가상(`TICK_xxx`, `ACME/GLBX/INIT`)이고 문서는 문장 은행에서 생성합니다.

## 19. 알고리즘 출처

| 아이디어 | 출처 (원문 인용) | 이 저장소에서 |
|---|---|---|
| 아이디어 → 구현 → 검증 → 피드백 루프 | Alpha-GPT (Wang et al., 2023; Yuan et al., 2024) | `pipeline.py`, `library.feedback()` |
| Manager-Analyst 계층 | FinCon (Yu et al., 2025) | `agents/manager.py`, `team.py` |
| 가중 투표 앙상블 | TradingAgents (Xiao et al., 2024) | `agents/ensemble.py` |
| Search-Analyze-Finalize | SAF (Kou et al., 2025) | 파이프라인 단계 구성 |
| RAG, CoT | 원문 이론 절 | `agents/base.py` |
| Newey-West HAC, Fama-MacBeth식 횡단면 회귀, 워크포워드 퍼지 | 계량재무 표준 기법 | `validation/` |
