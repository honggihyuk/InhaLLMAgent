# InhaLLMAgent — RAG 기반 LLM 알파 마이닝 에이전트

비정형 금융 텍스트(실적 발표 녹취록, SEC 공시, 뉴스)에서 알파 팩터를 찾는 RAG + 멀티 에이전트 시스템입니다.
"How Quants Use LLM Agents To Mine Alpha From Unstructured Data (The Complete RAG Framework)" 글의
아키텍처와 12주 로드맵을 순서대로 구현합니다.

## 로드맵 진행 상황

| 주차 | 내용 | 상태 |
|---|---|---|
| 1-2 | 벡터 DB(FAISS), 문서 수집, BGE-M3/FinBERT 임베딩, 종목 필터 검색 API | ✅ |
| 3-4 | LLM 백엔드, CoT QuantAgent, 아이디어·구현·평가 에이전트, 공유 메모리 | ✅ |
| 5-6 | AlphaGenerationPipeline, 코드 샌드박스, 백테스트, 예측 편향 탐지·직교화 | ✅ |
| 7-8 | 12개월+ 검증, 알려진 팩터 비교, 리스크 오버레이, 모니터링 대시보드 | ✅ |
| 9-10 | 관리자·포트폴리오·리스크 에이전트, 앙상블 결정 | ⏳ |
| 11-12 | 쿠버네티스, Kafka 실시간 수집, 회로 차단기·수동 개입, 감사 추적 | ⏳ |

## 설치

```bash
pip install -e ".[embeddings,llm,dev]"
```

## 1-2주차: 인프라 및 데이터 기초

```
alphaagent/
  documents.py          Document / Chunk / SearchFilter (시점 기준 필터 포함)
  ingestion/            수집 파이프라인
    transcripts.py      로컬 녹취록 (json, jsonl, TICKER_YYYY-MM-DD_*.txt)
    sec.py              SEC EDGAR 공시 (10-K/10-Q/8-K, API 키 불필요)
    news.py             RSS(야후 파이낸스 등) / JSONL 뉴스
    chunking.py         문단·문장 단위 청크 + 오버랩
    pipeline.py         수집 → 청크 → 임베딩 → 저장, 문서 중복 제거
  embeddings/           BGE-M3(기본), FinBERT, 해싱(오프라인·테스트용)
  vectorstore/          FAISS 저장소 (종목·문서유형·날짜 필터, 저장/로드)
  retrieval/            Retriever + FastAPI 검색 API
```

- **시점 기준 검색**: `end_date`를 주면 그 날짜 이후 문서는 검색되지 않습니다. 백테스트에서 미래 정보 누수를 막는 기본 장치입니다.
- **필터 검색**: FAISS `IDSelectorBatch`로 허용 ID 안에서만 검색하므로, 종목 필터를 걸어도 결과가 k개보다 적게 나오지 않습니다.
- **SEC 공시**: SEC 정책상 연락처 이메일이 들어간 User-Agent가 필요합니다. `SEC_USER_AGENT="Inha Research you@example.com"`.

### 사용 예

```bash
# 오프라인으로 빠르게 시험할 때는 ALPHA_EMBEDDER=hashing (기본은 bge-m3, 최초 실행 시 모델 다운로드)
alphaagent ingest transcripts --path data/samples/transcripts
alphaagent ingest news --path data/samples/news
alphaagent ingest news --tickers AAPL MSFT                 # 야후 파이낸스 RSS
alphaagent ingest sec --tickers AAPL --forms 10-K 10-Q --limit 4
alphaagent search "guidance cut and liquidity" --ticker GLBX --end-date 2026-02-28
alphaagent serve --port 8000
```

API:

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 상태, 청크 수 |
| GET | `/tickers` | 종목별 청크 수 |
| GET | `/search?q=...&ticker=AAPL&doc_type=news&end_date=...` | 필터 검색 |
| POST | `/search` | JSON 본문 검색 (`tickers`, `doc_types`, `start_date`, `end_date`, `k`) |
| GET | `/tickers/{ticker}/search?q=...` | 단일 종목 검색 |
| POST | `/documents` | 문서 추가 (즉시 임베딩·색인) |

### 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ALPHA_DATA_DIR` | `data` | 인덱스 저장 위치 (`data/index`) |
| `ALPHA_EMBEDDER` | `bge-m3` | `bge-m3`, `finbert`, `hashing`, 또는 sentence-transformers 모델 ID |
| `ALPHA_EMBED_DEVICE` | `cpu` | `cuda` 가능 |
| `ALPHA_CHUNK_SIZE` / `ALPHA_CHUNK_OVERLAP` | `1200` / `200` | 문자 단위 |
| `SEC_USER_AGENT` | (없음) | SEC 수집 시 필수 |

## 3-4주차: LLM 백엔드 및 에이전트 프레임워크

```
alphaagent/
  llm/
    base.py             LLMConfig / LLMClient 인터페이스 (에이전트는 이것만 사용)
    anthropic_client.py Claude (기본 claude-opus-5-5, 스트리밍, effort, 거절 시 서버측 fallback)
    vllm_client.py      로컬 Llama-3-70B 등 vLLM OpenAI 호환 서버
    mock.py             MockLLM / ScriptedLLM (API 키 없이 테스트·데모)
  agents/
    base.py             QuantAgent: RAG 검색 + CoT 프롬프트 + 신뢰도 추출 + 메모리 기록
    specialists.py      IdeationAgent / ImplementationAgent / EvaluationAgent
    memory.py           SharedMemory: 라운드별 기록 M(t) = M(t-1) ∪ {a_i(t)}, JSONL 영속화
    protocol.py         AgentMessage(TASK/RESULT/CRITIQUE/VOTE/ALERT) + MessageBus
    parsing.py          JSON·코드·신뢰도 추출
  backtest/metrics.py   IC, Rank IC, ICIR, t-stat, 롱숏 샤프, MDD, 회전율
```

원문 예제 대비 바꾼 점:
- **시점 고정 검색**: 에이전트가 `as_of`를 넘기면 그 날짜 이후 문서는 프롬프트에 들어가지 않습니다.
- **구조화 출력**: 아이디어와 평가는 JSON으로 받고, 코드는 ```python 블록에서 추출합니다(원문의 정규식은 코드 블록을 못 잡습니다).
- **평가 게이트**: LLM 평가자는 팩터를 거절할 수는 있지만, IC·t-stat 기준을 통과하지 못한 팩터를 승인할 수는 없습니다.
- **샤프 계산**: 원문은 알파 점수 평균으로 샤프를 계산했지만, 여기서는 순위 기반 달러중립 롱숏 포트폴리오의 실제 다음날 수익으로 계산합니다.

LLM 설정 환경 변수: `ALPHA_LLM_PROVIDER` (`anthropic`|`vllm`|`mock`), `ALPHA_LLM_MODEL`.
Claude는 `ANTHROPIC_API_KEY` 또는 `ant auth login` 프로필을 사용합니다.

## 5-6주차: 알파 생성 파이프라인

```
alphaagent/
  pipeline.py           AlphaGenerationPipeline (+ PipelineConfig)
  features/text_signals.py  문서 → LLM 점수(sentiment / guidance / risk) → 시점 기준 일별 패널
  sandbox/              생성 코드 실행 샌드박스 (AST 정책 + 격리 프로세스 + 타임아웃 + 출력 검증)
  validation/
    lookahead.py        예측(미래 정보) 편향 탐지: 정적 패턴 + 데이터 절단 테스트
    factors.py          알려진 팩터 라이브러리, 횡단면 직교화, 중복도
  backtest/engine.py    벡터화 백테스터 (체결 지연, 거래비용, 리밸런싱 주기, 리스크 오버레이 훅)
  backtest/backtrader_adapter.py  Backtrader 교차 검증
  library.py            FactorLibrary: 모든 팩터·지표·판정 기록(JSON), 아이디어 에이전트 피드백
  data/synthetic.py     합성 가격 + 합성 문서(문서 톤이 이후 수익을 예측하도록 심어둠)
scripts/demo_pipeline.py  전체 흐름 데모
```

파이프라인 단계 (원문의 SAF: Search → Analyze → Finalize):

1. **아이디어**: 검색된 문서 근거와 지난 라운드 결과(FactorLibrary)를 보고 팩터 제안
2. **구현**: `compute_alpha(df)` 코드 작성. 실패하면 오류 내용을 주고 수정 요청(기본 2회)
3. **안전성**: AST 정책(임포트·파일·네트워크·dunder 금지) → 별도 `python -I` 프로세스에서 실행(축소된 builtins, 타임아웃, POSIX 메모리 제한). 결과는 pickle이 아닌 float 배열로만 받습니다.
4. **예측 편향**: `shift(-k)`, `bfill`, `rolling(center=True)` 등 정적 탐지 + **절단 테스트**(기준일 이후 데이터를 지워도 그 이전 값이 같아야 함). 전체 표본 평균으로 정규화하는 것처럼 정적으로 안 보이는 누수도 잡습니다.
5. **평가**: 기간을 IS/OOS로 나누고 부호는 IS에서만 정합니다. OOS IC, t-stat, 거래비용 차감 백테스트.
6. **새로움**: 모멘텀·단기반전·저변동성·규모·유동성 대비 직교화 후 잔차 IC, 이미 채택된 팩터와의 상관.
7. **판정**: 통계 게이트와 평가 에이전트 검토를 모두 통과해야 채택됩니다.

```bash
python scripts/demo_pipeline.py                  # 오프라인(MockLLM)
python scripts/demo_pipeline.py --llm anthropic  # 실제 Claude 호출 (비용 발생)
```

오프라인 데모 결과(합성 데이터 50종목 × 300일): 텍스트 기반 `sentiment_drift`만 채택(OOS IC 0.139, 비용 차감 샤프 5.6).
단기반전 계열은 알려진 팩터와 상관 0.9 이상이라 기각, `vwap_gap`은 회전율 1.6으로 비용 차감 후 손실이라 기각됩니다.
합성 데이터의 신호는 일부러 강하게 심은 것이라, 이 수치는 파이프라인이 제대로 동작하는지 확인하는 용도입니다.

참고: 문서 점수 매기기는 문서마다 LLM 호출 1회입니다. 결과는 `doc_scores.jsonl`에 캐시됩니다.

## 7-8주차: 검증 및 운영 환경 강화

```
alphaagent/
  validation/walkforward.py  워크포워드 검증, 알려진 팩터 스패닝 테스트(Newey-West), 시점 기준 반복 실행
  risk/overlay.py            리스크 오버레이: 종목당 최대 비중, 섹터 중립, 총·순노출 한도, 섹터 총노출 한도
  monitoring/decay.py        롤링 IC, 예측 기간별 IC 감소 곡선, 반감기, 팩터 상태(healthy/decaying/dead)
  monitoring/dashboard.py    자체 완결형 HTML 대시보드(인라인 SVG, 다크 모드)
  data/market.py             실제 가격 로더 (CSV, yfinance), 섹터 매핑
scripts/historical_validation.py  12개월 이상 시점 기준 반복 실행 + 대시보드 생성
```

- **12개월 이상 과거 실행**: `historical_reruns`가 매월 기준일마다 그 날까지의 데이터만으로 파이프라인 전체를 다시 돌리고,
  채택된 팩터를 **다음 달** 데이터로 채점합니다. 팩터 하나가 아니라 연구 과정 자체를 백테스트하는 셈입니다.
- **알려진 팩터와 비교**: 기존 직교화(팩터 값 기준)에 더해, 팩터 수익률을 알려진 팩터 포트폴리오 수익률에 회귀한
  절편의 Newey-West t값(스패닝 테스트)을 채택 기준에 넣었습니다. 워크포워드 구간의 절반 이상에서 OOS IC가 양수여야 합니다.
- **리스크 오버레이**: `AlphaGenerationPipeline(weight_fn=RiskOverlay(...))`로 백테스트에 바로 적용됩니다.
  상한을 넘는 쪽만 줄이는 방식이라 섹터 중립과 종목 상한이 동시에 정확히 지켜집니다.

```bash
python scripts/historical_validation.py                  # 합성 데이터 440일, 매월 재실행
python scripts/historical_validation.py --prices px.csv  # 실제 가격 CSV (date,ticker,open,high,low,close,volume)
```

오프라인 실행 결과(합성 데이터 50종목, 2024-01 ~ 2025-09, 첫 실행 전 189일 이력, 21거래일마다 11회 재실행):

| 기준일 | 새로 채택 | 운용 중 팩터 수 | 다음 달 IC (sentiment) | 다음 달 IC (guidance) |
|---|---|---|---|---|
| 2024-09 ~ 2024-11 | 없음 | 0 | | |
| 2024-12-18 | sentiment_drift | 1 | 0.074 | |
| 2025-01 ~ 2025-03 | | 1 | 0.006 / 0.129 / 0.036 | |
| 2025-04-15 | guidance_surprise_unpriced | 2 | 0.082 | 0.077 |
| 2025-05 ~ 2025-07 | | 2 | 0.139 / -0.011 / 0.161 | 0.141 / 0.008 / 0.073 |

운용 중인 팩터-월 12개의 다음 달 평균 IC는 0.076이고, 12개 중 11개 달이 양수였습니다. 초기 3개월은 표본이 짧아 아무것도 채택하지 않았습니다.
대시보드는 `guidance_surprise_unpriced`를 최근 60일 IC가 전체 평균의 1/3로 떨어져 `decaying`으로 표시합니다.
실행 시간은 약 7분입니다(대부분 샌드박스 프로세스 기동 시간).

## 테스트

```bash
python -m pytest -q
```

`data/samples/`의 데이터는 가상 종목(ACME, GLBX, INIT)으로 만든 **합성 데이터**입니다.
