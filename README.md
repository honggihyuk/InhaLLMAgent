# InhaLLMAgent — RAG 기반 LLM 알파 마이닝 에이전트

비정형 금융 텍스트(실적 발표 녹취록, SEC 공시, 뉴스)에서 알파 팩터를 찾는 RAG + 멀티 에이전트 시스템입니다.
"How Quants Use LLM Agents To Mine Alpha From Unstructured Data (The Complete RAG Framework)" 글의
아키텍처와 12주 로드맵을 순서대로 구현합니다.

## 로드맵 진행 상황

| 주차 | 내용 | 상태 |
|---|---|---|
| 1-2 | 벡터 DB(FAISS), 문서 수집, BGE-M3/FinBERT 임베딩, 종목 필터 검색 API | ✅ |
| 3-4 | LLM 백엔드, CoT QuantAgent, 아이디어·구현·평가 에이전트, 공유 메모리 | ✅ |
| 5-6 | AlphaGenerationPipeline, 코드 샌드박스, 백테스트, 예측 편향 탐지·직교화 | ⏳ |
| 7-8 | 12개월+ 검증, 알려진 팩터 비교, 리스크 오버레이, 모니터링 대시보드 | ⏳ |
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

## 테스트

```bash
python -m pytest -q
```

`data/samples/`의 데이터는 가상 종목(ACME, GLBX, INIT)으로 만든 **합성 데이터**입니다.
