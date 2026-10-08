# 운영 가이드

## 설치

```bash
pip install -e ".[embeddings,llm,backtest,dev]"   # 기본
pip install -e ".[streaming,market]"              # Kafka, yfinance가 필요할 때
python -m pytest -q                               # 전체 테스트 (Windows 기준 약 5~11분)
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ALPHA_DATA_DIR` | `data` | 인덱스·라이브러리·감사 기록 위치 |
| `ALPHA_EMBEDDER` | `bge-m3` | `bge-m3`, `finbert`, `hashing`, 또는 sentence-transformers 모델 ID |
| `ALPHA_EMBED_DEVICE` | `cpu` | `cuda` 가능 |
| `ALPHA_EMBED_MAX_SEQ` | `1024` | 임베딩 최대 토큰 |
| `ALPHA_CHUNK_SIZE` / `ALPHA_CHUNK_OVERLAP` | `1200` / `200` | 문자 단위 |
| `ALPHA_LLM_PROVIDER` | `anthropic` | `anthropic`, `vllm`, `mock` |
| `ALPHA_LLM_MODEL` | `claude-opus-5-5` | |
| `ALPHA_BROKER_URL` | `memory://` | `kafka://host:9092` |
| `ANTHROPIC_API_KEY` | | Claude 사용 시 (또는 `ant auth login` 프로필) |
| `SEC_USER_AGENT` | | SEC 수집 시 필수, 예: `"Inha Research you@example.com"` |

## 자주 쓰는 명령

```bash
# 문서 수집과 검색
alphaagent ingest transcripts --path data/samples/transcripts
alphaagent ingest sec --tickers AAPL --forms 10-K 10-Q --limit 4
alphaagent ingest news --tickers AAPL MSFT
alphaagent search "guidance cut" --ticker AAPL --end-date 2025-12-31

# 연구
python scripts/demo_pipeline.py                     # 파이프라인 1회
python scripts/historical_validation.py             # 12개월+ 시점 기준 재실행 + 대시보드
python scripts/historical_validation.py --prices px.csv --llm anthropic
python scripts/demo_team.py                         # 멀티 에이전트 팀

# 서비스
alphaagent serve --with-ops                          # 검색 API + 운영자 API (:8000)
alphaagent consume --score                           # Kafka 수집 컨슈머
alphaagent poll --tickers AAPL MSFT NVDA --interval 900
python scripts/demo_live.py                         # 모의 운용 + 통제 + 감사 보고서
```

`--llm anthropic`은 실제 API를 호출하므로 비용이 듭니다. 문서 채점 결과는 `doc_scores.jsonl`에 캐시돼 같은 문서는 다시 호출하지 않습니다.

## 실제 가격 CSV 형식

```
date,ticker,open,high,low,close,volume[,vwap,cap]
2025-01-02,AAPL,...
```

`vwap`이 없으면 (고가+저가+종가)/3, `returns`는 종가 변화율로 채웁니다. `cap`이 없으면 규모 팩터가 빠집니다.

## 운영자 API

모든 변경 요청에 `X-Operator: 이름` 헤더가 필요하고, 감사 기록에 남습니다.

| 메서드 | 경로 | 용도 |
|---|---|---|
| GET | `/ops/status` | 킬 스위치, 일시정지 팩터, 배율, 대기 승인 수, 차단기 상태 |
| POST | `/ops/kill-switch/engage` · `/release` | 킬 스위치 (`{"reason": "..."}`) |
| POST | `/ops/factors/{name}/pause` · `/resume` | 팩터 제외/재개 |
| POST | `/ops/book-scale` | 전체 배율 0~1 (`{"scale": 0.5}`) |
| POST | `/ops/guard/reset` | 래치된 거래 차단기 해제 |
| GET | `/ops/approvals` | 승인 대기 목록 |
| POST | `/ops/approvals/{id}` | `{"approve": true, "note": "..."}` |
| GET | `/ops/audit/verify` | 감사 기록 무결성 |
| GET | `/ops/report` · `/ops/report.md` | 규제 보고서 (`?start=&end=`) |

## 상황별 대응

| 상황 | 확인 | 조치 |
|---|---|---|
| 손실이 급격히 커짐 | `/ops/status`, 대시보드 낙폭 | 킬 스위치 → 원인 확인 → 해제 |
| 거래가 멈춤 (`frozen`) | `trading_guard_tripped` 사유 | 원인(데이터 지연, 손실 한도 등) 해결 후 `/ops/guard/reset` |
| 특정 팩터 성과 악화 | 대시보드 상태 `decaying` / `dead` | `/ops/factors/{name}/pause`, 다음 연구 실행에서 재평가 |
| LLM 오류 증가 | 감사 기록의 `circuit_breaker` 이벤트 | 차단기가 자동으로 대체 모델 사용, 공급자 상태 확인 |
| 승인 대기 누적 | `/ops/approvals` | 검토 후 승인/거절. 승인 전까지 해당 종목은 기존 비중 유지 |
| 감사 기록 검증 실패 | `/ops/audit/verify`의 `broken_at` | 해당 레코드 이후 변경 경위 조사, 백업과 대조 |

## 배포

```bash
# 로컬
cp deploy/.env.example deploy/.env   # ANTHROPIC_API_KEY, SEC_USER_AGENT 작성 (파일은 직접 만들어 주세요)
docker compose -f deploy/docker-compose.yml up --build

# 쿠버네티스 (Strimzi 오퍼레이터 설치 후)
kubectl create namespace alphaagent
kubectl -n alphaagent create secret generic alphaagent-secrets \
  --from-literal=ANTHROPIC_API_KEY=... --from-literal=SEC_USER_AGENT="Org you@example.com"
kubectl apply -k deploy/k8s
```

- 인덱스 PVC는 ReadWriteMany(EFS, Filestore 등)가 필요합니다. 쓰는 쪽은 컨슈머 하나뿐이고 API는 읽기 전용으로 붙어 30초마다 변경을 반영합니다.
- 야간 연구 CronJob은 `/data/prices.csv`를 읽습니다. 가격 적재 작업은 별도로 준비해야 합니다.
- 이미지 이름 `ghcr.io/honggihyuk/inhallmagent`는 예시입니다. 실제 레지스트리에 맞게 `kustomization.yaml`의 `images`를 바꾸세요.
