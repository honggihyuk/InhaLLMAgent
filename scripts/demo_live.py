"""Week 11-12 demo: streaming ingestion + paper trading with circuit breakers, human override, and audit.

Runs entirely in one process with the in-memory broker (set ALPHA_BROKER_URL=kafka://... for Kafka).

    python scripts/demo_live.py
"""

from pathlib import Path

from alphaagent.agents import AlphaFactor, PortfolioAgent, RiskAgent, RiskMonitorLimits
from alphaagent.config import get_settings
from alphaagent.data import create_synthetic_dataset, synthetic_sectors
from alphaagent.embeddings import HashingEmbedder
from alphaagent.features import TextSignalExtractor, build_text_panel
from alphaagent.ingestion import IngestionPipeline
from alphaagent.llm import MockLLM
from alphaagent.llm.mock_handlers import _BY_NAME
from alphaagent.ops import (
    AuditedLLM,
    AuditTrail,
    CircuitBreaker,
    ControlCenter,
    GuardLimits,
    LiveTradingLoop,
    ResilientLLM,
    TradingGuard,
    regulatory_report,
    report_markdown,
)
from alphaagent.pipeline import AlphaGenerationPipeline
from alphaagent.risk import RiskLimits, RiskOverlay
from alphaagent.streaming import DocumentProducer, IngestionConsumer, build_broker
from alphaagent.vectorstore import FaissVectorStore


def main() -> None:
    out = Path("data/runs/live")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("audit.jsonl", "control.json"):
        (out / f).unlink(missing_ok=True)
    audit = AuditTrail(str(out / "audit.jsonl"))
    llm = AuditedLLM(ResilientLLM(MockLLM(), CircuitBreaker("llm", audit=audit)), audit)

    # 1) streaming ingestion: pollers publish documents, the consumer indexes + scores them
    prices, docs = create_synthetic_dataset(n_tickers=30, n_days=260)
    broker = build_broker(get_settings().broker_url)
    DocumentProducer(broker).send_all(docs)
    emb = HashingEmbedder()
    store = FaissVectorStore(emb.dim)
    scorer = TextSignalExtractor(llm)
    consumer = IngestionConsumer(broker, IngestionPipeline(emb, store), scorer)
    while consumer.run_once(max_records=100):
        pass
    print("consumer:", consumer.stats)
    text = build_text_panel(list(scorer.cache.values()), prices.index)

    # 2) paper trading with production controls
    control = ControlCenter(str(out / "control.json"), audit, approval_threshold=0.06)
    guard = TradingGuard(GuardLimits(max_turnover=1.0), control, audit)
    sectors = synthetic_sectors(prices.index.get_level_values("ticker").unique())
    factor = AlphaFactor("sentiment_drift", _BY_NAME["sentiment_drift"]["code"], metrics={"sign": 1.0, "oos_icir": 0.5})
    audit.log("factor_status", "pipeline", {"name": factor.name, "status": "accepted", "approved_by": "carol"})
    loop = LiveTradingLoop(
        AlphaGenerationPipeline(llm), [factor], PortfolioAgent(llm), RiskAgent(llm, limits=RiskMonitorLimits(max_position=0.1)),
        guard, control, audit, overlay=RiskOverlay(RiskLimits(max_position=0.05), sectors),
    )
    dates = prices.index.get_level_values("date").unique()
    for i, d in enumerate(dates[200:215]):
        res = loop.step(prices, text, d)
        for ap in control.pending():  # an operator approves queued trades
            control.decide(ap.approval_id, True, "carol", "reviewed")
        if i == 10:
            control.engage_kill_switch("alice", "end-of-demo drill")
        print(f"{d.date()} {res.status:<9} pnl {res.pnl:+.4%} gross {res.positions.abs().sum():.2f} "
              f"pending {len(res.pending_approvals)} {'; '.join(res.reasons)[:60]}")

    report = regulatory_report(audit)
    (out / "oversight_report.md").write_text(report_markdown(report), encoding="utf-8")
    print("\naudit trail:", audit.verify())
    print(f"report: {out / 'oversight_report.md'}")


if __name__ == "__main__":
    main()
