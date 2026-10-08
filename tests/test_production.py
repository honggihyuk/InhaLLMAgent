import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from alphaagent.agents import AgentMessage, MessageType, PortfolioAgent, RiskAgent, RiskMonitorLimits
from alphaagent.data import create_synthetic_dataset, synthetic_sectors
from alphaagent.documents import Document
from alphaagent.embeddings import HashingEmbedder
from alphaagent.features import TextSignalExtractor, build_text_panel
from alphaagent.ingestion import IngestionPipeline
from alphaagent.llm import MockLLM, ScriptedLLM
from alphaagent.llm.base import LLMError
from alphaagent.ops import (
    AuditedLLM,
    AuditTrail,
    CircuitBreaker,
    CircuitOpenError,
    ControlCenter,
    GuardLimits,
    LiveTradingLoop,
    ResilientLLM,
    TradingGuard,
    regulatory_report,
    report_markdown,
)
from alphaagent.ops.api import ops_router
from alphaagent.pipeline import AlphaGenerationPipeline, PipelineConfig
from alphaagent.risk import RiskLimits, RiskOverlay
from alphaagent.streaming import (
    TOPIC_DOCS_INDEXED,
    TOPIC_SIGNALS,
    BrokerMessageBus,
    DocumentProducer,
    InMemoryBroker,
    IngestionConsumer,
    build_broker,
    poll_sources,
)
from alphaagent.vectorstore import FaissVectorStore


def _docs(n=5):
    return [
        Document(ticker=f"T{i}", doc_type="news", title=f"t{i}", published_at="2026-01-0" + str(i + 1),
                 text="Record revenue and raising guidance" if i % 2 == 0 else "Revenue declined and guidance cut")
        for i in range(n)
    ]


# ---- streaming ------------------------------------------------------------------------
def test_kafka_style_ingestion_is_idempotent_and_at_least_once():
    broker = InMemoryBroker()
    emb = HashingEmbedder(128)
    store = FaissVectorStore(emb.dim)
    saves = []
    consumer = IngestionConsumer(broker, IngestionPipeline(emb, store), TextSignalExtractor(MockLLM()),
                                 on_batch=lambda: saves.append(len(store)))
    producer = DocumentProducer(broker)
    docs = _docs(5)
    producer.send_all(docs)
    producer.send_all(docs[:2])  # duplicate delivery from a re-polled source
    assert consumer.run_once() == 7
    assert consumer.stats.indexed == 5 and consumer.stats.duplicates == 2 and consumer.stats.scored == 5
    assert len(broker.logs[TOPIC_DOCS_INDEXED]) == 5
    sig = {r.value["ticker"]: r.value["sentiment"] for r in broker.logs[TOPIC_SIGNALS]}
    assert sig["T0"] > 0 > sig["T1"]
    assert consumer.run_once() == 0 and saves == [len(store)]

    # crash after polling but before commit -> records are redelivered, nothing is double-indexed
    producer.send(Document(ticker="T9", doc_type="news", title="x", text="new", published_at="2026-02-01"))
    broker.poll(["docs.raw"], "ingestion")
    broker.rewind("ingestion")
    assert consumer.run_once() == 1 and store.has_document(_docs(1)[0].doc_id)
    assert consumer.stats.indexed == 6


def test_poll_sources_survives_failing_source():
    class Good:
        name = "good"

        def load(self):
            return iter(_docs(2))

    class Bad:
        name = "bad"

        def load(self):
            raise ConnectionError("feed down")

    broker = InMemoryBroker()
    n = poll_sources([Bad(), Good()], DocumentProducer(broker), interval=0, iterations=2, sleep=lambda s: None)
    assert n == 4 and len(broker.logs["docs.raw"]) == 4


def test_broker_message_bus_connects_processes():
    broker = build_broker("memory://")
    bus_a, bus_b = BrokerMessageBus(broker), BrokerMessageBus(broker)
    bus_a.register("manager")
    bus_b.register("analyst")
    bus_a.send(AgentMessage("manager", "analyst", MessageType.TASK, "look at guidance"))
    assert bus_b.pull_remote("pod-b") == 1
    (msg,) = bus_b.receive("analyst")
    assert msg.content == "look at guidance" and msg.type == MessageType.TASK
    assert bus_a.pull_remote("pod-a") == 0  # own message is not redelivered to the sender's pod


# ---- audit trail -------------------------------------------------------------------------
def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditTrail(str(path))
    for i in range(5):
        audit.log("order", "loop", {"i": i})
    assert audit.verify() == {"ok": True, "records": 5, "head": audit._last_hash}
    assert AuditTrail(str(path)).log("order", "loop", {"i": 5})["seq"] == 6  # resumes the chain after restart

    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["details"]["i"] = 999
    lines[2] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert AuditTrail(str(path)).verify() == {"ok": False, "records": 2, "broken_at": 3}


def test_audited_llm_and_regulatory_report(tmp_path):
    audit = AuditTrail(str(tmp_path / "a.jsonl"))
    llm = AuditedLLM(MockLLM(), audit)
    llm.generate("TASK_KIND: ideation\nGenerate 1 novel alpha")
    audit.log("factor_status", "pipeline", {"name": "sentiment_drift", "status": "accepted", "code_sha256": "ab" * 32})
    audit.log("kill_switch", "alice", {"engaged": True, "reason": "test"})
    rep = regulatory_report(audit)
    assert rep["integrity"]["ok"] and rep["llm_usage"]["calls"] == 1 and rep["llm_usage"]["by_task"] == {"ideation": 1}
    assert rep["algorithm_inventory"]["sentiment_drift"]["status"] == "accepted"
    md = report_markdown(rep)
    assert "sentiment_drift" in md and "Kill-switch events: 1" in md and "alice" in md
    rec = next(r for r in audit.records() if r["event"] == "llm_call")
    assert "prompt" not in rec["details"] and len(rec["details"]["prompt_sha256"]) == 64


# ---- circuit breakers ----------------------------------------------------------------------
def test_circuit_breaker_transitions(tmp_path):
    now = [0.0]
    audit = AuditTrail(str(tmp_path / "a.jsonl"))
    cb = CircuitBreaker("llm", failure_threshold=2, reset_timeout=10, audit=audit, clock=lambda: now[0])

    def boom():
        raise TimeoutError("api timeout")

    for _ in range(2):
        with pytest.raises(TimeoutError):
            cb.call(boom)
    assert cb.state == "open"
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)
    now[0] = 11
    assert cb.call(lambda: 1) == 1 and cb.state == "closed"
    states = [r["details"]["state"] for r in audit.records()]
    assert states == ["open", "half_open", "closed"]


def test_resilient_llm_falls_back():
    def flaky(p, s):
        raise LLMError("overloaded")

    llm = ResilientLLM(ScriptedLLM(flaky), CircuitBreaker("llm", failure_threshold=1, reset_timeout=999),
                       fallback=ScriptedLLM(lambda p, s: "local model answer"))
    assert llm.generate("q") == "local model answer"
    assert llm.breaker.state == "open"
    assert llm.generate("q2") == "local model answer"  # primary not even tried while open


def test_trading_guard(tmp_path):
    audit = AuditTrail(str(tmp_path / "a.jsonl"))
    control = ControlCenter(str(tmp_path / "c.json"), audit)
    guard = TradingGuard(GuardLimits(max_daily_loss=0.02, max_drawdown=0.05, max_turnover=0.5), control, audit)
    t = pd.Series({"A": 0.5, "B": -0.5})
    cur = pd.Series(dtype=float)
    calm = pd.Series([0.001, -0.002, 0.001])
    day = pd.Timestamp("2026-03-02")
    d = guard.check(t, cur, calm, day, day)
    assert d.allowed and d.scale == pytest.approx(0.5)  # turnover 1.0 capped at 0.5
    assert not guard.check(t, cur, calm, day, day - pd.Timedelta(days=7)).allowed  # stale data trips
    assert "latched" in guard.check(t, cur, calm, day, day).reasons[0]  # stays tripped
    guard.reset("bob", "feed restored")
    assert guard.check(t, cur, calm, day, day).allowed
    assert not guard.check(t, cur, pd.Series([0.0, -0.03]), day, day).allowed  # daily loss
    guard.reset("bob", "reviewed")
    control.engage_kill_switch("alice", "manual stop")
    d = guard.check(t, cur, calm, day, day)
    assert not d.allowed and "alice" in d.reasons[0]


def test_control_center_persistence_and_approvals(tmp_path):
    path = str(tmp_path / "control.json")
    c = ControlCenter(path, approval_threshold=0.02)
    with pytest.raises(ValueError):
        c.pause_factor("x", operator="")
    c.pause_factor("sentiment_drift", "alice", "investigating decay")
    c.set_book_scale(3.0, "alice")  # cannot size up through the override
    big = c.needs_approval(pd.Series({"A": 0.05, "B": 0.01}), pd.Series({"A": 0.0, "B": 0.0}))
    assert list(big.index) == ["A"]
    ap = c.request_approval("big trade", {"weights": {"A": 0.05}})
    reloaded = ControlCenter(path)
    assert reloaded.state.paused_factors == ["sentiment_drift"] and reloaded.state.book_scale == 1.0
    assert [a.approval_id for a in reloaded.pending()] == [ap.approval_id]
    reloaded.decide(ap.approval_id, True, "carol", "ok")
    with pytest.raises(ValueError):
        reloaded.decide(ap.approval_id, False, "dave")


# ---- live loop + ops API ------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_setup():
    prices, docs = create_synthetic_dataset(n_tickers=20, n_days=200, seed=7)
    text = build_text_panel(TextSignalExtractor(MockLLM()).score_documents(docs), prices.index)
    pipe = AlphaGenerationPipeline(MockLLM(), config=PipelineConfig(num_ideas=1))
    from alphaagent.agents import AlphaFactor
    from alphaagent.llm.mock_handlers import _BY_NAME

    factor = AlphaFactor("sentiment_drift", _BY_NAME["sentiment_drift"]["code"], metrics={"sign": 1.0, "oos_icir": 0.5})
    factor.status = "accepted"
    return prices, text, pipe, factor


def test_live_loop_paper_trading_with_controls(tmp_path, live_setup):
    prices, text, pipe, factor = live_setup
    audit = AuditTrail(str(tmp_path / "audit.jsonl"))
    control = ControlCenter(str(tmp_path / "control.json"), audit, approval_threshold=0.08)
    sectors = synthetic_sectors(prices.index.get_level_values("ticker").unique(), 3)
    loop = LiveTradingLoop(
        pipe, [factor], PortfolioAgent(MockLLM()), RiskAgent(MockLLM(), limits=RiskMonitorLimits(max_position=0.2)),
        TradingGuard(GuardLimits(max_turnover=2.0, max_drawdown=0.5, max_daily_loss=0.5), control, audit),
        control, audit, overlay=RiskOverlay(RiskLimits(max_position=0.1), sectors),
    )
    dates = prices.index.get_level_values("date").unique()
    first = loop.step(prices, text, dates[150])
    assert first.status == "traded" and first.positions.abs().max() <= 0.08 + 1e-9  # big moves wait for approval
    assert first.pending_approvals
    control.decide(first.pending_approvals[0], True, "carol")
    second = loop.step(prices, text, dates[151])
    assert second.status == "traded" and loop.positions.abs().max() > 0.08 - 1e-9  # approved trades applied

    control.pause_factor("sentiment_drift", "alice")
    assert loop.step(prices, text, dates[152]).status == "no_factors"
    control.resume_factor("sentiment_drift", "alice")
    control.engage_kill_switch("alice", "drill")
    flat = loop.step(prices, text, dates[153])
    assert flat.status == "flattened" and (flat.positions == 0).all()

    rep = regulatory_report(audit)
    assert rep["integrity"]["ok"] and rep["orders"] >= 3 and len(rep["kill_switch_events"]) == 1
    assert any(r["details"].get("status") == "approved" for r in rep["human_overrides"])


def test_ops_api(tmp_path):
    audit = AuditTrail(str(tmp_path / "audit.jsonl"))
    control = ControlCenter(str(tmp_path / "control.json"), audit)
    guard = TradingGuard(control=control, audit=audit)
    app = FastAPI()
    app.include_router(ops_router(control, audit, guard))
    c = TestClient(app)
    assert c.post("/ops/kill-switch/engage", json={"reason": "x"}).status_code == 401  # operator required
    h = {"X-Operator": "alice"}
    assert c.post("/ops/kill-switch/engage", json={"reason": "drill"}, headers=h).json()["kill_switch"] is True
    assert c.post("/ops/factors/f1/pause", json={}, headers=h).json()["paused_factors"] == ["f1"]
    assert c.post("/ops/book-scale", json={"scale": 0.5}, headers=h).json()["book_scale"] == 0.5
    ap = control.request_approval("trade", {"weights": {"A": 0.1}})
    assert [a["approval_id"] for a in c.get("/ops/approvals").json()] == [ap.approval_id]
    assert c.post(f"/ops/approvals/{ap.approval_id}", json={"approve": False}, headers=h).json()["status"] == "rejected"
    assert c.post(f"/ops/approvals/{ap.approval_id}", json={"approve": True}, headers=h).status_code == 409
    assert c.get("/ops/audit/verify").json()["ok"] is True
    assert "Kill-switch events: 1" in c.get("/ops/report.md").text
