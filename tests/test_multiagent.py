import numpy as np
import pandas as pd
import pytest

from alphaagent.agents import (
    BUY,
    HOLD,
    SELL,
    AgentRole,
    AlphaFactor,
    AnalystAgent,
    EnsembleDecision,
    ManagerAgent,
    MessageBus,
    MessageType,
    PortfolioAgent,
    RiskAgent,
    RiskMonitorLimits,
    SharedMemory,
    llm_votes,
    signal_votes,
)
from alphaagent.backtest.metrics import forward_returns
from alphaagent.data import create_synthetic_dataset, create_synthetic_market_data, synthetic_sectors
from alphaagent.features import TextSignalExtractor, build_text_panel
from alphaagent.llm import MockLLM, ScriptedLLM
from alphaagent.pipeline import PipelineConfig
from alphaagent.risk import RiskLimits, RiskOverlay
from alphaagent.team import ResearchTeam


def reversal(df):
    return -df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())


# ---- manager --------------------------------------------------------------------
def test_manager_decomposes_and_dispatches():
    bus = MessageBus(SharedMemory())
    mgr = ManagerAgent(MockLLM(), bus=bus)
    for n in ("sentiment_analyst", "fundamental_analyst"):
        bus.register(n)
    tasks = mgr.decompose("find alpha in earnings calls", max_tasks=2)
    assert [t.task_id for t in tasks] == ["t1", "t2"] and tasks[0].specialty == "sentiment"
    mgr.dispatch(tasks, {"sentiment": "sentiment_analyst", "fundamental": "fundamental_analyst"})
    (msg,) = bus.receive("fundamental_analyst")
    assert msg.type == MessageType.TASK and msg.payload["task"]["task_id"] == "t2"


def test_manager_falls_back_on_malformed_plan():
    mgr = ManagerAgent(ScriptedLLM(["no json here [Confidence: 0.2]"]))
    (task,) = mgr.decompose("goal X")
    assert task.theme == "goal X"


# ---- portfolio ---------------------------------------------------------------------
def test_portfolio_weights_blend_and_vol_target():
    df = create_synthetic_market_data(30, 200, seed=1)
    f1 = AlphaFactor("a", "", metrics={"oos_icir": 0.3})
    f2 = AlphaFactor("b", "", metrics={"oos_icir": 0.1})
    f3 = AlphaFactor("c", "", metrics={"oos_icir": -0.2})
    pa = PortfolioAgent(MockLLM())
    w = pa.factor_weights([f1, f2, f3])
    assert w == pytest.approx({"a": 0.75, "b": 0.25, "c": 0.0})
    assert pa.factor_weights([f1, f2], emphasis={"b": 2.0})["b"] == pytest.approx(0.4)
    comp = pa.combine({"a": reversal(df), "b": df["returns"]}, w)
    pos = pa.size_positions(comp, df, target_vol=0.10)
    close = df["close"].unstack("ticker")[pos.columns]
    pnl = (pos.shift(1) * close.pct_change().reindex(pos.index)).sum(axis=1).iloc[80:]
    assert 0.05 < pnl.std() * np.sqrt(252) < 0.2  # realised vol near the 10% target
    assert pos.sum(axis=1).abs().max() < 1e-9  # dollar neutral


def test_portfolio_review_can_only_reduce_risk():
    df = create_synthetic_market_data(10, 30)
    wts = pd.DataFrame([[0.5, -0.5]], columns=["X", "Y"])
    greedy = PortfolioAgent(ScriptedLLM(['```json\n{"approve": true, "scale": 3.0}\n```']))
    assert greedy.review(wts, {}, {})["scale"] == 1.0
    veto = PortfolioAgent(ScriptedLLM(['```json\n{"approve": false, "scale": 0.8}\n```']))
    assert veto.review(wts, {}, {})["scale"] == 0.0


# ---- risk agent ------------------------------------------------------------------------
def test_risk_agent_alerts_and_scales():
    bus = MessageBus(SharedMemory())
    bus.register("portfolio")
    agent = RiskAgent(MockLLM(), bus=bus, limits=RiskMonitorLimits(max_vol=0.15, max_drawdown=0.05, halt_drawdown=0.2))
    w = pd.Series({"A": 0.03, "B": -0.03})
    calm = pd.Series(np.random.default_rng(0).normal(0, 0.002, 100))
    ok = agent.assess("2026-01-02", w, calm)
    assert ok.action == "none" and ok.scale == 1.0 and bus.receive("portfolio") == []

    wild = pd.Series(np.r_[np.random.default_rng(1).normal(0, 0.003, 80), np.full(20, -0.004)])
    bad = agent.assess("2026-02-02", w, wild)
    assert bad.action == "reduce" and bad.scale <= 0.5 and any("drawdown" in b for b in bad.breaches)
    (alert,) = bus.receive("portfolio")
    assert alert.type == MessageType.ALERT

    crash = pd.Series(np.r_[np.zeros(50), np.full(10, -0.03)])
    assert agent.assess("2026-03-02", w, crash).action == "halt"
    assert "Limits breached" in agent.explain(bad)


def test_risk_monitor_runs_through_history():
    agent = RiskAgent(MockLLM())
    idx = pd.bdate_range("2026-01-01", periods=60)
    weights = pd.DataFrame({"A": 0.04, "B": -0.04}, index=idx)
    pnl = pd.Series(np.random.default_rng(2).normal(0, 0.003, 60), index=idx)
    hist = agent.monitor(weights, pnl, every=5)
    assert len(hist) == 12 and {"var95", "drawdown", "gross", "action"} <= set(hist.columns)


# ---- ensemble --------------------------------------------------------------------------------
def test_signal_votes_quantiles():
    df = create_synthetic_market_data(10, 5)
    v = signal_votes(df["close"], quantile=0.2)
    per_day = v.groupby(level="date").value_counts().unstack()
    assert (per_day[BUY] == 2).all() and (per_day[SELL] == 2).all() and (per_day[HOLD] == 6).all()


def test_ensemble_learns_to_trust_the_accurate_voter():
    df = create_synthetic_market_data(30, 160, seed=6, planted_signal="reversal", signal_strength=0.3)
    fwd = forward_returns(df["close"], 1)
    good = signal_votes(reversal(df))
    rng = np.random.default_rng(0)
    bad = pd.Series(rng.choice([BUY, HOLD, SELL], size=len(df)), index=df.index)
    ens = EnsembleDecision(["good", "bad"], learning_rate=0.1)
    decisions = ens.run({"good": good, "bad": bad}, fwd, horizon=1)
    assert ens.weights["good"] > ens.weights["bad"]
    late = decisions.index.get_level_values("date") >= df.index.get_level_values("date").unique()[100]
    assert (decisions[late] == good[late]).mean() > 0.9  # the ensemble now follows the reliable voter


def test_ensemble_tie_resolves_to_hold():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-01-02"), "A")], names=["date", "ticker"])
    ens = EnsembleDecision(["x", "y"])
    out = ens.decide({"x": pd.Series([BUY], index=idx), "y": pd.Series([SELL], index=idx)})
    assert out.iloc[0] == HOLD


def test_llm_analyst_votes_use_point_in_time_context(loaded):
    retriever, _ = loaded
    analysts = [AnalystAgent(MockLLM(), retriever=retriever, specialty=s) for s in ("sentiment", "fundamental")]
    votes = llm_votes(analysts, ["ACME", "GLBX"], as_of="2026-02-28")
    assert set(votes) == {"sentiment_analyst", "fundamental_analyst"}
    s = votes["sentiment_analyst"]
    assert s.xs("ACME", level="ticker").iloc[0] == BUY and s.xs("GLBX", level="ticker").iloc[0] == SELL
    early = llm_votes(analysts[:1], ["ACME"], as_of="2025-12-31")["sentiment_analyst"]
    assert early.iloc[0] == HOLD  # nothing published yet -> no evidence -> hold


# ---- full team -------------------------------------------------------------------------------
def test_research_team_end_to_end():
    prices, docs = create_synthetic_dataset(n_tickers=30, n_days=260, seed=7)
    text = build_text_panel(TextSignalExtractor(MockLLM()).score_documents(docs), prices.index)
    sectors = synthetic_sectors(prices.index.get_level_values("ticker").unique(), 4)
    team = ResearchTeam(
        MockLLM(), sectors=sectors, risk_limits=RiskLimits(max_position=0.06),
        pipeline_config=PipelineConfig(num_ideas=2, lookahead_cutoffs=1),
    )
    res = team.research("mine alpha from earnings calls and news", prices, text, max_tasks=2)
    assert len(res.plan) == 2 and len(res.results) == 2
    assert res.factors and abs(sum(res.factor_weights.values()) - 1) < 1e-9
    last = res.weights.iloc[-1]
    assert abs(last.sum()) < 1e-6
    assert res.risk_report is not None and not res.risk_history.empty
    assert set(res.decisions.unique()) <= {BUY, HOLD, SELL} and len(res.ensemble_weights) == len(res.factors)
    kinds = {e.kind for e in team.memory.read()}
    assert {"plan", "idea", "evaluation", "strategy", "risk", "message"} <= kinds
    roles = {m.payload.get("role") for m in team.bus.history}
    assert {"manager", "analyst"} <= roles
