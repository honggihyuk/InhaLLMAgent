"""ResearchTeam: the full multi-agent hierarchy wired together.

Manager --TASK--> specialist analysts --(AlphaGenerationPipeline)--> RESULT --> Manager.synthesize
        --> PortfolioAgent (blend + size) --> RiskAgent (monitor, scale/halt) --> EnsembleDecision (votes)
All messages go through one MessageBus and every step is recorded in SharedMemory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from alphaagent.agents.ensemble import AnalystAgent, EnsembleDecision, signal_votes
from alphaagent.agents.manager import ANALYST_SPECIALTIES, ManagerAgent, SubTask
from alphaagent.agents.memory import SharedMemory
from alphaagent.agents.portfolio import PortfolioAgent
from alphaagent.agents.protocol import AgentMessage, MessageBus, MessageType
from alphaagent.agents.risk_agent import RiskAgent, RiskMonitorLimits, RiskReport
from alphaagent.agents.types import AlphaFactor
from alphaagent.backtest.metrics import forward_returns
from alphaagent.library import FactorLibrary
from alphaagent.llm.base import LLMClient
from alphaagent.pipeline import AlphaGenerationPipeline, PipelineConfig
from alphaagent.risk import RiskLimits, RiskOverlay

log = logging.getLogger(__name__)


class PipelineAnalyst(AnalystAgent):
    """Analyst that answers a TASK by running the alpha pipeline on the assigned theme."""

    def __init__(self, *args, pipeline: AlphaGenerationPipeline, data: Dict, **kw):
        super().__init__(*args, **kw)
        self.pipeline = pipeline
        self.data = data  # {"prices": ..., "text": ..., "as_of": ...}

    def handle(self, message: AgentMessage) -> AgentMessage:
        task = message.payload.get("task", {})
        accepted = self.pipeline.run_pipeline(
            task.get("theme", message.content),
            self.data["prices"],
            self.data.get("text"),
            tickers=task.get("tickers") or None,
            as_of=self.data.get("as_of"),
        )
        result = {
            "task_id": task.get("task_id"),
            "theme": task.get("theme"),
            "accepted": [
                {"name": f.name, "oos_ic": f.metrics.get("oos_ic"), "oos_icir": f.metrics.get("oos_icir"),
                 "residual_ic": f.metrics.get("residual_oos_ic"), "sharpe_net": f.metrics.get("bt_sharpe")}
                for f in accepted
            ],
        }
        reply = message.reply(self.name, MessageType.RESULT, f"{len(accepted)} factors accepted for {task.get('theme')}",
                              {"result": result, "role": self.role.value})
        self.bus.send(reply)
        return reply


@dataclass
class TeamResult:
    plan: List[SubTask]
    results: List[Dict]
    strategy: Dict
    factors: List[AlphaFactor]
    factor_weights: Dict[str, float]
    weights: pd.DataFrame  # final position weights after portfolio + risk scaling
    pnl: pd.Series
    risk_report: Optional[RiskReport]
    risk_history: pd.DataFrame
    decisions: Optional[pd.Series] = None  # ensemble buy/hold/sell for the latest date
    ensemble_weights: Dict[str, float] = field(default_factory=dict)


class ResearchTeam:
    def __init__(
        self,
        llm: LLMClient,
        retriever=None,
        sectors: Optional[Dict[str, str]] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        library: Optional[FactorLibrary] = None,
        risk_limits: Optional[RiskLimits] = None,
        monitor_limits: Optional[RiskMonitorLimits] = None,
        memory: Optional[SharedMemory] = None,
        target_vol: float = 0.10,
    ) -> None:
        self.llm = llm
        self.memory = memory if memory is not None else SharedMemory()
        self.bus = MessageBus(self.memory)
        self.sectors = sectors
        self.overlay = RiskOverlay(risk_limits or RiskLimits(), sectors)
        self.pipeline = AlphaGenerationPipeline(
            llm, retriever=retriever, memory=self.memory, library=library, config=pipeline_config, weight_fn=self.overlay
        )
        self.manager = ManagerAgent(llm, retriever=retriever, memory=self.memory, bus=self.bus)
        self.portfolio = PortfolioAgent(llm, memory=self.memory, bus=self.bus)
        lim = monitor_limits or RiskMonitorLimits(max_position=self.overlay.limits.max_position * 2)
        self.risk = RiskAgent(llm, memory=self.memory, bus=self.bus, limits=lim, sectors=sectors)
        self.retriever = retriever
        self.target_vol = target_vol
        self._data: Dict = {}
        self.analysts = {
            s: PipelineAnalyst(llm, retriever=retriever, memory=self.memory, bus=self.bus, specialty=s,
                               pipeline=self.pipeline, data=self._data)
            for s in ANALYST_SPECIALTIES
        }

    def research(
        self,
        goal: str,
        prices: pd.DataFrame,
        text_panel: Optional[pd.DataFrame] = None,
        as_of: Optional[str] = None,
        max_tasks: int = 3,
        review_portfolio: bool = True,
    ) -> TeamResult:
        self._data.update(prices=prices, text=text_panel, as_of=as_of)
        plan = self.manager.decompose(goal, max_tasks=max_tasks, as_of=as_of)
        self.manager.dispatch(plan, {s: a.name for s, a in self.analysts.items()})
        for a in self.analysts.values():
            a.process_inbox()
        results = [m.payload["result"] for m in self.bus.receive(self.manager.name, MessageType.RESULT)]
        strategy = self.manager.synthesize(goal, results)

        accepted = self.pipeline.library.accepted()
        deploy = set(strategy.get("deploy") or [f.name for f in accepted])
        factors = [f for f in accepted if f.name in deploy] or accepted
        panel, _ = self.pipeline.prepare_panel(prices, text_panel, as_of)
        if not factors:
            empty = pd.DataFrame()
            return TeamResult(plan, results, strategy, [], {}, empty, pd.Series(dtype=float), None, empty)

        values = {f.name: self.pipeline.factor_values(f, prices, text_panel, as_of) for f in factors}
        fw = self.portfolio.factor_weights(factors, emphasis=strategy.get("emphasis"))
        composite = self.portfolio.combine(values, fw)
        weights = self.portfolio.size_positions(composite, panel, overlay=self.overlay, target_vol=self.target_vol)

        # continuous risk monitoring through the history of the book (weekly checks)
        close = panel["close"].unstack("ticker").reindex(columns=weights.columns)
        pnl = (weights.shift(1) * close.pct_change().reindex(weights.index)).sum(axis=1, min_count=1).fillna(0.0)
        risk_history = self.risk.monitor(weights, pnl, every=5)
        last = weights.index[-1]
        report = self.risk.assess(last, weights.loc[last], pnl)
        scale = report.scale
        if review_portfolio and scale > 0:
            review = self.portfolio.review(weights, fw, report.metrics)
            scale *= review["scale"]
            strategy["portfolio_review"] = review
        weights.loc[last] = weights.loc[last] * scale

        # ensemble decision over factor voters, with accuracy-learned weights
        votes = {name: signal_votes(v) for name, v in values.items()}
        ens = EnsembleDecision(list(votes))
        fwd = forward_returns(panel["close"], 1)
        recent = panel.index.get_level_values("date").unique()[-60:]
        recent_votes = {k: v[v.index.get_level_values("date").isin(recent)] for k, v in votes.items()}
        decisions = ens.run(recent_votes, fwd, horizon=1)
        latest = decisions[decisions.index.get_level_values("date") == recent[-1]]

        return TeamResult(plan, results, strategy, factors, fw, weights, pnl, report, risk_history, latest, dict(ens.weights))
