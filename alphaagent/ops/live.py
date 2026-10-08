"""Paper-trading loop that ties production controls together (no broker connectivity).

Each ``step(as_of)``:
  1. realise P&L of the current book since the previous step
  2. recompute active (non-paused) factors in the sandbox on data up to ``as_of``
  3. portfolio agent blends + sizes; risk agent assesses; operator book scale applied
  4. TradingGuard: kill switch -> flatten; latched breaker -> freeze; turnover cap -> partial move
  5. trades above the approval threshold wait for a human; approved ones are applied on the next step
  6. every decision and order is written to the audit trail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from alphaagent.agents.portfolio import PortfolioAgent
from alphaagent.agents.risk_agent import RiskAgent
from alphaagent.agents.types import AlphaFactor
from alphaagent.ops.audit import ORDER, RISK_ACTION, AuditTrail
from alphaagent.ops.controls import ControlCenter, TradingGuard


@dataclass
class StepResult:
    as_of: pd.Timestamp
    status: str  # traded | frozen | flattened | no_factors
    positions: pd.Series
    target: pd.Series
    pnl: float
    reasons: List[str] = field(default_factory=list)
    pending_approvals: List[str] = field(default_factory=list)


class LiveTradingLoop:
    def __init__(
        self,
        pipeline,
        factors: List[AlphaFactor],
        portfolio: PortfolioAgent,
        risk: RiskAgent,
        guard: TradingGuard,
        control: ControlCenter,
        audit: AuditTrail,
        overlay=None,
        factor_weights: Optional[Dict[str, float]] = None,
        target_vol: float = 0.10,
    ) -> None:
        self.pipeline, self.factors, self.portfolio, self.risk = pipeline, factors, portfolio, risk
        self.guard, self.control, self.audit, self.overlay = guard, control, audit, overlay
        self.factor_weights = factor_weights
        self.target_vol = target_vol
        self.positions = pd.Series(dtype=float)
        self.pnl = pd.Series(dtype=float)
        self._last_close: Optional[pd.Series] = None

    def _realise(self, prices: pd.DataFrame, as_of: pd.Timestamp) -> float:
        close = prices["close"].xs(as_of, level="date")
        pnl = 0.0
        if self._last_close is not None and not self.positions.empty:
            rets = (close / self._last_close.reindex(close.index) - 1).reindex(self.positions.index).fillna(0.0)
            pnl = float((self.positions * rets).sum())
        self._last_close = close
        self.pnl.loc[as_of] = pnl
        return pnl

    def _apply_approved(self) -> None:
        for ap in list(self.control.state.approvals.values()):
            if ap.status == "approved" and not ap.payload.get("applied"):
                for t, w in ap.payload.get("weights", {}).items():
                    self.positions.loc[t] = float(w)
                ap.payload["applied"] = True
                self.audit.log(ORDER, ap.decided_by or "operator", {"approval_id": ap.approval_id, "weights": ap.payload["weights"]})

    def step(self, prices: pd.DataFrame, text_panel: Optional[pd.DataFrame], as_of) -> StepResult:
        as_of = pd.Timestamp(as_of)
        pnl = self._realise(prices, as_of)
        self._apply_approved()

        if self.control.state.kill_switch:
            flat = pd.Series(0.0, index=self.positions.index)
            self._record_orders(flat, as_of, "kill switch")
            self.positions = flat
            return StepResult(as_of, "flattened", flat, flat, pnl, [f"kill switch: {self.control.state.kill_switch_reason}"])

        active = [f for f in self.factors if f.name not in self.control.state.paused_factors]
        if not active:
            return StepResult(as_of, "no_factors", self.positions, pd.Series(dtype=float), pnl, ["all factors paused"])

        panel, _ = self.pipeline.prepare_panel(prices, text_panel, str(as_of.date()))
        values = {f.name: self.pipeline.factor_values(f, prices, text_panel, str(as_of.date())) for f in active}
        fw = self.factor_weights or self.portfolio.factor_weights(active)
        fw = {k: v for k, v in fw.items() if k in values}
        sized = self.portfolio.size_positions(self.portfolio.combine(values, fw), panel, overlay=self.overlay,
                                              target_vol=self.target_vol)
        target = sized.iloc[-1].fillna(0.0)
        report = self.risk.assess(as_of, target, self.pnl)
        self.audit.log(RISK_ACTION, "risk_agent", {"as_of": str(as_of.date()), "action": report.action,
                                                   "breaches": report.breaches, "scale": report.scale})
        target = target * report.scale * self.control.state.book_scale

        data_ts = panel.index.get_level_values("date").max()
        decision = self.guard.check(target, self.positions, self.pnl, as_of, data_ts, report.action)
        if not decision.allowed:
            return StepResult(as_of, "frozen", self.positions, target, pnl, decision.reasons)
        current = self.positions.reindex(target.index.union(self.positions.index), fill_value=0.0)
        new = current + (target.reindex(current.index, fill_value=0.0) - current) * decision.scale

        big = self.control.needs_approval(new, current)
        pending = []
        if not big.empty:
            ap = self.control.request_approval(
                f"{len(big)} position changes above {self.control.approval_threshold:.1%} on {as_of.date()}",
                {"weights": {t: float(new[t]) for t in big.index}, "as_of": str(as_of.date())},
            )
            pending.append(ap.approval_id)
            new.loc[big.index] = current.loc[big.index]  # hold until a human approves
        self._record_orders(new, as_of, "rebalance")
        self.positions = new
        return StepResult(as_of, "traded", new, target, pnl, decision.reasons, pending)

    def _record_orders(self, new: pd.Series, as_of: pd.Timestamp, reason: str) -> None:
        idx = new.index.union(self.positions.index)
        delta = new.reindex(idx, fill_value=0.0) - self.positions.reindex(idx, fill_value=0.0)
        delta = delta[delta.abs() > 1e-6]
        if len(delta):
            self.audit.log(ORDER, "live_loop", {"as_of": str(as_of.date()), "reason": reason, "n": int(len(delta)),
                                                "turnover": float(delta.abs().sum()),
                                                "orders": {k: round(float(v), 6) for k, v in delta.items()}})
