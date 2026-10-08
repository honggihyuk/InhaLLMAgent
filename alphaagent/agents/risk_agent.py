"""Risk agent: continuous monitoring of the live book, limit checks, and alerts on the message bus."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alphaagent.agents.base import QuantAgent
from alphaagent.agents.protocol import BROADCAST, MessageType
from alphaagent.agents.types import AgentRole


@dataclass
class RiskMonitorLimits:
    max_drawdown: float = 0.10  # from peak
    max_var95: float = 0.02  # 1-day historical VaR as a fraction of capital
    max_vol: float = 0.20  # annualised realised vol (20d)
    max_gross: float = 2.0
    max_position: float = 0.05
    max_sector_net: float = 0.05
    halt_drawdown: float = 0.15  # beyond this: flatten the book


@dataclass
class RiskReport:
    date: str
    metrics: Dict[str, float]
    breaches: List[str] = field(default_factory=list)
    action: str = "none"  # none | reduce | halt
    scale: float = 1.0  # multiplier the risk agent applies to the book

    def to_dict(self) -> Dict:
        return asdict(self)


def risk_metrics(weights_row: pd.Series, pnl_history: pd.Series, sectors: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    w = weights_row.fillna(0.0)
    hist = pnl_history.dropna()
    eq = (1 + hist).cumprod()
    tail = hist.iloc[-250:]
    var95 = float(-np.quantile(tail, 0.05)) if len(tail) >= 20 else 0.0
    cvar95 = float(-tail[tail <= np.quantile(tail, 0.05)].mean()) if len(tail) >= 20 else 0.0
    gross = float(w.abs().sum())
    hhi = float(((w.abs() / gross) ** 2).sum()) if gross > 0 else 0.0
    m = {
        "gross": gross,
        "net": float(w.sum()),
        "max_position": float(w.abs().max()) if len(w) else 0.0,
        "names": int((w != 0).sum()),
        "hhi": hhi,
        "vol_20d": float(hist.iloc[-20:].std() * np.sqrt(252)) if len(hist) >= 10 else 0.0,
        "var95": var95,
        "cvar95": cvar95,
        "drawdown": float(eq.iloc[-1] / eq.max() - 1) if len(eq) else 0.0,
    }
    if sectors:
        sec = pd.Series({t: sectors.get(t, "UNKNOWN") for t in w.index})
        m["max_sector_net"] = float(w.groupby(sec).sum().abs().max()) if len(w) else 0.0
    return m


class RiskAgent(QuantAgent):
    role = AgentRole.RISK
    system_prompt = (
        "You are the chief risk officer of a systematic equity fund. You are conservative, precise, and you "
        "explain limit breaches and recommended actions in plain language."
    )

    def __init__(self, *args, limits: Optional[RiskMonitorLimits] = None, sectors: Optional[Dict[str, str]] = None, **kw):
        super().__init__(*args, **kw)
        self.limits = limits or RiskMonitorLimits()
        self.sectors = sectors

    def assess(self, date, weights_row: pd.Series, pnl_history: pd.Series, factor_health: Optional[List] = None) -> RiskReport:
        lim = self.limits
        m = risk_metrics(weights_row, pnl_history, self.sectors)
        breaches = []
        if m["drawdown"] <= -lim.max_drawdown:
            breaches.append(f"drawdown {m['drawdown']:.1%} beyond {lim.max_drawdown:.0%}")
        if m["var95"] > lim.max_var95:
            breaches.append(f"VaR95 {m['var95']:.2%} above {lim.max_var95:.2%}")
        if m["vol_20d"] > lim.max_vol:
            breaches.append(f"20d vol {m['vol_20d']:.1%} above {lim.max_vol:.0%}")
        if m["gross"] > lim.max_gross + 1e-9:
            breaches.append(f"gross {m['gross']:.2f} above {lim.max_gross:.2f}")
        if m["max_position"] > lim.max_position + 1e-9:
            breaches.append(f"position {m['max_position']:.2%} above {lim.max_position:.2%}")
        if m.get("max_sector_net", 0) > lim.max_sector_net + 1e-9:
            breaches.append(f"sector net {m['max_sector_net']:.2%} above {lim.max_sector_net:.2%}")
        for h in factor_health or []:
            if getattr(h, "status", "") == "dead":
                breaches.append(f"factor {h.name} is dead (recent IC <= 0)")

        action, scale = "none", 1.0
        if m["drawdown"] <= -lim.halt_drawdown:
            action, scale = "halt", 0.0
        elif breaches:
            action = "reduce"
            ratios = [1.0]
            if m["vol_20d"] > lim.max_vol:
                ratios.append(lim.max_vol / m["vol_20d"])
            if m["var95"] > lim.max_var95:
                ratios.append(lim.max_var95 / m["var95"])
            if m["gross"] > lim.max_gross:
                ratios.append(lim.max_gross / m["gross"])
            if m["drawdown"] <= -lim.max_drawdown:
                ratios.append(0.5)
            scale = float(min(ratios))
        report = RiskReport(str(pd.Timestamp(date).date()), m, breaches, action, scale)
        if breaches and self.bus is not None:
            self.send(BROADCAST, MessageType.ALERT, f"{report.date}: {action} (scale {scale:.2f}): " + "; ".join(breaches),
                      {"report": report.to_dict()})
        self.memory.write(self.name, self.role.value, "risk", f"{report.date} action={action} breaches={len(breaches)}",
                          report.to_dict())
        return report

    def monitor(self, weights: pd.DataFrame, pnl: pd.Series, every: int = 1, factor_health=None) -> pd.DataFrame:
        """Run :meth:`assess` through history (continuous monitoring). Returns one row per checked date."""
        rows = []
        for i, date in enumerate(weights.index):
            if i % every:
                continue
            rep = self.assess(date, weights.loc[date], pnl.loc[:date], factor_health)
            rows.append({"date": date, "action": rep.action, "scale": rep.scale, "breaches": len(rep.breaches), **rep.metrics})
        return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

    def explain(self, report: RiskReport) -> str:
        task = f"""Explain this risk report to the portfolio manager in at most five sentences and recommend actions.

{json.dumps(report.to_dict(), indent=1, default=str)}"""
        return self.think(task, kind="risk_review", use_retrieval=False).content
