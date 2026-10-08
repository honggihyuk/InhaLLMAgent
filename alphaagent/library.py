"""Persistent registry of every factor the system has generated, with metrics and verdicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from alphaagent.agents.types import AgentRole, AlphaFactor


def _factor_from_dict(d: Dict) -> AlphaFactor:
    d = dict(d)
    d["author_agent"] = AgentRole(d.get("author_agent", "implementer"))
    return AlphaFactor(**d)


class FactorLibrary:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self.factors: Dict[str, AlphaFactor] = {}
        self.values: Dict[str, pd.Series] = {}  # in-memory alpha values for redundancy checks
        if self.path and self.path.exists():
            for d in json.loads(self.path.read_text(encoding="utf-8")):
                f = _factor_from_dict(d)
                self.factors[f.name] = f

    def __len__(self) -> int:
        return len(self.factors)

    def __contains__(self, name: str) -> bool:
        return name in self.factors

    def unique_name(self, name: str) -> str:
        if name not in self.factors:
            return name
        i = 2
        while f"{name}_v{i}" in self.factors:
            i += 1
        return f"{name}_v{i}"

    def add(self, factor: AlphaFactor, values: Optional[pd.Series] = None) -> None:
        self.factors[factor.name] = factor
        if values is not None:
            self.values[factor.name] = values
        self.save()

    def accepted(self) -> List[AlphaFactor]:
        return [f for f in self.factors.values() if f.status == "accepted"]

    def feedback(self, limit: int = 15) -> List[Dict]:
        """Compact history fed back to the ideator (Alpha-GPT style evolutionary loop)."""
        rows = []
        for f in list(self.factors.values())[-limit:]:
            rows.append(
                {
                    "name": f.name,
                    "expression": f.expression,
                    "status": f.status,
                    "oos_ic": f.metrics.get("oos_ic"),
                    "oos_rank_ic": f.metrics.get("oos_rank_ic"),
                    "residual_ic": f.metrics.get("residual_oos_ic"),
                    "closest_known_factor": f.metrics.get("closest_known_factor"),
                    "max_known_corr": f.metrics.get("max_known_corr"),
                    "errors": f.errors[-1:] if f.errors else [],
                }
            )
        return rows

    def report(self) -> pd.DataFrame:
        rows = []
        for f in self.factors.values():
            m = f.metrics
            rows.append(
                {
                    "name": f.name,
                    "theme": f.theme,
                    "status": f.status,
                    "IS_IC": m.get("is_ic"),
                    "OOS_IC": m.get("oos_ic"),
                    "OOS_RankIC": m.get("oos_rank_ic"),
                    "OOS_tstat": m.get("oos_ic_tstat"),
                    "Residual_IC": m.get("residual_oos_ic"),
                    "MaxKnownCorr": m.get("max_known_corr"),
                    "WF_pos": m.get("wf_positive"),
                    "Span_t": m.get("span_alpha_tstat"),
                    "Sharpe_net": m.get("bt_sharpe"),
                    "MaxDD": m.get("bt_max_drawdown"),
                    "Turnover": m.get("bt_turnover"),
                }
            )
        return pd.DataFrame(rows)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [f.to_dict() for f in self.factors.values()]
        self.path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
