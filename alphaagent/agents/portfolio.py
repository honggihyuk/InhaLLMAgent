"""Portfolio agent: combines accepted factors into one signal and sizes positions.

Quantitative core (deterministic): ICIR-weighted factor blend -> rank weights -> risk overlay -> volatility
targeting. The LLM reviews the allocation and may only *reduce* risk (scale in [0, 1]) - it never sizes up.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alphaagent.agents.base import QuantAgent
from alphaagent.agents.types import AgentRole, AlphaFactor
from alphaagent.backtest.metrics import rank_weights
from alphaagent.validation.factors import cs_zscore


class PortfolioAgent(QuantAgent):
    role = AgentRole.PORTFOLIO
    system_prompt = (
        "You are a portfolio manager for a market-neutral equity book. You size positions conservatively, "
        "respect risk limits, and prefer diversified, low-turnover allocations."
    )

    def factor_weights(
        self, factors: List[AlphaFactor], emphasis: Optional[Dict[str, float]] = None, method: str = "icir"
    ) -> Dict[str, float]:
        raw = {}
        for f in factors:
            if method == "equal":
                raw[f.name] = 1.0
            else:
                raw[f.name] = max(float(f.metrics.get("oos_icir", f.metrics.get("icir", 0.0)) or 0.0), 0.0)
            if emphasis and f.name in emphasis:
                raw[f.name] *= float(np.clip(emphasis[f.name], 0.0, 2.0))
        total = sum(raw.values())
        if total <= 0:
            return {k: 1.0 / len(raw) for k in raw} if raw else {}
        return {k: v / total for k, v in raw.items()}

    @staticmethod
    def combine(values: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
        parts = [cs_zscore(values[k]).fillna(0.0) * w for k, w in weights.items() if k in values and w > 0]
        if not parts:
            raise ValueError("no factors to combine")
        return sum(parts[1:], parts[0]).rename("composite")

    @staticmethod
    def size_positions(
        composite: pd.Series,
        prices: pd.DataFrame,
        overlay=None,
        target_vol: float = 0.10,
        vol_lookback: int = 60,
        max_leverage: float = 2.0,
    ) -> pd.DataFrame:
        """Dollar-neutral weights scaled so the book's trailing realised vol tracks ``target_vol`` (annual).
        Scaling at t uses only P&L realised up to t-1."""
        base = rank_weights(composite.dropna())
        if overlay is not None:
            base = overlay(base)
        close = prices["close"].unstack("ticker").reindex(columns=base.columns)
        rets = close.pct_change().reindex(base.index)
        unscaled_pnl = (base.shift(1) * rets).sum(axis=1, min_count=1)
        realised = unscaled_pnl.rolling(vol_lookback, min_periods=20).std().shift(1) * np.sqrt(252)
        scale = (target_vol / realised).clip(upper=max_leverage).fillna(1.0)
        return base.mul(scale, axis=0)

    def review(self, weights: pd.DataFrame, factor_weights: Dict[str, float], risk_summary: Dict) -> Dict:
        last = weights.iloc[-1] if not weights.empty else pd.Series(dtype=float)
        top = last.abs().sort_values(ascending=False).head(10)
        task = f"""Review today's proposed book before it goes to trading.

Factor blend: {json.dumps(factor_weights)}
Gross exposure: {last.abs().sum():.3f}, net: {last.sum():.4f}, names: {(last != 0).sum()}
Largest positions: {json.dumps({k: round(float(last[k]), 4) for k in top.index})}
Risk metrics: {json.dumps(risk_summary, default=str)}

You may approve the book or scale it DOWN (scale between 0 and 1) if risk looks unjustified."""
        decision = self.think(
            task, kind="portfolio_review", use_retrieval=False, expect_json=True,
            answer_hint=' containing a ```json block: {"approve": true|false, "scale": 0.0-1.0, "notes": [...]}',
        )
        out = decision.output if isinstance(decision.output, dict) else {}
        out["scale"] = float(np.clip(float(out.get("scale", 1.0) or 0.0), 0.0, 1.0))
        if out.get("approve") is False:
            out["scale"] = min(out["scale"], 0.0)
        out["confidence"] = decision.confidence
        return out
