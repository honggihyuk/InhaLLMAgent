"""Risk management overlay applied to target weights before trading.

Constraints (applied per date, in order):
  1. sector neutrality - weights demeaned within each sector (net 0 per sector)
  2. per-name cap      - |w_i| <= max_position (as a fraction of gross capital)
  3. gross / net       - sum|w| == max_gross and |sum w| <= max_net
Capping and re-scaling interact, so steps 2-3 iterate until stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class RiskLimits:
    max_position: float = 0.05
    max_gross: float = 1.0
    max_net: float = 0.05
    sector_neutral: bool = True
    max_sector_gross: Optional[float] = None  # cap on sum|w| within one sector
    iterations: int = 20


@dataclass
class RiskOverlay:
    limits: RiskLimits = field(default_factory=RiskLimits)
    sectors: Optional[Dict[str, str]] = None  # ticker -> sector

    def _sector_neutralise(self, w: pd.Series) -> pd.Series:
        if not self.sectors:
            return w - w.mean()
        sec = pd.Series({t: self.sectors.get(t, "UNKNOWN") for t in w.index})
        return w - w.groupby(sec).transform("mean")

    @staticmethod
    def _balance(w: pd.Series, tolerance: float = 0.0) -> pd.Series:
        """Make sum(w) ~ 0 (within ``tolerance``) by shrinking the heavier side only, so caps stay satisfied."""
        long, short = w[w > 0].sum(), -w[w < 0].sum()
        if long - short > tolerance and long > 0:
            w = w.where(w <= 0, w * (short + tolerance) / long)
        elif short - long > tolerance and short > 0:
            w = w.where(w >= 0, w * (long + tolerance) / short)
        return w

    def _groups(self, w: pd.Series) -> pd.Series:
        if self.sectors and self.limits.sector_neutral:
            return pd.Series({t: self.sectors.get(t, "UNKNOWN") for t in w.index})
        return pd.Series("ALL", index=w.index)

    def apply_row(self, w: pd.Series) -> pd.Series:
        lim = self.limits
        w = w.fillna(0.0).astype(float)
        if not w.abs().sum():
            return w
        groups = self._groups(w)
        tol = 0.0 if lim.sector_neutral else lim.max_net
        if lim.sector_neutral:
            w = self._sector_neutralise(w)
        for _ in range(lim.iterations):
            gross = w.abs().sum()
            if gross == 0:
                break
            w = (w * (lim.max_gross / gross)).clip(-lim.max_position, lim.max_position)
            if lim.max_sector_gross and self.sectors:
                sec = pd.Series({t: self.sectors.get(t, "UNKNOWN") for t in w.index})
                sg = w.abs().groupby(sec).transform("sum")
                w = w * np.minimum(1.0, lim.max_sector_gross / sg.replace(0, np.nan)).fillna(1.0)
            w = w.groupby(groups).transform(lambda g: self._balance(g, tol))
            if abs(w.abs().sum() - lim.max_gross) < 1e-6:
                break
        # uniform scaling preserves neutrality: grow towards target gross without breaching the cap
        gross, peak = w.abs().sum(), w.abs().max()
        if gross > 0 and peak > 0:
            w = w * min(lim.max_gross / gross, lim.max_position / peak)
        return w

    def __call__(self, weights: pd.DataFrame) -> pd.DataFrame:
        return weights.apply(self.apply_row, axis=1)

    def check(self, weights: pd.DataFrame) -> Dict[str, float]:
        """Report realised exposures (used by the risk agent / monitoring)."""
        out = {
            "max_abs_position": float(weights.abs().max().max()) if not weights.empty else 0.0,
            "max_gross": float(weights.abs().sum(axis=1).max()) if not weights.empty else 0.0,
            "max_abs_net": float(weights.sum(axis=1).abs().max()) if not weights.empty else 0.0,
        }
        if self.sectors and not weights.empty:
            sec = pd.Series({t: self.sectors.get(t, "UNKNOWN") for t in weights.columns})
            sector_net = weights.T.groupby(sec).sum().T
            out["max_abs_sector_net"] = float(sector_net.abs().max().max())
        return out
