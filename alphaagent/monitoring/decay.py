"""Factor decay and live-performance monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from alphaagent.backtest.engine import BacktestConfig, run_backtest
from alphaagent.backtest.metrics import cross_sectional_corr, forward_returns


def rolling_ic(alpha: pd.Series, close: pd.Series, horizon: int = 5, window: int = 20) -> pd.Series:
    ic = cross_sectional_corr(alpha, forward_returns(close, horizon))
    return ic.rolling(window, min_periods=max(5, window // 2)).mean()


def ic_decay_curve(alpha: pd.Series, close: pd.Series, horizons: Sequence[int] = (1, 2, 3, 5, 10, 20)) -> pd.Series:
    """Mean IC of the signal against *non-overlapping-start* returns h days ahead (t+h-1 -> t+h)."""
    daily_fwd = forward_returns(close, 1)
    out = {}
    for h in horizons:
        lagged = daily_fwd.groupby(level="ticker").shift(-(h - 1))  # label only: return on day t+h
        ic = cross_sectional_corr(alpha, lagged)
        out[h] = float(ic.mean()) if len(ic) else np.nan
    return pd.Series(out, name="ic")


def ic_half_life(curve: pd.Series) -> Optional[float]:
    """Horizon (days) at which the per-day IC falls to half its day-1 value, by log-linear interpolation."""
    curve = curve.dropna()
    if curve.empty or curve.iloc[0] <= 0:
        return None
    half = curve.iloc[0] / 2
    below = curve[curve <= half]
    if below.empty:
        return float(curve.index[-1])  # still above half at the longest horizon
    h1 = below.index[0]
    prev = curve.index[curve.index.get_loc(h1) - 1]
    v0, v1 = curve[prev], curve[h1]
    if v0 == v1:
        return float(h1)
    return float(prev + (v0 - half) / (v0 - v1) * (h1 - prev))


@dataclass
class FactorHealth:
    name: str
    status: str  # healthy | decaying | dead | insufficient_data
    full_ic: float
    ic_20d: float
    ic_60d: float
    decay_ratio: float
    half_life_days: Optional[float]
    sharpe_60d: float
    current_drawdown: float

    def to_dict(self) -> Dict:
        return asdict(self)


def factor_health(
    name: str,
    alpha: pd.Series,
    prices: pd.DataFrame,
    horizon: int = 5,
    decay_threshold: float = 0.5,
    dead_ic: float = 0.0,
    cost_bps: float = 5.0,
) -> FactorHealth:
    close = prices["close"]
    ic = cross_sectional_corr(alpha, forward_returns(close, horizon))
    if len(ic) < 40:
        return FactorHealth(name, "insufficient_data", float(ic.mean()) if len(ic) else 0.0, 0, 0, 0, None, 0, 0)
    full = float(ic.mean())
    ic20, ic60 = float(ic.iloc[-20:].mean()), float(ic.iloc[-60:].mean())
    ratio = ic60 / full if full else 0.0
    bt = run_backtest(alpha, prices, BacktestConfig(cost_bps=cost_bps))
    r60 = bt.returns.iloc[-60:]
    eq = (1 + bt.returns).cumprod()
    dd = float(eq.iloc[-1] / eq.max() - 1) if len(eq) else 0.0
    sharpe60 = float(r60.mean() / r60.std() * np.sqrt(252)) if len(r60) > 5 and r60.std() > 0 else 0.0
    if ic60 <= dead_ic and ic20 <= dead_ic:
        status = "dead"
    elif ratio < decay_threshold:
        status = "decaying"
    else:
        status = "healthy"
    return FactorHealth(name, status, full, ic20, ic60, ratio, ic_half_life(ic_decay_curve(alpha, close)), sharpe60, dd)
