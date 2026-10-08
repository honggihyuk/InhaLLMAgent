"""Vectorised cross-sectional backtester (the "custom framework" option) plus a portfolio hook.

Timing: alpha at close t -> target weights at t -> traded with ``execution_lag`` days of delay
(default 1: traded at close t+1, earning the t+1 -> t+2 return). ``weight_fn`` lets a risk overlay
(max position, sector neutrality) reshape weights before trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import pandas as pd

from alphaagent.backtest.metrics import TRADING_DAYS, annualized_sharpe, max_drawdown, rank_weights

WeightFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class BacktestConfig:
    execution_lag: int = 1
    cost_bps: float = 5.0  # one-way cost per unit of turnover
    rebalance_every: int = 1  # trading days between rebalances
    weight_fn: Optional[WeightFn] = None


@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    stats: Dict[str, float] = field(default_factory=dict)


def _stats(returns: pd.Series, turnover: pd.Series) -> Dict[str, float]:
    if returns.empty:
        return {"sharpe": 0.0, "annual_return": 0.0, "annual_vol": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "days": 0}
    return {
        "sharpe": annualized_sharpe(returns),
        "annual_return": float(returns.mean() * TRADING_DAYS),
        "annual_vol": float(returns.std() * TRADING_DAYS ** 0.5),
        "max_drawdown": max_drawdown(returns),
        "turnover": float(turnover.mean()),
        "hit_rate": float((returns > 0).mean()),
        "days": int(len(returns)),
    }


def run_backtest(alpha: pd.Series, prices: pd.DataFrame, config: Optional[BacktestConfig] = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    target = rank_weights(alpha.dropna())
    if cfg.weight_fn is not None:
        target = cfg.weight_fn(target)
    if cfg.rebalance_every > 1:
        keep = target.index[:: cfg.rebalance_every]
        target = target.where(target.index.to_series().isin(keep), axis=0).ffill()

    close = prices["close"].unstack("ticker").reindex(columns=target.columns)
    daily_ret = close.pct_change().reindex(target.index)
    # weights held during day d+1 were decided at d - execution_lag
    held = target.shift(cfg.execution_lag).fillna(0.0)
    gross = (held.shift(1) * daily_ret).sum(axis=1, min_count=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    net = (gross - turnover * cfg.cost_bps / 1e4).dropna()
    net = net.iloc[cfg.execution_lag + 1 :]
    return BacktestResult(net, held, turnover, _stats(net, turnover))
