"""Factor evaluation metrics on a (date, ticker) panel.

Convention: ``alpha`` at date t uses information available at the close of t; it is evaluated against
returns from t to t+h (``forward_returns``). Portfolio P&L at t+1 uses weights formed at t.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _as_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("alpha must be a Series or single-column DataFrame")
        x = x.iloc[:, 0]
    return x


def forward_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Return from t to t+h per ticker. This is a *label*, so looking forward is intended here."""
    by_ticker = close.groupby(level="ticker")
    return by_ticker.shift(-horizon) / close - 1.0


def cross_sectional_corr(alpha: pd.Series, fwd: pd.Series, method: str = "pearson", min_names: int = 5) -> pd.Series:
    df = pd.concat({"a": _as_series(alpha), "r": fwd}, axis=1).dropna()
    out = {}
    for date, g in df.groupby(level="date"):
        if len(g) < min_names or g["a"].nunique() < 2 or g["r"].nunique() < 2:
            continue
        if method == "spearman":
            out[date] = g["a"].rank().corr(g["r"].rank())
        else:
            out[date] = g["a"].corr(g["r"])
    return pd.Series(out, dtype=float).sort_index()


def rank_weights(alpha: pd.Series) -> pd.DataFrame:
    """Dollar-neutral, unit-gross weights from cross-sectional ranks (date x ticker)."""
    wide = _as_series(alpha).unstack("ticker")
    ranks = wide.rank(axis=1)
    demeaned = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return demeaned.div(gross, axis=0).fillna(0.0)


def portfolio_returns(weights: pd.DataFrame, close: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    """Daily P&L: weights at t times returns t->t+1, minus turnover * cost."""
    nxt = forward_returns(close, 1).unstack("ticker").reindex_like(weights)
    gross = (weights * nxt).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    pnl = gross - turnover * cost_bps / 1e4
    return pnl.dropna()


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def summarize_factor(
    alpha: pd.Series, close: pd.Series, horizon: int = 5, cost_bps: float = 0.0, min_names: int = 5
) -> Dict[str, float]:
    alpha = _as_series(alpha).astype(float).replace([np.inf, -np.inf], np.nan)
    fwd = forward_returns(close, horizon)
    ic = cross_sectional_corr(alpha, fwd, "pearson", min_names)
    ric = cross_sectional_corr(alpha, fwd, "spearman", min_names)
    weights = rank_weights(alpha.dropna())
    pnl = portfolio_returns(weights, close, cost_bps)
    turnover = weights.diff().abs().sum(axis=1).iloc[1:].mean() if len(weights) > 1 else 0.0
    n_obs = int(alpha.notna().sum())
    return {
        "ic": float(ic.mean()) if len(ic) else 0.0,
        "ic_std": float(ic.std()) if len(ic) > 1 else 0.0,
        "icir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else 0.0,
        "ic_tstat": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 1 and ic.std() > 0 else 0.0,
        "rank_ic": float(ric.mean()) if len(ric) else 0.0,
        "ic_positive_ratio": float((ic > 0).mean()) if len(ic) else 0.0,
        "sharpe": annualized_sharpe(pnl),
        "annual_return": float(pnl.mean() * TRADING_DAYS) if len(pnl) else 0.0,
        "max_drawdown": max_drawdown(pnl),
        "turnover": float(turnover) if turnover == turnover else 0.0,
        "coverage": float(n_obs / max(len(close), 1)),
        "n_dates": int(len(ic)),
    }
