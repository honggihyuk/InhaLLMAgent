"""Walk-forward validation, spanning tests against known factors, and point-in-time historical reruns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from alphaagent.backtest.metrics import cross_sectional_corr, forward_returns, portfolio_returns, rank_weights
from alphaagent.validation.factors import KNOWN_FACTORS


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame  # one row per test window
    mean_oos_ic: float = 0.0
    positive_windows: float = 0.0  # share of windows with OOS IC > 0
    oos_ic_tstat: float = 0.0


def walk_forward(
    alpha: pd.Series,
    close: pd.Series,
    train_days: int = 126,
    test_days: int = 21,
    horizon: int = 5,
) -> WalkForwardResult:
    """Rolling train/test windows. The sign is fixed on each train window, then scored on the next test window."""
    dates = alpha.index.get_level_values("date").unique().sort_values()
    fwd = forward_returns(close, horizon)
    ic = cross_sectional_corr(alpha, fwd)
    rows = []
    start = train_days
    while start + test_days <= len(dates):
        train = dates[start - train_days : start - horizon]  # purge: train labels must end before test starts
        test = dates[start : start + test_days - horizon] if test_days > horizon else dates[start : start + test_days]
        ic_train = ic.reindex(train).dropna()
        ic_test = ic.reindex(test).dropna()
        if len(ic_train) and len(ic_test):
            sign = 1.0 if ic_train.mean() >= 0 else -1.0
            rows.append(
                {
                    "train_end": train[-1],
                    "test_start": test[0],
                    "test_end": test[-1],
                    "train_ic": float(ic_train.mean()),
                    "sign": sign,
                    "oos_ic": float(sign * ic_test.mean()),
                }
            )
        start += test_days
    df = pd.DataFrame(rows)
    if df.empty:
        return WalkForwardResult(df)
    t = df["oos_ic"].mean() / (df["oos_ic"].std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 and df["oos_ic"].std() > 0 else 0.0
    return WalkForwardResult(df, float(df["oos_ic"].mean()), float((df["oos_ic"] > 0).mean()), float(t))


@dataclass
class SpanningResult:
    alpha_daily: float  # intercept: return not explained by known factor portfolios
    alpha_tstat: float
    betas: Dict[str, float] = field(default_factory=dict)
    r2: float = 0.0
    annualized_alpha: float = 0.0


def known_factor_returns(panel: pd.DataFrame, factors: Optional[Dict[str, Callable]] = None) -> pd.DataFrame:
    factors = factors or KNOWN_FACTORS
    out = {}
    for name, fn in factors.items():
        try:
            out[name] = portfolio_returns(rank_weights(fn(panel).dropna()), panel["close"])
        except KeyError:
            continue
    return pd.DataFrame(out)


def spanning_test(factor_returns: pd.Series, known_returns: pd.DataFrame, nw_lags: int = 5) -> SpanningResult:
    """Regress the candidate's long-short returns on known factor returns; Newey-West t-stat on the intercept."""
    known_returns = known_returns.reindex(factor_returns.index).dropna(axis=1, thresh=max(10, len(factor_returns) // 2))
    df = pd.concat([factor_returns.rename("y"), known_returns], axis=1).dropna()
    if len(df) < known_returns.shape[1] + 10:
        return SpanningResult(0.0, 0.0)
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[known_returns.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    # Newey-West HAC covariance
    S = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, nw_lags + 1):
        w = 1 - lag / (nw_lags + 1)
        G = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        S += w * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(max(cov[0, 0], 1e-18))
    ss = ((y - y.mean()) ** 2).sum()
    return SpanningResult(
        alpha_daily=float(beta[0]),
        alpha_tstat=float(beta[0] / se),
        betas={c: float(b) for c, b in zip(known_returns.columns, beta[1:])},
        r2=float(1 - (e ** 2).sum() / ss) if ss > 0 else 0.0,
        annualized_alpha=float(beta[0] * 252),
    )


@dataclass
class HistoricalRun:
    as_of: pd.Timestamp
    accepted: List[str]  # newly accepted in this run
    forward_ic: Dict[str, float]  # IC over the period AFTER as_of of every factor live at as_of
    live: List[str] = field(default_factory=list)  # all factors accepted so far (deployed set)


def historical_reruns(
    pipeline,
    theme: str,
    price_data: pd.DataFrame,
    text_panel: Optional[pd.DataFrame] = None,
    start_days: int = 189,
    step_days: int = 21,
    horizon: int = 5,
) -> List[HistoricalRun]:
    """Re-run the whole pipeline at successive as-of dates, each run seeing only data up to its date,
    then score what it accepted on the following ``step_days``. This is the honest 12+ month backtest
    of the *research process*, not just of one factor."""
    dates = price_data.index.get_level_values("date").unique().sort_values()
    runs: List[HistoricalRun] = []
    i = start_days
    while i + step_days <= len(dates):
        as_of = dates[i - 1]
        accepted = pipeline.run_pipeline(theme, price_data, text_panel, as_of=str(as_of.date()))
        # the deployed set is everything accepted so far; each was accepted on data up to its own as-of date
        library = getattr(pipeline, "library", None)
        live = library.accepted() if library is not None else accepted
        nxt = dates[i : i + step_days]
        window = price_data[price_data.index.get_level_values("date").isin(dates[max(0, i - 260) : i + step_days])]
        tp = text_panel.reindex(window.index) if text_panel is not None else None
        fwd_ic = {}
        for f in live:
            vals = pipeline.factor_values(f, window, tp)
            ic = cross_sectional_corr(vals, forward_returns(window["close"], horizon)).reindex(nxt[:-horizon] if step_days > horizon else nxt)
            fwd_ic[f.name] = float(ic.mean()) if ic.notna().any() else float("nan")
        runs.append(HistoricalRun(as_of, [f.name for f in accepted], fwd_ic, [f.name for f in live]))
        i += step_days
    return runs
