"""Synthetic market panels for tests and offline demos (fictional tickers, reproducible by seed)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def create_synthetic_market_data(
    n_tickers: int = 50,
    n_days: int = 252,
    start_date: str = "2024-01-02",
    seed: int = 42,
    planted_signal: Optional[str] = "reversal",
    signal_strength: float = 0.15,
) -> pd.DataFrame:
    """OHLCV panel indexed by (date, ticker).

    ``planted_signal="reversal"`` injects a weak short-term reversal effect (next-day return negatively
    related to the trailing 5-day return) so that factor evaluation has something real to find.
    Use ``planted_signal=None`` for a pure random walk (every factor should have ~zero IC).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    tickers = [f"TICK_{i:03d}" for i in range(n_tickers)]

    market = rng.normal(0.0003, 0.008, n_days)
    beta = rng.uniform(0.6, 1.4, n_tickers)
    idio_vol = rng.uniform(0.01, 0.03, n_tickers)
    rets = np.zeros((n_days, n_tickers))
    for t in range(n_days):
        noise = rng.normal(0, idio_vol)
        if planted_signal == "reversal" and t >= 5:
            past = rets[t - 5 : t].sum(axis=0)
            z = (past - past.mean()) / (past.std() + 1e-12)
            noise = noise - signal_strength * z * idio_vol
        rets[t] = beta * market[t] + noise

    close = 100 * np.exp(np.cumsum(rets, axis=0))
    spread = np.abs(rng.normal(0, 0.008, (n_days, n_tickers)))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = np.vstack([close[:1], close[:-1]]) * (1 + rng.normal(0, 0.003, (n_days, n_tickers)))
    vwap = (high + low + close) / 3
    volume = rng.lognormal(14, 0.4, (n_days, n_tickers)) * (1 + 20 * np.abs(rets))
    shares = rng.lognormal(18, 1.0, n_tickers)
    cap = close * shares

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame(
        {
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "volume": volume.ravel(),
            "vwap": vwap.ravel(),
            "returns": rets.ravel(),
            "cap": cap.ravel(),
        },
        index=idx,
    )
    return df
