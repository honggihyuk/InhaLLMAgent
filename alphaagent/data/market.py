"""Real market data loaders producing the (date, ticker) panel the pipeline expects.

Columns: open, high, low, close, volume, vwap, returns, cap (cap may be NaN when shares are unknown).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "returns", "cap"]


def finalize_panel(df: pd.DataFrame, shares_outstanding: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.set_names(["date", "ticker"])
    df = df.sort_index()
    if "vwap" not in df:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3  # typical-price proxy without intraday data
    if "returns" not in df:
        df["returns"] = df.groupby(level="ticker")["close"].pct_change()
    if "cap" not in df:
        shares = pd.Series(shares_outstanding or {}, dtype=float)
        df["cap"] = df["close"] * df.index.get_level_values("ticker").map(shares).to_numpy(dtype=float)
    return df[PANEL_COLUMNS + [c for c in df.columns if c not in PANEL_COLUMNS]]


def load_prices_csv(path: str) -> pd.DataFrame:
    """Long CSV with columns date, ticker, open, high, low, close, volume[, vwap, cap]."""
    raw = pd.read_csv(Path(path), parse_dates=["date"])
    raw["ticker"] = raw["ticker"].str.upper()
    return finalize_panel(raw.set_index(["date", "ticker"]))


def load_prices_yfinance(tickers: Iterable[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Daily adjusted OHLCV from Yahoo Finance via the ``yfinance`` package (free, no key; best effort)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install yfinance to download prices") from e
    tickers = [t.upper() for t in tickers]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    frames = []
    for t in tickers:
        sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
        sub = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(how="all")
        sub["ticker"] = t
        frames.append(sub)
    df = pd.concat(frames).reset_index().rename(columns={"Date": "date", "index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return finalize_panel(df.set_index(["date", "ticker"]))


def synthetic_sectors(tickers: Iterable[str], n_sectors: int = 5, seed: int = 0) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    names = ["TECH", "HEALTH", "ENERGY", "FINANCIALS", "CONSUMER", "INDUSTRIALS", "UTILITIES", "MATERIALS"][:n_sectors]
    return {t: names[int(rng.integers(0, len(names)))] for t in tickers}
