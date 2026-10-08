"""Synthetic market panels for tests and offline demos (fictional tickers, reproducible by seed)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def create_synthetic_market_data(
    n_tickers: int = 50,
    n_days: int = 252,
    start_date: str = "2024-01-02",
    seed: int = 42,
    planted_signal: Optional[str] = "reversal",
    signal_strength: float = 0.15,
    drift: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """OHLCV panel indexed by (date, ticker).

    ``planted_signal="reversal"`` injects a weak short-term reversal effect (next-day return negatively
    related to the trailing 5-day return) so that factor evaluation has something real to find.
    Use ``planted_signal=None`` for a pure random walk (every factor should have ~zero IC).
    ``drift`` (n_days x n_tickers) adds a deterministic return component in units of each name's idio vol.
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
        if drift is not None:
            noise = noise + drift[t] * idio_vol
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


_POS_LINES = [
    "Revenue grew {g}% year over year, ahead of our plan, with record free cash flow.",
    "We are raising full-year guidance on strong demand and expanded margins.",
    "Backlog reached an all-time high and pricing remains robust.",
    "Gross margin expanded {m} basis points as cost initiatives delivered.",
    "Customer wins accelerated and we exceeded our targets in every region.",
]
_NEG_LINES = [
    "Revenue declined {g}% as a key customer delayed orders.",
    "We are lowering guidance for the second half given weak end demand.",
    "Gross margin compressed {m} basis points on pricing pressure.",
    "Visibility is limited and we see continued headwinds in the channel.",
    "Results were challenging and fell short of our expectations.",
]
_RISK_LINES = [
    "We drew on the revolver to preserve liquidity.",
    "We received a subpoena and the investigation is ongoing; litigation outcomes are uncertain.",
    "We recorded an impairment on a prior acquisition.",
]
_NEUTRAL_LINES = [
    "Operator: Welcome to the quarterly conference call.",
    "Headcount was broadly flat quarter over quarter.",
    "Capital expenditure was in line with the prior year.",
    "We will host an investor day later this year.",
]


def create_synthetic_dataset(
    n_tickers: int = 40,
    n_days: int = 300,
    start_date: str = "2024-01-02",
    seed: int = 7,
    event_every: int = 21,
    drift_days: int = 10,
    text_strength: float = 0.2,
    reversal_strength: float = 0.05,
) -> Tuple[pd.DataFrame, list]:
    """Prices plus synthetic earnings-call / news documents whose tone predicts post-publication drift.

    Each name publishes roughly every ``event_every`` trading days. A document with tone s in [-1, 1]
    adds ``text_strength * s`` idio-vol units of drift on each of the next ``drift_days`` days, i.e. the
    market under-reacts to text. A text-based factor with no look-ahead should therefore find positive IC.
    All tickers are fictional (TICK_xxx) and all text is generated.
    """
    from alphaagent.documents import Document

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    tickers = [f"TICK_{i:03d}" for i in range(n_tickers)]
    drift = np.zeros((n_days, n_tickers))
    docs: List[Document] = []
    for j, tic in enumerate(tickers):
        t = int(rng.integers(5, event_every + 5))
        q = 0
        while t < n_days - 1:
            tone = float(np.clip(rng.normal(0, 0.6), -1, 1))
            drift[t + 1 : t + 1 + drift_days, j] += text_strength * tone
            docs.append(_make_doc(rng, tic, dates[t], tone, q))
            q += 1
            t += int(rng.integers(event_every - 5, event_every + 6))
    prices = create_synthetic_market_data(
        n_tickers, n_days, start_date, seed, planted_signal="reversal" if reversal_strength else None,
        signal_strength=reversal_strength, drift=drift,
    )
    return prices, docs


def _make_doc(rng, ticker, date, tone, q):
    from alphaagent.documents import Document

    n_signal = 1 + int(round(abs(tone) * 4))
    bank = _POS_LINES if tone >= 0 else _NEG_LINES
    lines = list(rng.choice(_NEUTRAL_LINES, size=2, replace=False))
    lines += [str(x) for x in rng.choice(bank, size=min(n_signal, len(bank)), replace=False)]
    if rng.random() < 0.25:  # noise: an off-tone remark
        other = _NEG_LINES if tone >= 0 else _POS_LINES
        lines.append(str(rng.choice(other)))
    if tone < -0.3 and rng.random() < 0.5:
        lines.append(str(rng.choice(_RISK_LINES)))
    rng.shuffle(lines)
    text = "\n\n".join(l.format(g=int(rng.integers(2, 25)), m=int(rng.integers(50, 400))) for l in lines)
    doc_type = "transcript" if q % 3 == 0 else "news"
    title = f"{ticker} {'earnings call' if doc_type == 'transcript' else 'news'} {date.date()}"
    return Document(ticker=ticker, doc_type=doc_type, title=title, text=text, published_at=date.date(),
                    source=f"synthetic://{ticker}/{date.date()}", metadata={"synthetic": True, "true_tone": round(tone, 3)})
