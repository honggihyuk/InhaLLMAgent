"""Known-factor library and factor orthogonalisation.

A "novel" LLM alpha is often a known factor in disguise (Wang et al., 2023). We z-score each factor
cross-sectionally, regress the candidate on the known factors date by date, and keep the residual.
The residual's IC is the factor's *incremental* information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


def cs_zscore(x: pd.Series, clip: float = 5.0) -> pd.Series:
    g = x.groupby(level="date")
    z = (x - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.clip(-clip, clip)


def _ts(df: pd.DataFrame, col: str, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
    return df.groupby(level="ticker")[col].transform(fn)


# Proxies for the classic style factors (Fama-French / Barra style) computable from an OHLCV panel.
KNOWN_FACTORS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "momentum": lambda df: _ts(df, "returns", lambda s: s.rolling(120, min_periods=60).sum().shift(5)),
    "short_term_reversal": lambda df: -_ts(df, "returns", lambda s: s.rolling(5).sum()),
    "low_volatility": lambda df: -_ts(df, "returns", lambda s: s.rolling(20).std()),
    "size": lambda df: np.log(df["cap"]),
    "liquidity": lambda df: np.log(
        (df["close"] * df["volume"]).groupby(level="ticker").transform(lambda s: s.rolling(20).mean())
    ),
}


def compute_known_factors(df: pd.DataFrame, factors: Optional[Dict[str, Callable]] = None) -> pd.DataFrame:
    factors = factors or KNOWN_FACTORS
    out = {}
    for name, fn in factors.items():
        try:
            out[name] = cs_zscore(fn(df).astype(float))
        except KeyError:
            continue  # required column missing from this panel
    return pd.DataFrame(out, index=df.index)


@dataclass
class OrthogonalizationResult:
    residual: pd.Series
    correlations: Dict[str, float] = field(default_factory=dict)  # mean cross-sectional corr per known factor
    r2: float = 0.0  # mean cross-sectional R^2 explained by known factors
    max_abs_corr: float = 0.0
    closest_factor: str = ""


def orthogonalize(alpha: pd.Series, known: pd.DataFrame, min_names: int = 10) -> OrthogonalizationResult:
    z = cs_zscore(alpha.astype(float))
    joined = pd.concat([z.rename("__alpha__"), known], axis=1)
    resid = pd.Series(np.nan, index=alpha.index, dtype=float)
    r2s = []
    corrs: Dict[str, list] = {c: [] for c in known.columns}
    for _, g in joined.groupby(level="date"):
        g = g.dropna()
        if len(g) < max(min_names, known.shape[1] + 2):
            continue
        y = g["__alpha__"].to_numpy()
        X = g[known.columns].to_numpy()
        if y.std() < 1e-12:
            continue
        for j, c in enumerate(known.columns):
            if X[:, j].std() > 1e-12:
                corrs[c].append(float(np.corrcoef(y, X[:, j])[0, 1]))
        Xc = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        e = y - Xc @ beta
        resid.loc[g.index] = e
        ss = ((y - y.mean()) ** 2).sum()
        r2s.append(1 - (e ** 2).sum() / ss)
    mean_corr ={c: float(np.mean(v)) if v else 0.0 for c, v in corrs.items()}
    closest = max(mean_corr, key=lambda c: abs(mean_corr[c])) if mean_corr else ""
    return OrthogonalizationResult(
        residual=resid,
        correlations=mean_corr,
        r2=float(np.mean(r2s)) if r2s else 0.0,
        max_abs_corr=abs(mean_corr[closest]) if closest else 0.0,
        closest_factor=closest,
    )


def factor_correlation(a: pd.Series, b: pd.Series) -> float:
    """Mean cross-sectional rank correlation between two factors (redundancy check)."""
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if df.empty:
        return 0.0
    vals = [
        g["a"].rank().corr(g["b"].rank())
        for _, g in df.groupby(level="date")
        if len(g) > 3 and g["a"].nunique() > 1 and g["b"].nunique() > 1
    ]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else 0.0
