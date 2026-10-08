"""Look-ahead (future information leakage) detection for alpha code.

Static scan catches the common patterns named in the literature (negative shifts, backward fill,
centred windows, full-sample normalisation). The dynamic *truncation test* is the decisive check:
a causal factor's value at date t must not change when every row after a cutoff date is deleted.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class LookaheadReport:
    static_errors: List[str] = field(default_factory=list)
    static_warnings: List[str] = field(default_factory=list)
    dynamic_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.static_errors or self.dynamic_errors)

    def summary(self) -> str:
        lines = [f"ERROR: {e}" for e in self.static_errors + self.dynamic_errors]
        lines += [f"WARNING: {w}" for w in self.static_warnings]
        return "\n".join(lines) or "no look-ahead issues found"


def _num(node) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        return -v if v is not None else None
    return None


_PERIOD_METHODS = {"shift", "diff", "pct_change"}
_FULL_SAMPLE_STATS = {"mean", "std", "var", "max", "min", "median", "quantile", "sum"}
_SAFE_PARENTS = {"rolling", "expanding", "ewm", "groupby", "resample", "transform"}


def static_scan(code: str) -> LookaheadReport:
    rep = LookaheadReport()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        rep.static_errors.append(f"syntax error: {e.msg}")
        return rep

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        line = getattr(node, "lineno", "?")
        if name in _PERIOD_METHODS:
            periods = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in {"periods", "n"}), None)
            v = _num(periods) if periods is not None else None
            if v is not None and v < 0:
                rep.static_errors.append(f"line {line}: .{name}({int(v)}) reads future rows")
        elif name in {"bfill", "backfill"}:
            rep.static_errors.append(f"line {line}: .{name}() propagates future values backwards")
        elif name == "fillna":
            for k in node.keywords:
                if k.arg == "method" and isinstance(k.value, ast.Constant) and k.value.value in {"bfill", "backfill"}:
                    rep.static_errors.append(f"line {line}: fillna(method='{k.value.value}') leaks future values")
        elif name in {"rolling"}:
            for k in node.keywords:
                if k.arg == "center" and isinstance(k.value, ast.Constant) and k.value.value is True:
                    rep.static_errors.append(f"line {line}: rolling(center=True) uses future rows")
        elif name in _FULL_SAMPLE_STATS and not node.args and not node.keywords:
            # x.mean() directly on a column (not after rolling/expanding/groupby) = full-sample statistic
            inner = node.func.value
            parent = inner.func.attr if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) else None
            if parent not in _SAFE_PARENTS and isinstance(inner, (ast.Subscript, ast.Name, ast.Attribute)):
                rep.static_warnings.append(
                    f"line {line}: .{name}() over the whole column may use full-sample (future) statistics"
                )
    return rep


AlphaFn = Callable[[pd.DataFrame], pd.Series]


def truncation_test(
    compute: AlphaFn,
    data: pd.DataFrame,
    cutoffs: Optional[Sequence] = None,
    n_cutoffs: int = 3,
    tol: float = 1e-8,
    compare_last: int = 30,
    full: Optional[pd.Series] = None,
) -> List[str]:
    """Return a list of violations. ``compute`` runs the alpha (e.g. through the sandbox)."""
    dates = data.index.get_level_values("date").unique().sort_values()
    if cutoffs is None:
        qs = np.linspace(0.45, 0.85, n_cutoffs)
        cutoffs = [dates[int(q * (len(dates) - 1))] for q in qs]
    if full is None:
        full = compute(data)
    errors: List[str] = []
    for cut in cutoffs:
        cut = pd.Timestamp(cut)
        trunc_data = data[data.index.get_level_values("date") <= cut]
        trunc = compute(trunc_data)
        window = dates[(dates <= cut)][-compare_last:]
        mask_full = full.index.get_level_values("date").isin(window)
        a = full[mask_full]
        b = trunc.reindex(a.index)
        both_nan = a.isna() & b.isna()
        diff = (a - b).abs()
        scale = np.maximum(1.0, a.abs())
        bad = ~both_nan & ((a.isna() != b.isna()) | (diff > tol * scale))
        if bad.any():
            first = a.index[bad.values][0]
            errors.append(
                f"values up to {cut.date()} change when later data is removed "
                f"({int(bad.sum())} of {len(a)} checked rows differ, e.g. {first[0].date()} {first[1]})"
            )
    return errors


def check_lookahead(code: str, compute: Optional[AlphaFn] = None, data: Optional[pd.DataFrame] = None, **kw) -> LookaheadReport:
    rep = static_scan(code)
    if compute is not None and data is not None and not rep.static_errors:
        rep.dynamic_errors = truncation_test(compute, data, **kw)
    return rep
