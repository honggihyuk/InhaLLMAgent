"""Self-contained HTML monitoring dashboard (inline SVG, no external assets)."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from alphaagent.backtest.engine import BacktestConfig, run_backtest
from alphaagent.monitoring.decay import FactorHealth, factor_health, ic_decay_curve, rolling_ic

PALETTE = ["#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2", "#65a30d", "#dc2626"]
STATUS_COLOR = {"healthy": "#059669", "decaying": "#d97706", "dead": "#dc2626", "insufficient_data": "#6b7280"}


def _line_chart(series: Dict[str, pd.Series], title: str, width=640, height=220, zero_line=True, pct=False) -> str:
    pad_l, pad_r, pad_t, pad_b = 52, 12, 28, 26
    clean = {k: s.dropna() for k, s in series.items() if s is not None and s.dropna().size > 1}
    if not clean:
        return f'<div class="chart"><div class="ct">{html.escape(title)}</div><p class="muted">no data</p></div>'
    xs = sorted(set().union(*[set(s.index) for s in clean.values()]))
    xpos = {x: i for i, x in enumerate(xs)}
    lo = min(float(s.min()) for s in clean.values())
    hi = max(float(s.max()) for s in clean.values())
    if zero_line:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        hi = lo + 1e-9
    W, H = width - pad_l - pad_r, height - pad_t - pad_b

    def X(x):
        return pad_l + W * xpos[x] / max(len(xs) - 1, 1)

    def Y(v):
        return pad_t + H * (1 - (v - lo) / (hi - lo))

    fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.3f}")
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for v in np.linspace(lo, hi, 4):
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
    if zero_line:
        parts.append(f'<line class="zero" x1="{pad_l}" x2="{width - pad_r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    for i, (name, s) in enumerate(clean.items()):
        pts = " ".join(f"{X(x):.1f},{Y(float(v)):.1f}" for x, v in s.items())
        parts.append(f'<polyline fill="none" stroke="{PALETTE[i % len(PALETTE)]}" stroke-width="1.8" points="{pts}"/>')
    for x in (xs[0], xs[-1]):
        label = x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)
        anchor = "start" if x == xs[0] else "end"
        parts.append(f'<text class="tick" x="{X(x):.1f}" y="{height - 6}" text-anchor="{anchor}">{label}</text>')
    parts.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{PALETTE[i % len(PALETTE)]}"></i>{html.escape(n)}</span>' for i, n in enumerate(clean)
    )
    return f'<div class="chart"><div class="ct">{html.escape(title)}</div>{"".join(parts)}<div class="legend">{legend}</div></div>'


def _bar_chart(values: Dict[str, pd.Series], title: str) -> str:
    # decay curves: x = horizon, rendered as lines for multi-factor comparison
    return _line_chart({k: v.astype(float) for k, v in values.items()}, title, zero_line=True)


def _health_table(health: List[FactorHealth], extra: Dict[str, Dict]) -> str:
    rows = []
    for h in health:
        e = extra.get(h.name, {})
        color = STATUS_COLOR.get(h.status, "#6b7280")
        hl = f"{h.half_life_days:.1f}" if h.half_life_days is not None else "–"
        rows.append(
            f"<tr><td>{html.escape(h.name)}</td>"
            f'<td><span class="pill" style="border-color:{color};color:{color}">{h.status}</span></td>'
            f"<td>{h.full_ic:.3f}</td><td>{h.ic_60d:.3f}</td><td>{h.ic_20d:.3f}</td><td>{h.decay_ratio:.2f}</td>"
            f"<td>{hl}</td><td>{h.sharpe_60d:.2f}</td><td>{h.current_drawdown:.1%}</td>"
            f"<td>{e.get('max_known_corr', float('nan')):.2f}</td></tr>"
        )
    head = "".join(
        f"<th>{c}</th>"
        for c in ["Factor", "Status", "IC (all)", "IC 60d", "IC 20d", "Decay ratio", "Half-life (d)", "Sharpe 60d", "Drawdown", "Max known corr"]
    )
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


CSS = """
:root{--bg:#f8fafc;--panel:#ffffff;--text:#0f172a;--muted:#64748b;--grid:#e2e8f0;--zero:#94a3b8}
@media (prefers-color-scheme: dark){:root{--bg:#0b1220;--panel:#111a2e;--text:#e2e8f0;--muted:#94a3b8;--grid:#1e293b;--zero:#475569}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1320px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}.muted{color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px;margin-top:16px}
.chart,.card{background:var(--panel);border-radius:10px;padding:12px 14px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.ct{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}.zero{stroke:var(--zero);stroke-dasharray:3 3}.tick{fill:var(--muted);font-size:10px}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:10px;height:3px;margin-right:5px;vertical-align:middle}
.tablewrap{overflow-x:auto;margin-top:16px;background:var(--panel);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}.pill{border:1px solid;border-radius:999px;padding:1px 8px;font-size:12px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}.kpi{background:var(--panel);border-radius:10px;padding:10px 14px;min-width:150px}
.kpi b{display:block;font-size:20px}
"""


def build_dashboard(
    factors: Dict[str, pd.Series],
    prices: pd.DataFrame,
    out_path: str,
    title: str = "Alpha factor monitor",
    extra: Optional[Dict[str, Dict]] = None,
    horizon: int = 5,
    cost_bps: float = 5.0,
    weight_fn=None,
    alerts: Optional[Sequence[str]] = None,
) -> List[FactorHealth]:
    """``factors``: name -> signed alpha values on ``prices``' index. Writes HTML; returns health rows."""
    extra = extra or {}
    health, cum, ric, decay, dd = [], {}, {}, {}, {}
    for name, alpha in factors.items():
        health.append(factor_health(name, alpha, prices, horizon=horizon, cost_bps=cost_bps))
        bt = run_backtest(alpha, prices, BacktestConfig(cost_bps=cost_bps, weight_fn=weight_fn))
        eq = (1 + bt.returns).cumprod()
        cum[name] = eq - 1
        dd[name] = eq / eq.cummax() - 1
        ric[name] = rolling_ic(alpha, prices["close"], horizon, 20)
        decay[name] = ic_decay_curve(alpha, prices["close"])
    counts = pd.Series([h.status for h in health]).value_counts().to_dict() if health else {}
    kpis = "".join(
        f'<div class="kpi"><span class="muted">{k}</span><b style="color:{STATUS_COLOR.get(k, "inherit")}">{counts.get(k, 0)}</b></div>'
        for k in ("healthy", "decaying", "dead")
    )
    alert_html = ""
    if alerts:
        alert_html = '<div class="card" style="margin-top:16px"><div class="ct">Alerts</div><ul>' + "".join(
            f"<li>{html.escape(a)}</li>" for a in alerts
        ) + "</ul></div>"
    body = f"""<main>
<h1>{html.escape(title)}</h1>
<div class="muted">Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC · {len(factors)} factors · horizon {horizon}d · costs {cost_bps:g} bps</div>
<div class="kpis">{kpis}</div>
{alert_html}
{_health_table(health, extra)}
<div class="grid2">
{_line_chart(cum, "Cumulative net return (long-short)", pct=True)}
{_line_chart(ric, f"Rolling 20-day IC ({horizon}d forward)")}
{_bar_chart(decay, "IC decay by horizon (days ahead)")}
{_line_chart(dd, "Drawdown", pct=True)}
</div></main>"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Factor Monitor</title><style>{CSS}</style></head><body>{body}</body></html>"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(page, encoding="utf-8")
    return health
