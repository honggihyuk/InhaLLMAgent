"""Backtrader integration: replay target weights from an alpha through backtrader's broker simulation.

Useful as an independent cross-check of the vectorised engine (order execution, cash, commissions).
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from alphaagent.backtest.metrics import rank_weights


def run_backtrader(
    alpha: pd.Series,
    prices: pd.DataFrame,
    cash: float = 1_000_000.0,
    commission: float = 0.0005,
    weights: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    try:
        import backtrader as bt
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install backtrader to use the Backtrader adapter") from e

    target = weights if weights is not None else rank_weights(alpha.dropna())
    tickers = list(target.columns)

    class TargetWeights(bt.Strategy):
        def next(self):
            dt = pd.Timestamp(self.datas[0].datetime.date(0))
            if dt not in target.index:
                return
            row = target.loc[dt]
            for data in self.datas:
                self.order_target_percent(data=data, target=float(row.get(data._name, 0.0)))

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_shortcash(False)
    for t in tickers:
        px = prices.xs(t, level="ticker")[["open", "high", "low", "close", "volume"]].copy()
        px.index = pd.to_datetime(px.index)
        cerebro.adddata(bt.feeds.PandasData(dataname=px, name=t))
    cerebro.addstrategy(TargetWeights)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="ret")
    (strat,) = cerebro.run()
    rets = pd.Series(strat.analyzers.ret.get_analysis()).sort_index()
    from alphaagent.backtest.metrics import annualized_sharpe, max_drawdown

    return {
        "final_value": float(cerebro.broker.getvalue()),
        "total_return": float(cerebro.broker.getvalue() / cash - 1),
        "sharpe": annualized_sharpe(rets),
        "max_drawdown": max_drawdown(rets),
    }
