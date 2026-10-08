import numpy as np
import pandas as pd
import pytest

from alphaagent.agents.types import AlphaFactor
from alphaagent.backtest import BacktestConfig, run_backtest
from alphaagent.data import create_synthetic_dataset, create_synthetic_market_data, load_prices_csv, synthetic_sectors
from alphaagent.features import TextSignalExtractor, build_text_panel
from alphaagent.llm import MockLLM
from alphaagent.monitoring import build_dashboard, factor_health, ic_decay_curve, ic_half_life
from alphaagent.risk import RiskLimits, RiskOverlay
from alphaagent.validation import historical_reruns, known_factor_returns, spanning_test, walk_forward


def reversal(df):
    return -df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())


@pytest.fixture(scope="module")
def text_data():
    prices, docs = create_synthetic_dataset(n_tickers=40, n_days=320, seed=7)
    text = build_text_panel(TextSignalExtractor(MockLLM()).score_documents(docs), prices.index)
    return prices, text


# ---- risk overlay ------------------------------------------------------------
def test_risk_overlay_enforces_limits():
    df = create_synthetic_market_data(40, 60, seed=1)
    tickers = df.index.get_level_values("ticker").unique()
    sectors = synthetic_sectors(tickers, 4)
    overlay = RiskOverlay(RiskLimits(max_position=0.04, max_gross=1.0, sector_neutral=True), sectors)
    res = run_backtest(reversal(df), df, BacktestConfig(weight_fn=overlay))
    w = res.weights[res.weights.abs().sum(axis=1) > 0]
    report = overlay.check(w)
    assert report["max_abs_position"] <= 0.04 + 1e-9
    assert report["max_gross"] <= 1.0 + 1e-9
    assert report["max_abs_sector_net"] < 1e-6
    assert (w.abs().sum(axis=1) > 0.9).mean() > 0.9  # still (nearly) fully invested


# ---- validation -----------------------------------------------------------------
def test_walk_forward_stability():
    sig = create_synthetic_market_data(40, 300, seed=2, planted_signal="reversal")
    noise = create_synthetic_market_data(40, 300, seed=2, planted_signal=None)
    wf_sig = walk_forward(reversal(sig), sig["close"], 126, 21, 1)
    wf_noise = walk_forward(reversal(noise), noise["close"], 126, 21, 1)
    assert len(wf_sig.windows) >= 7
    assert wf_sig.positive_windows > 0.8 and wf_sig.oos_ic_tstat > 3
    assert abs(wf_noise.mean_oos_ic) < abs(wf_sig.mean_oos_ic) / 2


def test_spanning_test_separates_new_alpha_from_known_exposure():
    df = create_synthetic_market_data(30, 250, seed=3)
    known = known_factor_returns(df)
    rng = np.random.default_rng(0)
    idx = known.dropna().index
    mimic = 0.8 * known.loc[idx, "low_volatility"] + rng.normal(0, 1e-4, len(idx))
    novel = mimic + 0.002  # 0.2% a day not explained by any known factor
    assert abs(spanning_test(mimic, known).alpha_tstat) < 2
    res = spanning_test(novel, known)
    assert res.alpha_tstat > 5 and res.betas["low_volatility"] == pytest.approx(0.8, abs=0.05)


def test_historical_reruns_are_point_in_time(text_data):
    prices, text = text_data
    seen = []

    class SpyPipeline:
        def run_pipeline(self, theme, price_data, text_panel=None, as_of=None):
            seen.append(as_of)
            f = AlphaFactor("sent", "code", metrics={"sign": 1.0})
            f.status = "accepted"
            return [f]

        def factor_values(self, factor, price_data, text_panel=None):
            assert price_data.index.get_level_values("date").max() > pd.Timestamp(seen[-1])  # scored on the future
            return text_panel["sentiment"]

    runs = historical_reruns(SpyPipeline(), "t", prices, text, start_days=200, step_days=21)
    assert len(runs) == 5 and seen == sorted(seen)
    assert np.nanmean([r.forward_ic["sent"] for r in runs]) > 0


# ---- monitoring -----------------------------------------------------------------
def test_decay_curve_and_half_life(text_data):
    prices, text = text_data
    curve = ic_decay_curve(text["sentiment"], prices["close"], horizons=(1, 2, 3, 5, 10, 15, 20))
    assert curve[1] > 0 and curve[20] < curve[1] / 2  # drift lasts ~10 days, then the signal is spent
    hl = ic_half_life(curve)
    assert hl is not None and 2 <= hl <= 20
    assert ic_half_life(pd.Series({1: 0.1, 2: 0.08, 3: 0.04})) == pytest.approx(2.75)


def test_factor_health_flags_dying_signal():
    df = create_synthetic_market_data(40, 300, seed=4, planted_signal="reversal")
    alpha = reversal(df)
    dates = df.index.get_level_values("date").unique()
    late = df.index.get_level_values("date") >= dates[200]
    rng = np.random.default_rng(1)
    dying = alpha.copy()
    dying[late] = rng.normal(size=late.sum())  # signal replaced by noise in the last third
    assert factor_health("ok", alpha, df).status == "healthy"
    assert factor_health("dying", dying, df).status in {"decaying", "dead"}


def test_dashboard_html(tmp_path, text_data):
    prices, text = text_data
    out = tmp_path / "monitor.html"
    health = build_dashboard(
        {"sentiment": text["sentiment"], "reversal": reversal(prices)},
        prices,
        str(out),
        extra={"sentiment": {"max_known_corr": 0.1}},
        alerts=["reversal: rolling IC below zero for 20 days"],
    )
    page = out.read_text(encoding="utf-8")
    assert len(health) == 2 and page.count("<svg") == 4
    assert "sentiment" in page and "prefers-color-scheme: dark" in page and "Alerts" in page


# ---- data loaders --------------------------------------------------------------
def test_load_prices_csv(tmp_path):
    rows = []
    for d in pd.bdate_range("2026-01-05", periods=3):
        for t, px in (("aaa", 10.0), ("bbb", 20.0)):
            rows.append({"date": d, "ticker": t, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 100})
    path = tmp_path / "px.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    panel = load_prices_csv(str(path))
    assert list(panel.columns[:8]) == ["open", "high", "low", "close", "volume", "vwap", "returns", "cap"]
    assert set(panel.index.get_level_values("ticker")) == {"AAA", "BBB"}
    assert panel["returns"].dropna().eq(0).all()
