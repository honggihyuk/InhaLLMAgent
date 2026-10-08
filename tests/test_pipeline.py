import numpy as np
import pandas as pd
import pytest

from alphaagent.backtest import BacktestConfig, run_backtest
from alphaagent.data import create_synthetic_dataset, create_synthetic_market_data
from alphaagent.documents import Document
from alphaagent.features import DocumentScore, TextSignalExtractor, build_text_panel
from alphaagent.library import FactorLibrary
from alphaagent.llm import MockLLM
from alphaagent.llm.mock_handlers import default_handlers
from alphaagent.pipeline import AlphaGenerationPipeline, PipelineConfig
from alphaagent.sandbox import AlphaSandbox, check_alpha_code
from alphaagent.validation import (
    check_lookahead,
    compute_known_factors,
    orthogonalize,
    static_scan,
    truncation_test,
)

GOOD = '''def compute_alpha(df):
    return -df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())'''


@pytest.fixture(scope="module")
def prices():
    return create_synthetic_market_data(20, 80, seed=3)


# ---- sandbox ------------------------------------------------------------
@pytest.mark.parametrize(
    "code,needle",
    [
        ("import os\ndef compute_alpha(df):\n    return df['close']", "imports"),
        ("def compute_alpha(df):\n    open('x','w')\n    return df['close']", "'open'"),
        ("def compute_alpha(df):\n    return df.__class__", "dunder"),
        ("def compute_alpha(df):\n    df.to_csv('x')\n    return df['close']", ".to_csv"),
        ("def other(df):\n    return df", "no function named compute_alpha"),
        ("x = print('side effect')\ndef compute_alpha(df):\n    return df['close']", None),
        ("print('hi')\ndef compute_alpha(df):\n    return df['close']", "module level"),
    ],
)
def test_static_policy(code, needle):
    res = check_alpha_code(code)
    if needle is None:
        assert res.ok
    else:
        assert not res.ok and any(needle in e for e in res.errors), res.errors


def test_sandbox_runs_valid_code(prices):
    res = AlphaSandbox().run(GOOD, prices)
    assert res.ok, res.error
    assert res.values.index.equals(prices.index) and res.values.notna().mean() > 0.8


def test_sandbox_reports_runtime_errors_and_bad_outputs(prices):
    sbx = AlphaSandbox()
    err = sbx.run("def compute_alpha(df):\n    return df['analyst_sentiment']", prices)
    assert not err.ok and "analyst_sentiment" in err.error
    const = sbx.run("def compute_alpha(df):\n    return df['close'] * 0 + 1", prices)
    assert not const.ok and "constant" in const.error
    sparse = sbx.run("def compute_alpha(df):\n    return df['close'].where(df['close'] < 0)", prices)
    assert not sparse.ok and "covers only" in sparse.error


def test_sandbox_restricted_builtins_and_timeout(prices):
    # __import__ is blocked statically; even if smuggled past the AST check, builtins are stripped
    sneaky = "def compute_alpha(df):\n    b = df.close\n    return b"
    assert AlphaSandbox().run(sneaky, prices).ok
    slow = "def compute_alpha(df):\n    x = 0\n    for i in range(10**10):\n        x += i\n    return df['close']"
    res = AlphaSandbox(timeout=3).run(slow, prices)
    assert not res.ok and "timeout" in res.error


# ---- look-ahead ------------------------------------------------------------
@pytest.mark.parametrize(
    "snippet",
    [
        'df.groupby(level="ticker")["close"].shift(-1)',
        'df.groupby(level="ticker")["close"].pct_change(periods=-5)',
        'df["close"].bfill()',
        'df["close"].fillna(method="bfill")',
        'df.groupby(level="ticker")["close"].transform(lambda s: s.rolling(5, center=True).mean())',
    ],
)
def test_static_lookahead_patterns(snippet):
    rep = static_scan(f"def compute_alpha(df):\n    return {snippet}")
    assert rep.static_errors


def test_full_sample_normalisation_is_caught_dynamically(prices):
    leaky = '''def compute_alpha(df):
    r = df["returns"]
    return (r - r.mean()) / r.std()'''
    rep = static_scan(leaky)
    assert not rep.static_errors and rep.static_warnings  # only a warning statically...
    sbx = AlphaSandbox()
    report = check_lookahead(leaky, compute=lambda d: sbx.run(leaky, d).values, data=prices, n_cutoffs=1)
    assert report.dynamic_errors  # ...but the truncation test proves the leak


def test_truncation_test_passes_causal_and_fails_future_peeking(prices):
    causal = lambda d: -d.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())
    peeking = lambda d: d.groupby(level="ticker")["returns"].shift(-1)
    assert truncation_test(causal, prices, n_cutoffs=2) == []
    assert truncation_test(peeking, prices, n_cutoffs=2)


# ---- orthogonalisation & backtest ----------------------------------------
def test_orthogonalize_flags_disguised_known_factor():
    df = create_synthetic_market_data(30, 200, seed=5)
    known = compute_known_factors(df)
    disguised = -2.5 * df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum()) + 3
    res = orthogonalize(disguised, known)
    assert res.closest_factor == "short_term_reversal" and res.max_abs_corr > 0.95
    assert res.residual.abs().mean() < 0.05


def test_backtest_has_no_same_day_leak():
    df = create_synthetic_market_data(30, 250, seed=9, planted_signal=None)
    # today's return is known at t; with proper lagging it must not predict anything
    res = run_backtest(df["returns"], df, BacktestConfig(execution_lag=1, cost_bps=0))
    assert abs(res.stats["sharpe"]) < 1.5
    # a "perfect foresight" alpha only looks good if the engine were leaking
    future = df.groupby(level="ticker")["returns"].shift(-2)
    assert run_backtest(future, df, BacktestConfig(execution_lag=1, cost_bps=0)).stats["sharpe"] > 5


def test_backtest_costs_reduce_returns():
    df = create_synthetic_market_data(20, 120, seed=2)
    alpha = -df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())
    free = run_backtest(alpha, df, BacktestConfig(cost_bps=0)).stats["annual_return"]
    costly = run_backtest(alpha, df, BacktestConfig(cost_bps=20)).stats["annual_return"]
    assert costly < free


def test_backtrader_adapter_runs():
    pytest.importorskip("backtrader")
    from alphaagent.backtest.backtrader_adapter import run_backtrader

    df = create_synthetic_market_data(5, 40, seed=4)
    alpha = -df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())
    out = run_backtrader(alpha, df)
    assert out["final_value"] > 0 and np.isfinite(out["sharpe"])


# ---- text features -----------------------------------------------------------
def test_text_panel_is_point_in_time_and_decays():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2026-01-05", periods=10), ["AAA", "BBB"]], names=["date", "ticker"])
    scores = [DocumentScore("d1", "AAA", "2026-01-06", "transcript", sentiment=1.0)]
    panel = build_text_panel(scores, idx, half_life=2, max_age=5)
    s = panel["sentiment"].xs("AAA", level="ticker")
    assert s.loc["2026-01-06"] == 0.0  # published that day: not tradable until the next session
    assert s.loc["2026-01-07"] == pytest.approx(1.0)
    assert s.loc["2026-01-09"] == pytest.approx(0.5)
    assert s.loc["2026-01-15"] == 0.0  # expired after max_age
    assert (panel["sentiment"].xs("BBB", level="ticker") == 0).all()


def test_text_scoring_extracts_tone_and_caches(tmp_path):
    llm = MockLLM()
    ex = TextSignalExtractor(llm, cache_path=str(tmp_path / "scores.jsonl"))
    doc = Document(ticker="ACME", doc_type="transcript", title="call", published_at="2026-01-02",
                   text="Record revenue and strong demand. We are raising full-year guidance.")
    s = ex.score_document(doc)
    assert s.sentiment > 0.5 and s.guidance == 1.0
    ex.score_document(doc)
    assert llm.calls == 1
    assert TextSignalExtractor(llm, cache_path=str(tmp_path / "scores.jsonl")).cache[doc.doc_id].sentiment == s.sentiment


# ---- end-to-end ------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset():
    prices, docs = create_synthetic_dataset(n_tickers=30, n_days=260, seed=7)
    llm = MockLLM()
    text = build_text_panel(TextSignalExtractor(llm).score_documents(docs), prices.index)
    return prices, text


def test_pipeline_finds_text_alpha_and_rejects_known_factors(dataset, tmp_path):
    prices, text = dataset
    pipe = AlphaGenerationPipeline(MockLLM(), library=FactorLibrary(str(tmp_path / "lib.json")),
                                   config=PipelineConfig(num_ideas=7, lookahead_cutoffs=1))
    accepted = pipe.run_pipeline("earnings call tone", prices, text)
    report = pipe.get_factor_report().set_index("name")
    assert "sentiment_drift" in {f.name for f in accepted}
    assert report.loc["short_term_reversal", "status"] == "rejected"
    assert report.loc["short_term_reversal", "MaxKnownCorr"] > 0.9
    assert FactorLibrary(str(tmp_path / "lib.json")).factors["sentiment_drift"].status == "accepted"
    assert pipe.pipeline_history[0]["accepted"] == len(accepted)


def test_pipeline_repairs_lookahead_code(dataset):
    prices, text = dataset
    handlers = default_handlers()
    leaky = '```python\ndef compute_alpha(df):\n    return df.groupby(level="ticker")["returns"].shift(-1)\n```\n[Confidence: 0.9]'
    handlers["implementation"] = lambda p, s: leaky  # first attempt peeks into the future
    llm = MockLLM(handlers)
    pipe = AlphaGenerationPipeline(llm, config=PipelineConfig(num_ideas=1, lookahead_cutoffs=1, max_repairs=1))
    pipe.run_pipeline("price reversal", prices)
    (factor,) = pipe.library.factors.values()
    assert "Look-ahead" in factor.errors[0] and ".shift(-1)" in factor.errors[0]
    assert factor.status in {"accepted", "rejected"}  # repaired code passed validation and was evaluated
    assert any("TASK_KIND: repair" in p for p in llm.prompts)


def test_pipeline_marks_unfixable_code_invalid(dataset):
    prices, _ = dataset
    handlers = default_handlers()
    broken = '```python\nimport os\ndef compute_alpha(df):\n    return df["close"]\n```'
    handlers["implementation"] = handlers["repair"] = lambda p, s: broken
    pipe = AlphaGenerationPipeline(MockLLM(handlers), config=PipelineConfig(num_ideas=1, max_repairs=1))
    assert pipe.run_pipeline("anything", prices) == []
    (factor,) = pipe.library.factors.values()
    assert factor.status == "invalid" and len(factor.errors) == 2
