"""Week 7-8: run the research pipeline point-in-time over 12+ months, validate, and build the monitor.

    python scripts/historical_validation.py                     # synthetic data, MockLLM (offline)
    python scripts/historical_validation.py --prices px.csv     # your own price panel (date,ticker,ohlcv)

Outputs (under --out, default data/runs/historical):
    reruns.csv        accepted factors at each monthly as-of date and their IC over the following month
    factor_report.csv library of every factor ever generated with all validation metrics
    monitor.html      factor decay / performance dashboard
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alphaagent.data import create_synthetic_dataset, load_prices_csv, synthetic_sectors
from alphaagent.features import TextSignalExtractor, build_text_panel
from alphaagent.library import FactorLibrary
from alphaagent.llm import LLMConfig, build_llm
from alphaagent.monitoring import build_dashboard
from alphaagent.pipeline import AlphaGenerationPipeline, PipelineConfig
from alphaagent.risk import RiskLimits, RiskOverlay
from alphaagent.validation import historical_reruns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", choices=["mock", "anthropic", "vllm"])
    ap.add_argument("--prices", help="CSV price panel; default is the synthetic dataset")
    ap.add_argument("--days", type=int, default=440, help="synthetic history length (trading days)")
    ap.add_argument("--tickers", type=int, default=50)
    ap.add_argument("--first-run-day", type=int, default=189, help="trading days of history before the first run")
    ap.add_argument("--step", type=int, default=21, help="rerun every N trading days")
    ap.add_argument("--ideas", type=int, default=3)
    ap.add_argument("--theme", default="earnings call tone and guidance revisions")
    ap.add_argument("--out", default="data/runs/historical")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    llm = build_llm(LLMConfig(provider=args.llm))
    if args.prices:
        prices, text = load_prices_csv(args.prices), None
    else:
        prices, docs = create_synthetic_dataset(n_tickers=args.tickers, n_days=args.days)
        scores = TextSignalExtractor(llm, cache_path=str(out / "doc_scores.jsonl")).score_documents(docs)
        text = build_text_panel(scores, prices.index)
    dates = prices.index.get_level_values("date").unique()
    print(f"history: {dates[0].date()} .. {dates[-1].date()} ({len(dates)} trading days)")

    sectors = synthetic_sectors(prices.index.get_level_values("ticker").unique())
    overlay = RiskOverlay(RiskLimits(max_position=0.04, sector_neutral=True), sectors)
    library = FactorLibrary(str(out / "factor_library.json"))
    pipe = AlphaGenerationPipeline(
        llm, library=library, weight_fn=overlay, config=PipelineConfig(num_ideas=args.ideas, lookahead_cutoffs=1)
    )

    runs = historical_reruns(pipe, args.theme, prices, text, start_days=args.first_run_day, step_days=args.step)
    rows = [{"as_of": r.as_of.date(), "new": ",".join(r.accepted), "live": len(r.live),
             **{f"fwd_ic:{k}": v for k, v in r.forward_ic.items()}} for r in runs]
    reruns = pd.DataFrame(rows)
    reruns.to_csv(out / "reruns.csv", index=False)
    library.report().to_csv(out / "factor_report.csv", index=False)

    fwd = [v for r in runs for v in r.forward_ic.values() if v == v]
    print(f"\n{len(runs)} point-in-time reruns; live factor-months: {len(fwd)}; "
          f"mean next-month IC of live factors: {np.mean(fwd) if fwd else float('nan'):.4f}; "
          f"positive months: {np.mean([v > 0 for v in fwd]) if fwd else 0:.0%}")
    print(reruns.to_string(index=False))

    accepted = {f.name: f for f in library.accepted()}
    values = {name: pipe.factor_values(f, prices, text) for name, f in accepted.items()}
    extra = {name: f.metrics for name, f in accepted.items()}
    health = build_dashboard(values, prices, str(out / "monitor.html"), extra=extra, weight_fn=overlay,
                             title="Alpha factor monitor (point-in-time research reruns)")
    print("\nfactor health:")
    for h in health:
        print(f"  {h.name:<32} {h.status:<10} IC all {h.full_ic:+.3f}  60d {h.ic_60d:+.3f}  half-life {h.half_life_days}")
    print(f"\ndashboard: {out / 'monitor.html'}")


if __name__ == "__main__":
    main()
