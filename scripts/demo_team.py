"""Week 9-10 demo: the full multi-agent hierarchy on the synthetic dataset.

    python scripts/demo_team.py                  # offline (MockLLM)
    python scripts/demo_team.py --llm anthropic  # real Claude calls (needs ANTHROPIC_API_KEY; costs money)
"""

import argparse
import json

import pandas as pd

from alphaagent.agents.ensemble import LABELS
from alphaagent.data import create_synthetic_dataset, synthetic_sectors
from alphaagent.embeddings import HashingEmbedder
from alphaagent.features import TextSignalExtractor, build_text_panel
from alphaagent.ingestion import IngestionPipeline
from alphaagent.llm import LLMConfig, build_llm
from alphaagent.pipeline import PipelineConfig
from alphaagent.retrieval import Retriever
from alphaagent.risk import RiskLimits
from alphaagent.team import ResearchTeam
from alphaagent.vectorstore import FaissVectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", choices=["mock", "anthropic", "vllm"])
    ap.add_argument("--goal", default="Mine tradable alpha from earnings calls, filings and news")
    ap.add_argument("--tasks", type=int, default=3)
    args = ap.parse_args()

    llm = build_llm(LLMConfig(provider=args.llm))
    prices, docs = create_synthetic_dataset(n_tickers=40, n_days=300)
    emb = HashingEmbedder()
    store = FaissVectorStore(emb.dim)
    IngestionPipeline(emb, store).ingest(docs)
    text = build_text_panel(TextSignalExtractor(llm).score_documents(docs), prices.index)
    sectors = synthetic_sectors(prices.index.get_level_values("ticker").unique())

    team = ResearchTeam(llm, retriever=Retriever(emb, store), sectors=sectors, risk_limits=RiskLimits(max_position=0.05),
                        pipeline_config=PipelineConfig(num_ideas=2, lookahead_cutoffs=1))
    res = team.research(args.goal, prices, text, max_tasks=args.tasks)

    print("PLAN")
    for t in res.plan:
        print(f"  {t.task_id} [{t.specialty}] {t.theme}")
    print("\nANALYST RESULTS")
    for r in res.results:
        print(f"  {r['task_id']}: {[a['name'] for a in r['accepted']]}")
    print("\nSTRATEGY", json.dumps({k: v for k, v in res.strategy.items() if k != "summary"}, default=str, indent=1)[:1200])
    print("\nFACTOR BLEND", {k: round(v, 3) for k, v in res.factor_weights.items()})
    if res.risk_report:
        m = res.risk_report.metrics
        print(f"\nRISK action={res.risk_report.action} scale={res.risk_report.scale:.2f} "
              f"gross={m['gross']:.2f} vol20d={m['vol_20d']:.1%} VaR95={m['var95']:.2%} dd={m['drawdown']:.1%}")
        print(f"book: annualised return {res.pnl.mean() * 252:.1%}, vol {res.pnl.std() * 252 ** 0.5:.1%}")
    if res.decisions is not None:
        counts = res.decisions.map(LABELS).value_counts().to_dict()
        print("\nENSEMBLE latest decisions", counts, "voter weights", {k: round(v, 3) for k, v in res.ensemble_weights.items()})
    print(f"\nmessages on the bus: {len(team.bus.history)}, memory entries: {len(team.memory)}, LLM calls: {llm.calls}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
