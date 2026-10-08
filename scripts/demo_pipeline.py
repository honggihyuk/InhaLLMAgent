"""End-to-end demo on the synthetic dataset.

    python scripts/demo_pipeline.py                 # offline, MockLLM
    python scripts/demo_pipeline.py --llm anthropic # real Claude calls (needs ANTHROPIC_API_KEY; costs money)
"""

import argparse
import logging

import pandas as pd

from alphaagent.data import create_synthetic_dataset
from alphaagent.embeddings import HashingEmbedder
from alphaagent.features import TextSignalExtractor, build_text_panel
from alphaagent.ingestion import IngestionPipeline
from alphaagent.library import FactorLibrary
from alphaagent.llm import LLMConfig, build_llm
from alphaagent.pipeline import AlphaGenerationPipeline, PipelineConfig
from alphaagent.retrieval import Retriever
from alphaagent.vectorstore import FaissVectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", choices=["mock", "anthropic", "vllm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--theme", default="earnings call tone and guidance revisions")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ideas", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=50)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--library", default="data/runs/factor_library.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = LLMConfig(provider=args.llm)
    if args.model:
        cfg.model_name = args.model
    llm = build_llm(cfg)

    prices, docs = create_synthetic_dataset(n_tickers=args.tickers, n_days=args.days)
    print(f"synthetic panel {prices.shape}, {len(docs)} documents")

    embedder = HashingEmbedder()
    store = FaissVectorStore(embedder.dim)
    IngestionPipeline(embedder, store).ingest(docs)
    retriever = Retriever(embedder, store)

    scores = TextSignalExtractor(llm, cache_path="data/runs/doc_scores.jsonl").score_documents(docs)
    text_panel = build_text_panel(scores, prices.index)

    pipe = AlphaGenerationPipeline(
        llm, retriever=retriever, library=FactorLibrary(args.library), config=PipelineConfig(num_ideas=args.ideas)
    )
    accepted = pipe.run(args.theme, prices, text_panel, rounds=args.rounds)

    pd.set_option("display.width", 220)
    print("\nFACTOR REPORT")
    print(pipe.get_factor_report().drop(columns=["theme"]).round(3).to_string(index=False))
    print(f"\naccepted: {[f.name for f in accepted]}")
    print(f"LLM calls: {llm.calls}, tokens in/out: {llm.total_input_tokens}/{llm.total_output_tokens}")


if __name__ == "__main__":
    main()
