"""Command-line entry point: ``alphaagent <command>`` (or ``python -m alphaagent``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from alphaagent.config import get_settings


def _store_and_embedder(settings):
    from alphaagent.embeddings import build_embedder
    from alphaagent.vectorstore import FaissVectorStore

    embedder = build_embedder(settings.embedder, settings.embed_device, settings.embed_max_seq_length)
    store = FaissVectorStore.load_or_create(settings.index_dir, embedder.dim)
    return embedder, store


def cmd_ingest(args, settings) -> int:
    from alphaagent.ingestion import (
        IngestionPipeline,
        JSONLNewsSource,
        RSSNewsSource,
        SECFilingSource,
        TextChunker,
        TranscriptSource,
    )

    if args.source == "transcripts":
        source = TranscriptSource(args.path)
    elif args.source == "news":
        source = JSONLNewsSource(args.path) if args.path else RSSNewsSource.yahoo(args.tickers)
    elif args.source == "sec":
        source = SECFilingSource(
            args.tickers,
            user_agent=args.user_agent or settings.sec_user_agent,
            forms=args.forms,
            limit_per_ticker=args.limit,
            start_date=args.start_date,
        )
    else:  # pragma: no cover
        raise SystemExit(f"unknown source {args.source}")

    embedder, store = _store_and_embedder(settings)
    pipeline = IngestionPipeline(embedder, store, TextChunker(settings.chunk_size, settings.chunk_overlap))
    stats = pipeline.ingest(source.load())
    store.save(settings.index_dir)
    print(json.dumps({**stats.__dict__, "total_chunks": len(store)}, indent=2))
    return 0


def cmd_search(args, settings) -> int:
    from alphaagent.retrieval import Retriever

    embedder, store = _store_and_embedder(settings)
    results = Retriever(embedder, store).search(
        args.query,
        k=args.k,
        tickers=args.ticker,
        doc_types=args.doc_type,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    for r in results:
        c = r.chunk
        print(f"{r.score:.3f}  {c.ticker:<6} {c.doc_type:<11} {c.published_at}  {c.title}")
        print("       " + c.text.replace("\n", " ")[:240])
    return 0


def cmd_serve(args, settings) -> int:
    import os

    import uvicorn

    if args.with_ops:
        os.environ["ALPHA_WITH_OPS"] = "1"
    uvicorn.run("alphaagent.retrieval.api:app", host=args.host, port=args.port)
    return 0


def _ops_llm(settings):
    """Production LLM stack: provider client -> circuit breaker -> audit trail."""
    from alphaagent.llm import LLMConfig, build_llm
    from alphaagent.ops import AuditedLLM, AuditTrail, CircuitBreaker, ResilientLLM

    audit = AuditTrail(str(settings.data_dir / "audit" / "audit.jsonl"))
    llm = build_llm(LLMConfig(provider=settings.llm_provider, model_name=settings.llm_model))
    return AuditedLLM(ResilientLLM(llm, CircuitBreaker("llm", audit=audit)), audit)


def cmd_consume(args, settings) -> int:
    from alphaagent.features import TextSignalExtractor
    from alphaagent.ingestion import IngestionPipeline, TextChunker
    from alphaagent.streaming import IngestionConsumer, build_broker

    embedder, store = _store_and_embedder(settings)
    pipeline = IngestionPipeline(embedder, store, TextChunker(settings.chunk_size, settings.chunk_overlap))
    scorer = None
    if args.score:
        scorer = TextSignalExtractor(_ops_llm(settings), cache_path=str(settings.data_dir / "signals" / "doc_scores.jsonl"))
    consumer = IngestionConsumer(build_broker(settings.broker_url), pipeline, scorer,
                                 on_batch=lambda: store.save(settings.index_dir))
    if args.once:
        consumer.run_once()
    else:
        consumer.run_forever()
    print(json.dumps(consumer.stats.__dict__))
    return 0


def cmd_poll(args, settings) -> int:
    from alphaagent.ingestion import RSSNewsSource, SECFilingSource
    from alphaagent.streaming import DocumentProducer, build_broker, poll_sources

    sources = [RSSNewsSource.yahoo(args.tickers)]
    if settings.sec_user_agent:
        sources.append(SECFilingSource(args.tickers, settings.sec_user_agent, limit_per_ticker=args.limit))
    producer = DocumentProducer(build_broker(settings.broker_url))
    n = poll_sources(sources, producer, interval=args.interval, iterations=1 if args.once else None)
    print(json.dumps({"published": n}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alphaagent")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="ingest documents into the vector index")
    ing.add_argument("source", choices=["transcripts", "news", "sec"])
    ing.add_argument("--path", help="local directory/file (transcripts, news JSONL)")
    ing.add_argument("--tickers", nargs="+", default=[])
    ing.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    ing.add_argument("--limit", type=int, default=5, help="filings per ticker")
    ing.add_argument("--start-date")
    ing.add_argument("--user-agent", help="SEC User-Agent (overrides SEC_USER_AGENT)")
    ing.set_defaults(func=cmd_ingest)

    se = sub.add_parser("search", help="semantic search with optional ticker/date filters")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--ticker", nargs="+")
    se.add_argument("--doc-type", nargs="+")
    se.add_argument("--start-date")
    se.add_argument("--end-date")
    se.set_defaults(func=cmd_search)

    sv = sub.add_parser("serve", help="run the retrieval API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--with-ops", action="store_true", help="mount the operator control API under /ops")
    sv.set_defaults(func=cmd_serve)

    co = sub.add_parser("consume", help="Kafka ingestion consumer (docs.raw -> index [+ LLM scores])")
    co.add_argument("--score", action="store_true", help="also score documents with the LLM (signals.text)")
    co.add_argument("--once", action="store_true")
    co.set_defaults(func=cmd_consume)

    po = sub.add_parser("poll", help="poll SEC/RSS sources and publish new documents to docs.raw")
    po.add_argument("--tickers", nargs="+", required=True)
    po.add_argument("--interval", type=float, default=900)
    po.add_argument("--limit", type=int, default=3)
    po.add_argument("--once", action="store_true")
    po.set_defaults(func=cmd_poll)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return args.func(args, get_settings())


if __name__ == "__main__":
    sys.exit(main())
