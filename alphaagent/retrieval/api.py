"""FastAPI retrieval service: ticker-filtered semantic search over the document index.

Run with ``alphaagent serve`` or ``uvicorn alphaagent.retrieval.api:app``.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from alphaagent.documents import DOC_TYPES, Document
from alphaagent.ingestion.pipeline import IngestionPipeline
from alphaagent.retrieval.retriever import Retriever


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=100)
    tickers: Optional[List[str]] = None
    doc_types: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = Field(None, description="Point-in-time cutoff (inclusive)")


class Hit(BaseModel):
    score: float
    chunk_id: str
    doc_id: str
    ticker: str
    doc_type: str
    published_at: str
    title: str
    source: str
    text: str
    position: int


class DocumentIn(BaseModel):
    ticker: str
    doc_type: str
    title: str
    text: str
    published_at: str
    source: str = ""
    metadata: dict = {}


def create_app(retriever: Retriever, pipeline: Optional[IngestionPipeline] = None, on_change=None) -> FastAPI:
    app = FastAPI(title="alphaagent retrieval API", version="0.1.0")
    lock = threading.Lock()

    def _search(req: SearchRequest) -> List[Hit]:
        if req.doc_types and set(req.doc_types) - set(DOC_TYPES):
            raise HTTPException(422, f"doc_types must be within {DOC_TYPES}")
        try:
            with lock:
                results = retriever.search(
                    req.query,
                    k=req.k,
                    tickers=req.tickers,
                    doc_types=req.doc_types,
                    start_date=req.start_date,
                    end_date=req.end_date,
                )
        except ValueError as e:
            raise HTTPException(422, str(e))
        return [Hit(**r.to_dict()) for r in results]

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(retriever.store)}

    @app.get("/tickers")
    def tickers():
        return retriever.store.ticker_counts()

    @app.post("/search", response_model=List[Hit])
    def search_post(req: SearchRequest):
        return _search(req)

    @app.get("/search", response_model=List[Hit])
    def search_get(
        q: str = Query(..., min_length=1),
        ticker: Optional[List[str]] = Query(None),
        doc_type: Optional[List[str]] = Query(None),
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        k: int = Query(5, ge=1, le=100),
    ):
        return _search(
            SearchRequest(query=q, k=k, tickers=ticker, doc_types=doc_type, start_date=start_date, end_date=end_date)
        )

    @app.get("/tickers/{ticker}/search", response_model=List[Hit])
    def search_ticker(ticker: str, q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=100)):
        return _search(SearchRequest(query=q, k=k, tickers=[ticker]))

    @app.post("/documents")
    def add_documents(docs: List[DocumentIn]):
        if pipeline is None:
            raise HTTPException(503, "ingestion disabled")
        try:
            parsed = [Document(**d.model_dump()) for d in docs]
        except ValueError as e:
            raise HTTPException(422, str(e))
        with lock:
            stats = pipeline.ingest(parsed)
            if on_change:
                on_change()
        return stats.__dict__

    return app


def _default_app() -> FastAPI:
    from alphaagent.config import get_settings
    from alphaagent.embeddings import build_embedder
    from alphaagent.vectorstore import FaissVectorStore

    s = get_settings()
    embedder = build_embedder(s.embedder, s.embed_device, s.embed_max_seq_length)
    store = FaissVectorStore.load_or_create(s.index_dir, embedder.dim)
    pipeline = IngestionPipeline(embedder, store)
    return create_app(Retriever(embedder, store), pipeline, on_change=lambda: store.save(s.index_dir))


def __getattr__(name):  # lazy module-level ``app`` for ``uvicorn alphaagent.retrieval.api:app``
    if name == "app":
        return _default_app()
    raise AttributeError(name)
