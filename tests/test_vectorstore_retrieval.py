import numpy as np
from fastapi.testclient import TestClient

from alphaagent.documents import Document
from alphaagent.ingestion import IngestionPipeline, TranscriptSource
from alphaagent.retrieval.api import create_app
from alphaagent.vectorstore import FaissVectorStore

from .conftest import SAMPLES


def test_hashing_embedder_is_normalized_and_deterministic(embedder):
    a = embedder.embed_documents(["gross margin expanded", "gross margin expanded"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], a[1])


def test_ticker_filter_returns_only_that_ticker(loaded):
    retriever, _ = loaded
    hits = retriever.search("margin guidance", k=5, tickers=["GLBX"])
    assert hits and {h.chunk.ticker for h in hits} == {"GLBX"}
    # unfiltered search reaches more than one ticker
    assert len({h.chunk.ticker for h in retriever.search("margin guidance", k=10)}) > 1


def test_point_in_time_and_doc_type_filters(loaded):
    retriever, _ = loaded
    hits = retriever.search("Globex", k=10, tickers=["GLBX"], end_date="2026-02-28")
    assert hits and all(h.chunk.published_at <= "2026-02-28" for h in hits)
    news = retriever.search("Globex", k=10, doc_types=["news"])
    assert news and all(h.chunk.doc_type == "news" for h in news)
    assert retriever.search("anything", tickers=["NOPE"]) == []


def test_relevant_passage_ranks_first(loaded):
    retriever, _ = loaded
    top = retriever.search("paused the buyback to preserve liquidity", k=1)[0]
    assert top.chunk.ticker == "GLBX"


def test_ingest_is_idempotent(loaded):
    retriever, pipeline = loaded
    before = len(retriever.store)
    stats = pipeline.ingest(TranscriptSource(SAMPLES / "transcripts").load())
    assert stats.documents_added == 0 and len(retriever.store) == before


def test_save_load_roundtrip(loaded, tmp_path, embedder):
    retriever, _ = loaded
    retriever.store.save(tmp_path)
    restored = FaissVectorStore.load(tmp_path)
    assert len(restored) == len(retriever.store)
    q = embedder.embed_query("record free cash flow")
    assert [r.chunk.chunk_id for r in restored.search(q, 3)] == [
        r.chunk.chunk_id for r in retriever.store.search(q, 3)
    ]


def test_delete_document(store, embedder):
    pipe = IngestionPipeline(embedder, store)
    doc = Document(ticker="A", doc_type="news", title="t", text="hello", published_at="2026-01-01")
    pipe.ingest([doc])
    assert store.has_document(doc.doc_id)
    assert store.delete_document(doc.doc_id) == 1
    assert len(store) == 0 and store.index.ntotal == 0


def test_api_search_and_ingest(loaded):
    retriever, pipeline = loaded
    client = TestClient(create_app(retriever, pipeline))
    assert client.get("/health").json()["chunks"] == len(retriever.store)
    assert set(client.get("/tickers").json()) == {"ACME", "GLBX", "INIT"}

    r = client.get("/search", params={"q": "restructuring headcount", "ticker": "INIT", "k": 3})
    assert r.status_code == 200 and {h["ticker"] for h in r.json()} == {"INIT"}

    r = client.post("/search", json={"query": "backlog", "tickers": ["ACME"], "end_date": "2026-01-28"})
    assert r.status_code == 200 and all(h["published_at"] <= "2026-01-28" for h in r.json())

    assert client.post("/search", json={"query": "x", "doc_types": ["tweet"]}).status_code == 422

    new = {"ticker": "NEWCO", "doc_type": "news", "title": "NewCo IPO", "text": "NewCo priced its IPO", "published_at": "2026-03-01"}
    assert client.post("/documents", json=[new]).json()["documents_added"] == 1
    assert client.get("/tickers/NEWCO/search", params={"q": "IPO"}).json()[0]["ticker"] == "NEWCO"
