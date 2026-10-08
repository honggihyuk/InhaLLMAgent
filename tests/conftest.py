from pathlib import Path

import pytest

from alphaagent.embeddings import HashingEmbedder
from alphaagent.ingestion import IngestionPipeline, JSONLNewsSource, TextChunker, TranscriptSource
from alphaagent.retrieval import Retriever
from alphaagent.vectorstore import FaissVectorStore

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


@pytest.fixture
def store(embedder):
    return FaissVectorStore(embedder.dim)


@pytest.fixture
def loaded(embedder, store):
    pipeline = IngestionPipeline(embedder, store, TextChunker(400, 80))
    pipeline.ingest(TranscriptSource(SAMPLES / "transcripts").load())
    pipeline.ingest(JSONLNewsSource(SAMPLES / "news").load())
    return Retriever(embedder, store), pipeline
