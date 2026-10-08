"""Source -> chunk -> embed -> vector store, with doc-level de-duplication."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

from alphaagent.documents import Chunk, Document
from alphaagent.embeddings.base import Embedder
from alphaagent.ingestion.chunking import TextChunker
from alphaagent.vectorstore.base import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents_seen: int = 0
    documents_added: int = 0
    documents_skipped: int = 0
    chunks_added: int = 0


class IngestionPipeline:
    def __init__(
        self, embedder: Embedder, store: VectorStore, chunker: TextChunker = None, batch_size: int = 32
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker or TextChunker()
        self.batch_size = batch_size

    def _flush(self, buffer: List[Chunk], stats: IngestStats) -> None:
        if not buffer:
            return
        vectors = self.embedder.embed_documents([c.text for c in buffer])
        stats.chunks_added += self.store.add(buffer, vectors)
        buffer.clear()

    def ingest(self, documents: Iterable[Document]) -> IngestStats:
        stats = IngestStats()
        buffer: List[Chunk] = []
        for doc in documents:
            stats.documents_seen += 1
            if self.store.has_document(doc.doc_id) or not doc.text.strip():
                stats.documents_skipped += 1
                continue
            buffer.extend(self.chunker.chunk(doc))
            stats.documents_added += 1
            if len(buffer) >= self.batch_size:
                self._flush(buffer, stats)
        self._flush(buffer, stats)
        log.info("ingest: %s", stats)
        return stats
