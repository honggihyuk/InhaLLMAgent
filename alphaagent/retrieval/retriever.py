from __future__ import annotations

from typing import List, Optional, Sequence

from alphaagent.documents import SearchFilter, SearchResult
from alphaagent.embeddings.base import Embedder
from alphaagent.vectorstore.base import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        k: int = 5,
        tickers: Optional[Sequence[str]] = None,
        doc_types: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[SearchResult]:
        flt = SearchFilter(
            tickers=list(tickers) if tickers else None,
            doc_types=list(doc_types) if doc_types else None,
            start_date=start_date,
            end_date=end_date,
        )
        return self.store.search(self.embedder.embed_query(query), k=k, flt=flt)

    def context(self, query: str, k: int = 5, max_chars: int = 6000, **filters) -> str:
        """Retrieved passages formatted as numbered, cited context for an LLM prompt."""
        blocks, used = [], 0
        for i, r in enumerate(self.search(query, k=k, **filters), 1):
            c = r.chunk
            block = f"[{i}] ({c.ticker}, {c.doc_type}, {c.published_at}, {c.source})\n{c.text}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)
