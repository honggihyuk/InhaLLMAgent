"""Paragraph/sentence-aware text chunker with character overlap."""

from __future__ import annotations

import re
from typing import List

from alphaagent.documents import Chunk, Document, stable_id

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TextChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _units(self, text: str) -> List[str]:
        """Split into paragraphs; break paragraphs longer than chunk_size into sentences (then hard slices)."""
        units: List[str] = []
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if not para:
                continue
            if len(para) <= self.chunk_size:
                units.append(para)
                continue
            for sent in _SENTENCE_SPLIT.split(para):
                while len(sent) > self.chunk_size:
                    units.append(sent[: self.chunk_size])
                    sent = sent[self.chunk_size - self.chunk_overlap :]
                if sent:
                    units.append(sent)
        return units

    def split_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        current = ""
        for unit in self._units(text):
            candidate = f"{current}\n{unit}" if current else unit
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                # start the overlap at a word boundary
                tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
                current = f"{tail}\n{unit}" if tail and len(tail) + len(unit) + 1 <= self.chunk_size else unit
            else:
                current = unit
        if current:
            chunks.append(current)
        return chunks

    def chunk(self, doc: Document) -> List[Chunk]:
        prefix = f"[{doc.ticker} | {doc.doc_type} | {doc.published_at}] {doc.title}".strip()
        out = []
        for i, piece in enumerate(self.split_text(doc.text)):
            out.append(
                Chunk(
                    chunk_id=stable_id(doc.doc_id or "", str(i)),
                    doc_id=doc.doc_id or "",
                    ticker=doc.ticker,
                    doc_type=doc.doc_type,
                    published_at=doc.published_at,
                    title=doc.title,
                    source=doc.source,
                    # header gives the embedder entity/date context that the raw passage may lack
                    text=f"{prefix}\n{piece}",
                    position=i,
                )
            )
        return out
