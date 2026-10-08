"""FAISS-backed vector store with metadata filtering (ticker / doc type / point-in-time date)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import faiss
import numpy as np

from alphaagent.documents import Chunk, SearchFilter, SearchResult
from alphaagent.vectorstore.base import VectorStore


class FaissVectorStore(VectorStore):
    """Exact inner-product search (``IndexFlatIP``) behind an ``IndexIDMap2``.

    Filters are resolved to an id allow-list first, then passed to FAISS as an ``IDSelectorBatch``,
    so a ticker filter never returns fewer than ``k`` hits just because other tickers dominate.
    """

    INDEX_FILE = "index.faiss"
    META_FILE = "chunks.jsonl"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._chunks: Dict[int, Chunk] = {}
        self._id_by_chunk: Dict[str, int] = {}
        self._ids_by_doc: Dict[str, Set[int]] = defaultdict(set)
        self._ids_by_ticker: Dict[str, Set[int]] = defaultdict(set)
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def _register(self, fid: int, chunk: Chunk) -> None:
        self._chunks[fid] = chunk
        self._id_by_chunk[chunk.chunk_id] = fid
        self._ids_by_doc[chunk.doc_id].add(fid)
        self._ids_by_ticker[chunk.ticker].add(fid)
        self._next_id = max(self._next_id, fid + 1)

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> int:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32).reshape(len(chunks), -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != store dim {self.dim}")
        keep = [i for i, c in enumerate(chunks) if c.chunk_id not in self._id_by_chunk]
        if not keep:
            return 0
        ids = np.arange(self._next_id, self._next_id + len(keep), dtype=np.int64)
        self.index.add_with_ids(vectors[keep], ids)
        for fid, i in zip(ids.tolist(), keep):
            self._register(fid, chunks[i])
        return len(keep)

    def _allowed_ids(self, flt: SearchFilter) -> np.ndarray:
        if flt.tickers:
            candidates: Set[int] = set()
            for t in flt.tickers:
                candidates |= self._ids_by_ticker.get(t, set())
        else:
            candidates = set(self._chunks)
        return np.fromiter((i for i in candidates if flt.matches(self._chunks[i])), dtype=np.int64)

    def search(self, vector: np.ndarray, k: int = 5, flt: Optional[SearchFilter] = None) -> List[SearchResult]:
        if not self._chunks or k <= 0:
            return []
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        params = None
        if flt is not None and not flt.is_empty:
            allowed = self._allowed_ids(flt)
            if allowed.size == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
            k = min(k, int(allowed.size))
        k = min(k, len(self._chunks))
        scores, ids = self.index.search(query, k, params=params)
        return [
            SearchResult(chunk=self._chunks[int(i)], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def has_document(self, doc_id: str) -> bool:
        return bool(self._ids_by_doc.get(doc_id))

    def delete_document(self, doc_id: str) -> int:
        ids = self._ids_by_doc.pop(doc_id, set())
        if not ids:
            return 0
        self.index.remove_ids(np.fromiter(ids, dtype=np.int64))
        for fid in ids:
            chunk = self._chunks.pop(fid)
            self._id_by_chunk.pop(chunk.chunk_id, None)
            self._ids_by_ticker[chunk.ticker].discard(fid)
        return len(ids)

    def ticker_counts(self) -> Dict[str, int]:
        return dict(Counter(c.ticker for c in self._chunks.values()))

    def get_chunks(self, flt: Optional[SearchFilter] = None) -> List[Chunk]:
        if flt is None or flt.is_empty:
            return list(self._chunks.values())
        return [self._chunks[i] for i in self._allowed_ids(flt).tolist()]

    # ---- persistence -------------------------------------------------
    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # faiss' C++ writer can choke on non-ASCII paths on Windows; serialize in Python instead
        (directory / self.INDEX_FILE).write_bytes(faiss.serialize_index(self.index).tobytes())
        with open(directory / self.META_FILE, "w", encoding="utf-8") as fh:
            for fid, chunk in self._chunks.items():
                fh.write(json.dumps({"fid": fid, **chunk.to_dict()}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory) -> "FaissVectorStore":
        directory = Path(directory)
        raw = np.frombuffer((directory / cls.INDEX_FILE).read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(raw)
        store = cls(index.d)
        store.index = index
        with open(directory / cls.META_FILE, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    store._register(int(rec.pop("fid")), Chunk.from_dict(rec))
        return store

    @classmethod
    def load_or_create(cls, directory, dim: int) -> "FaissVectorStore":
        if (Path(directory) / cls.INDEX_FILE).exists():
            store = cls.load(directory)
            if store.dim != dim:
                raise ValueError(
                    f"index at {directory} has dim {store.dim} but embedder has dim {dim}; "
                    "use a separate ALPHA_DATA_DIR per embedder"
                )
            return store
        return cls(dim)
