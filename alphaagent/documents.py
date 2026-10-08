"""Core document and chunk types shared by ingestion, storage, and retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Optional, Union

DOC_TYPES = ("transcript", "sec_filing", "news")


def normalize_date(value: Union[str, date, datetime, None]) -> str:
    """Return an ISO ``YYYY-MM-DD`` string. Dates drive point-in-time filtering, so they are mandatory."""
    if value is None or value == "":
        raise ValueError("published_at is required for point-in-time retrieval")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return datetime.fromisoformat(str(value).strip()[:10]).date().isoformat()


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class Document:
    ticker: str
    doc_type: str
    title: str
    text: str
    published_at: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()
        if self.doc_type not in DOC_TYPES:
            raise ValueError(f"doc_type must be one of {DOC_TYPES}, got {self.doc_type!r}")
        self.published_at = normalize_date(self.published_at)
        if not self.doc_id:
            self.doc_id = stable_id(self.ticker, self.doc_type, self.published_at, self.source or self.title)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    doc_type: str
    published_at: str
    title: str
    source: str
    text: str
    position: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class SearchFilter:
    """Metadata filter. ``end_date`` doubles as the point-in-time cutoff to avoid look-ahead bias."""

    tickers: Optional[list] = None
    doc_types: Optional[list] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tickers:
            self.tickers = [t.upper().strip() for t in self.tickers]
        if self.start_date:
            self.start_date = normalize_date(self.start_date)
        if self.end_date:
            self.end_date = normalize_date(self.end_date)

    @property
    def is_empty(self) -> bool:
        return not (self.tickers or self.doc_types or self.start_date or self.end_date)

    def matches(self, chunk: Chunk) -> bool:
        if self.tickers and chunk.ticker not in self.tickers:
            return False
        if self.doc_types and chunk.doc_type not in self.doc_types:
            return False
        if self.start_date and chunk.published_at < self.start_date:
            return False
        if self.end_date and chunk.published_at > self.end_date:
            return False
        return True


@dataclass
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, **self.chunk.to_dict()}
