from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from alphaagent.documents import Chunk, SearchFilter, SearchResult


class VectorStore(ABC):
    """Prototype backend is FAISS; Pinecone/Weaviate adapters implement the same interface."""

    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> int:
        """Add chunks (skipping known chunk_ids); return the number actually added."""

    @abstractmethod
    def search(self, vector: np.ndarray, k: int = 5, flt: Optional[SearchFilter] = None) -> List[SearchResult]:
        ...

    @abstractmethod
    def has_document(self, doc_id: str) -> bool:
        ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        ...

    @abstractmethod
    def ticker_counts(self) -> Dict[str, int]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
