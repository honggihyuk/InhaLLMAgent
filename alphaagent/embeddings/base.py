from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class Embedder(ABC):
    """Returns L2-normalised float32 vectors so inner product == cosine similarity."""

    name: str = "embedder"
    dim: int

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
