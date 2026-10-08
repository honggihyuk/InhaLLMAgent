"""Deterministic feature-hashing embedder: no model download, used offline and in tests."""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from alphaagent.embeddings.base import Embedder, l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 512, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> List[str]:
        toks = _TOKEN.findall(text.lower())
        feats = list(toks)
        for n in range(2, self.ngram + 1):
            feats += [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        return feats

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return l2_normalize(out)
