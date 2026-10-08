"""Transformer embedders via sentence-transformers: BGE-M3 (default) and FinBERT."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from alphaagent.embeddings.base import Embedder, l2_normalize

BGE_M3 = "BAAI/bge-m3"
FINBERT = "ProsusAI/finbert"


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = BGE_M3,
        device: str = "cpu",
        max_seq_length: Optional[int] = 1024,
        batch_size: int = 16,
        query_instruction: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'alphaagent[embeddings]' to use transformer embedders") from e
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return l2_normalize(np.asarray(vecs))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self.query_instruction + text])[0]


def bge_m3(device: str = "cpu", max_seq_length: int = 1024) -> SentenceTransformerEmbedder:
    # BGE-M3 dense retrieval needs no query instruction
    return SentenceTransformerEmbedder(BGE_M3, device=device, max_seq_length=max_seq_length)


def finbert(device: str = "cpu") -> SentenceTransformerEmbedder:
    # FinBERT is a BERT classifier; sentence-transformers wraps it with mean pooling
    return SentenceTransformerEmbedder(FINBERT, device=device, max_seq_length=512)
