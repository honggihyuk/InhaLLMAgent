from __future__ import annotations

from alphaagent.embeddings.base import Embedder, l2_normalize
from alphaagent.embeddings.hashing import HashingEmbedder


def build_embedder(name: str, device: str = "cpu", max_seq_length: int = 1024) -> Embedder:
    key = name.lower()
    if key in {"hashing", "hash"}:
        return HashingEmbedder()
    from alphaagent.embeddings import sentence

    if key in {"bge-m3", "bge_m3", "bgem3"}:
        return sentence.bge_m3(device=device, max_seq_length=max_seq_length)
    if key == "finbert":
        return sentence.finbert(device=device)
    return sentence.SentenceTransformerEmbedder(name, device=device, max_seq_length=max_seq_length)


__all__ = ["Embedder", "HashingEmbedder", "build_embedder", "l2_normalize"]
