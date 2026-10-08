"""Runtime settings, read from environment variables with sensible local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("ALPHA_DATA_DIR", "data")))
    # "bge-m3" (default), "finbert", "hashing" (offline / tests), or any sentence-transformers model id
    embedder: str = field(default_factory=lambda: _env("ALPHA_EMBEDDER", "bge-m3"))
    embed_device: str = field(default_factory=lambda: _env("ALPHA_EMBED_DEVICE", "cpu"))
    embed_max_seq_length: int = field(default_factory=lambda: int(_env("ALPHA_EMBED_MAX_SEQ", "1024")))
    chunk_size: int = field(default_factory=lambda: int(_env("ALPHA_CHUNK_SIZE", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("ALPHA_CHUNK_OVERLAP", "200")))
    # SEC EDGAR requires a descriptive User-Agent with a contact e-mail, e.g. "Inha Research you@example.com"
    sec_user_agent: str = field(default_factory=lambda: _env("SEC_USER_AGENT", ""))
    # LLM backend: "anthropic" or "mock"
    llm_provider: str = field(default_factory=lambda: _env("ALPHA_LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: _env("ALPHA_LLM_MODEL", "claude-opus-5-5"))
    # "kafka://host:9092" in production; "memory://" runs everything in one process
    broker_url: str = field(default_factory=lambda: _env("ALPHA_BROKER_URL", "memory://"))

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


def get_settings() -> Settings:
    return Settings()
