from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from alphaagent.documents import Document


class DocumentSource(ABC):
    """Anything that yields :class:`Document` objects (files, EDGAR, RSS, Kafka...)."""

    name: str = "source"

    @abstractmethod
    def load(self) -> Iterator[Document]:
        ...
