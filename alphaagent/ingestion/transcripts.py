"""Earnings-call transcripts from local files.

Supported layouts inside ``root``:

* ``*.json``  -> one object or a list of objects with ``ticker``, ``date``, ``title``, ``text``
* ``*.jsonl`` -> one such object per line
* ``*.txt``   -> filename ``TICKER_YYYY-MM-DD[_anything].txt``; the whole file is the transcript
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator

from alphaagent.documents import Document
from alphaagent.ingestion.base import DocumentSource

_TXT_NAME = re.compile(r"^(?P<ticker>[A-Za-z.\-]+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<rest>.*))?$")


def _record_to_doc(rec: Dict[str, Any], source: str) -> Document:
    meta = {k: v for k, v in rec.items() if k not in {"ticker", "date", "published_at", "title", "text"}}
    return Document(
        ticker=rec["ticker"],
        doc_type="transcript",
        title=rec.get("title") or f"{rec['ticker']} earnings call",
        text=rec["text"],
        published_at=rec.get("published_at") or rec["date"],
        source=source,
        metadata=meta,
    )


class TranscriptSource(DocumentSource):
    name = "transcripts"

    def __init__(self, root) -> None:
        self.root = Path(root)

    def load(self) -> Iterator[Document]:
        for path in sorted(self.root.rglob("*")):
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                for i, rec in enumerate(data if isinstance(data, list) else [data]):
                    yield _record_to_doc(rec, f"{path.as_posix()}#{i}")
            elif path.suffix == ".jsonl":
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    if line.strip():
                        yield _record_to_doc(json.loads(line), f"{path.as_posix()}#{i}")
            elif path.suffix == ".txt":
                m = _TXT_NAME.match(path.stem)
                if not m:
                    continue
                title = (m.group("rest") or "earnings call").replace("_", " ")
                yield Document(
                    ticker=m.group("ticker"),
                    doc_type="transcript",
                    title=f"{m.group('ticker').upper()} {title}",
                    text=path.read_text(encoding="utf-8"),
                    published_at=m.group("date"),
                    source=path.as_posix(),
                )
