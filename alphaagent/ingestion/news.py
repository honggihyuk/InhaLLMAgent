"""News from RSS/Atom feeds or local JSONL files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional

import feedparser
import httpx

from alphaagent.documents import Document
from alphaagent.ingestion.base import DocumentSource
from alphaagent.ingestion.html_text import html_to_text

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


class RSSNewsSource(DocumentSource):
    """``feeds`` maps ticker -> feed URL. Use :meth:`yahoo` for per-ticker Yahoo Finance headlines."""

    name = "rss"

    def __init__(self, feeds: Dict[str, str], client: Optional[httpx.Client] = None) -> None:
        self.feeds = {t.upper(): url for t, url in feeds.items()}
        self.client = client or httpx.Client(
            headers={"User-Agent": "alphaagent/0.1 (research)"}, timeout=20.0, follow_redirects=True
        )

    @classmethod
    def yahoo(cls, tickers, client: Optional[httpx.Client] = None) -> "RSSNewsSource":
        return cls({t: YAHOO_RSS.format(ticker=t.upper()) for t in tickers}, client=client)

    def load(self) -> Iterator[Document]:
        for ticker, url in self.feeds.items():
            resp = self.client.get(url)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                parsed = entry.get("published_parsed") or entry.get("updated_parsed")
                if not parsed:
                    continue
                published = datetime(*parsed[:6], tzinfo=timezone.utc)
                summary = html_to_text(entry.get("summary", "") or "")
                title = entry.get("title", "").strip()
                yield Document(
                    ticker=ticker,
                    doc_type="news",
                    title=title,
                    text=f"{title}\n\n{summary}".strip(),
                    published_at=published,
                    source=entry.get("link", url),
                    metadata={"feed": url, "published_ts": published.isoformat()},
                )


class JSONLNewsSource(DocumentSource):
    """Local news archive: one JSON object per line with ``ticker``, ``date``, ``title``, ``text``/``summary``."""

    name = "news_jsonl"

    def __init__(self, path) -> None:
        self.path = Path(path)

    def load(self) -> Iterator[Document]:
        files = sorted(self.path.rglob("*.jsonl")) if self.path.is_dir() else [self.path]
        for f in files:
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                rec = json.loads(line)
                body = rec.get("text") or rec.get("summary") or ""
                yield Document(
                    ticker=rec["ticker"],
                    doc_type="news",
                    title=rec.get("title", ""),
                    text=f"{rec.get('title', '')}\n\n{body}".strip(),
                    published_at=rec.get("published_at") or rec["date"],
                    source=rec.get("url") or f"{f.as_posix()}#{i}",
                    metadata={k: v for k, v in rec.items() if k in {"publisher", "sentiment", "url"}},
                )
