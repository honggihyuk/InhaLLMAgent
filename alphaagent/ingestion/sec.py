"""SEC EDGAR filings (10-K, 10-Q, 8-K, ...) via the free public EDGAR JSON endpoints.

EDGAR needs no API key, but its fair-access policy requires a descriptive ``User-Agent``
containing a contact e-mail and at most 10 requests/second.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, Iterator, List, Optional

import httpx

from alphaagent.documents import Document
from alphaagent.ingestion.base import DocumentSource
from alphaagent.ingestion.html_text import html_to_text

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class SECFilingSource(DocumentSource):
    name = "sec"

    def __init__(
        self,
        tickers: Iterable[str],
        user_agent: str,
        forms: Iterable[str] = ("10-K", "10-Q", "8-K"),
        limit_per_ticker: int = 5,
        start_date: Optional[str] = None,
        max_chars: int = 400_000,
        client: Optional[httpx.Client] = None,
        min_interval: float = 0.12,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC EDGAR requires a User-Agent with a contact e-mail, e.g. "
                "SEC_USER_AGENT='Inha Research you@example.com'"
            )
        self.tickers = [t.upper() for t in tickers]
        self.forms = set(forms)
        self.limit = limit_per_ticker
        self.start_date = start_date
        self.max_chars = max_chars
        self.min_interval = min_interval
        self._last_request = 0.0
        self.client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )
        self._cik_map: Optional[Dict[str, int]] = None

    def _get(self, url: str) -> httpx.Response:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp

    def cik_for(self, ticker: str) -> int:
        if self._cik_map is None:
            data = self._get(TICKER_MAP_URL).json()
            self._cik_map = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
        try:
            return self._cik_map[ticker.upper()]
        except KeyError:
            raise KeyError(f"Ticker {ticker} not found in EDGAR company_tickers.json") from None

    def list_filings(self, ticker: str) -> List[Dict[str, str]]:
        cik = self.cik_for(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        out = []
        for i, form in enumerate(recent["form"]):
            if form not in self.forms:
                continue
            filed = recent["filingDate"][i]
            if self.start_date and filed < self.start_date:
                continue
            out.append(
                {
                    "cik": str(cik),
                    "form": form,
                    "filing_date": filed,
                    "report_date": recent.get("reportDate", [""] * (i + 1))[i],
                    "accession": recent["accessionNumber"][i],
                    "document": recent["primaryDocument"][i],
                }
            )
            if len(out) >= self.limit:
                break
        return out

    def load(self) -> Iterator[Document]:
        for ticker in self.tickers:
            for f in self.list_filings(ticker):
                url = ARCHIVE_URL.format(
                    cik=f["cik"], accession=f["accession"].replace("-", ""), document=f["document"]
                )
                text = html_to_text(self._get(url).text)[: self.max_chars]
                if not text:
                    continue
                yield Document(
                    ticker=ticker,
                    doc_type="sec_filing",
                    title=f"{ticker} {f['form']} filed {f['filing_date']}",
                    text=text,
                    # filing date (not period end) is when the market could first read it
                    published_at=f["filing_date"],
                    source=url,
                    metadata={"form": f["form"], "accession": f["accession"], "report_date": f["report_date"]},
                )
