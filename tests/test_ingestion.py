import json

import httpx
import pytest

from alphaagent.documents import Document
from alphaagent.ingestion import JSONLNewsSource, RSSNewsSource, SECFilingSource, TextChunker, TranscriptSource
from alphaagent.ingestion.html_text import html_to_text

from .conftest import SAMPLES


def test_chunker_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    pieces = TextChunker(500, 100).split_text(text)
    assert len(pieces) > 1
    assert all(len(p) <= 500 for p in pieces)
    # consecutive chunks share overlapping text
    assert pieces[0][-40:].split()[-1] in pieces[1]


def test_chunk_carries_metadata_header():
    doc = Document(ticker="acme", doc_type="news", title="t", text="hello world", published_at="2026-01-02")
    (chunk,) = TextChunker().chunk(doc)
    assert chunk.ticker == "ACME" and chunk.published_at == "2026-01-02"
    assert chunk.text.startswith("[ACME | news | 2026-01-02]")


def test_document_requires_valid_type_and_date():
    with pytest.raises(ValueError):
        Document(ticker="A", doc_type="tweet", title="", text="x", published_at="2026-01-01")
    with pytest.raises(ValueError):
        Document(ticker="A", doc_type="news", title="", text="x", published_at="")


def test_transcript_source_reads_json_and_txt():
    docs = list(TranscriptSource(SAMPLES / "transcripts").load())
    assert {d.ticker for d in docs} == {"ACME", "GLBX", "INIT"}
    init = next(d for d in docs if d.ticker == "INIT")
    assert init.published_at == "2026-02-11"
    assert len({d.doc_id for d in docs}) == len(docs)


def test_jsonl_news_source():
    docs = list(JSONLNewsSource(SAMPLES / "news").load())
    assert len(docs) == 4 and all(d.doc_type == "news" for d in docs)


def test_html_to_text_strips_markup():
    html = "<html><head><style>x{}</style></head><body><p>Risk&nbsp;Factors</p><div>Item 1A</div><script>bad()</script></body></html>"
    text = html_to_text(html)
    assert "Risk Factors" in text and "Item 1A" in text and "bad()" not in text


def _sec_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "@" in request.headers["user-agent"]
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "ACME", "title": "Acme"}})
        if "submissions" in url:
            assert "CIK0000320193" in url
            return httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {
                            "form": ["8-K", "4", "10-Q"],
                            "filingDate": ["2026-02-01", "2026-01-15", "2025-11-01"],
                            "reportDate": ["2026-01-31", "", "2025-09-30"],
                            "accessionNumber": ["0000320193-26-000010", "x", "0000320193-25-000099"],
                            "primaryDocument": ["a8k.htm", "x.xml", "a10q.htm"],
                        }
                    }
                },
            )
        if "/Archives/" in url:
            return httpx.Response(200, text="<html><body><p>Results of operations improved.</p></body></html>")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_sec_source_with_mock_edgar():
    client = httpx.Client(transport=_sec_transport(), headers={"User-Agent": "test you@example.com"})
    src = SECFilingSource(["ACME"], user_agent="test you@example.com", client=client, min_interval=0)
    docs = list(src.load())
    assert [d.metadata["form"] for d in docs] == ["8-K", "10-Q"]
    assert docs[0].published_at == "2026-02-01"
    assert "000032019326000010/a8k.htm" in docs[0].source
    assert "Results of operations improved." in docs[0].text


def test_sec_source_requires_contact_user_agent():
    with pytest.raises(ValueError):
        SECFilingSource(["ACME"], user_agent="")


def test_rss_news_source():
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    <item><title>ACME beats estimates</title><link>http://x/1</link>
    <description>&lt;p&gt;Strong quarter&lt;/p&gt;</description><pubDate>Tue, 03 Feb 2026 14:00:00 GMT</pubDate></item>
    </channel></rss>"""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=rss)))
    (doc,) = list(RSSNewsSource({"acme": "http://feed"}, client=client).load())
    assert doc.ticker == "ACME" and doc.published_at == "2026-02-03"
    assert "Strong quarter" in doc.text and doc.source == "http://x/1"
