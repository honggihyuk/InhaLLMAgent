"""Turn unstructured documents into point-in-time numeric panels the alpha code can use.

An LLM reads each document once and scores it (sentiment, guidance direction, risk tone). Scores become
available on the first trading day *after* publication (transcripts and filings often land after the
close), then decay with a half-life and expire after ``max_age`` trading days.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from alphaagent.agents.parsing import extract_json
from alphaagent.documents import Document
from alphaagent.llm.base import LLMClient

TEXT_FIELDS: Dict[str, str] = {
    "sentiment": "LLM-scored tone of the latest earnings call / news / filing, -1 (very negative) .. +1 (very positive), decaying since publication, 0 if none",
    "guidance": "LLM-scored direction of management guidance revision, -1 (cut) .. +1 (raised), decaying, 0 if none",
    "risk": "LLM-scored intensity of newly disclosed risks (litigation, liquidity, impairment), 0 .. 1, decaying",
}

SCORING_PROMPT = """TASK_KIND: text_scoring
Read the document below about {ticker} (published {date}) and score it from the perspective of an equity
investor who must decide what the market has NOT yet priced in.

Return ONLY a ```json block with:
  "sentiment": float in [-1, 1]  overall tone of results and outlook
  "guidance":  float in [-1, 1]  direction of forward guidance revision (0 if none given)
  "risk":      float in [0, 1]   severity of newly disclosed risks
  "rationale": one sentence

Document ({doc_type}): {title}
---
{text}
---"""


@dataclass
class DocumentScore:
    doc_id: str
    ticker: str
    published_at: str
    doc_type: str
    sentiment: float = 0.0
    guidance: float = 0.0
    risk: float = 0.0
    rationale: str = ""


def _clip(v, lo, hi) -> float:
    try:
        return float(min(hi, max(lo, float(v))))
    except (TypeError, ValueError):
        return 0.0


class TextSignalExtractor:
    def __init__(self, llm: LLMClient, cache_path: Optional[str] = None, max_chars: int = 8000) -> None:
        self.llm = llm
        self.max_chars = max_chars
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, DocumentScore] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    s = DocumentScore(**json.loads(line))
                    self.cache[s.doc_id] = s

    def score_document(self, doc: Document) -> DocumentScore:
        if doc.doc_id in self.cache:
            return self.cache[doc.doc_id]
        prompt = SCORING_PROMPT.format(
            ticker=doc.ticker, date=doc.published_at, doc_type=doc.doc_type, title=doc.title, text=doc.text[: self.max_chars]
        )
        resp = self.llm.complete(
            prompt,
            system="You are a meticulous equity analyst who converts financial text into calibrated numeric scores.",
            max_tokens=2000,
        )
        data = extract_json(resp.text) or {}
        score = DocumentScore(
            doc_id=doc.doc_id or "",
            ticker=doc.ticker,
            published_at=doc.published_at,
            doc_type=doc.doc_type,
            sentiment=_clip(data.get("sentiment", 0), -1, 1),
            guidance=_clip(data.get("guidance", 0), -1, 1),
            risk=_clip(data.get("risk", 0), 0, 1),
            rationale=str(data.get("rationale", ""))[:300],
        )
        self.cache[score.doc_id] = score
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(score)) + "\n")
        return score

    def score_documents(self, docs: Iterable[Document]) -> List[DocumentScore]:
        return [self.score_document(d) for d in docs]


def build_text_panel(
    scores: List[DocumentScore],
    index: pd.MultiIndex,
    fields=("sentiment", "guidance", "risk"),
    half_life: float = 10.0,
    max_age: int = 60,
) -> pd.DataFrame:
    """Daily (date, ticker) panel of decayed document scores, strictly point-in-time."""
    dates = index.get_level_values("date").unique().sort_values()
    tickers = index.get_level_values("ticker").unique()
    out = {f: pd.DataFrame(0.0, index=dates, columns=tickers) for f in fields}
    if not scores:
        return pd.DataFrame({f: out[f].stack() for f in fields}).reindex(index).fillna(0.0)

    ev = pd.DataFrame([asdict(s) for s in scores])
    ev = ev[ev["ticker"].isin(tickers)]
    pub = pd.to_datetime(ev["published_at"]).to_numpy()
    # first trading day strictly after publication
    pos = np.searchsorted(dates.to_numpy(), pub, side="right")
    ev = ev.assign(pos=pos)
    ev = ev[ev["pos"] < len(dates)]
    ev["avail"] = dates[ev["pos"].to_numpy()]

    day_num = pd.Series(np.arange(len(dates)), index=dates)
    for f in fields:
        events = ev.groupby(["avail", "ticker"])[f].mean().unstack("ticker").reindex(index=dates, columns=tickers)
        last_val = events.ffill()
        last_day = events.notna().mul(day_num, axis=0).where(events.notna()).ffill()
        age = (-last_day).add(day_num, axis=0)
        decayed = last_val * np.power(0.5, age / half_life)
        out[f] = decayed.where(age <= max_age).fillna(0.0)
    panel = pd.DataFrame({f: out[f].stack() for f in fields})
    panel.index.names = ["date", "ticker"]
    return panel.reindex(index).fillna(0.0)
