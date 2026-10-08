"""Ensemble decision protocol (TradingAgents style):

    y_hat(j, t) = argmax_{y in {buy, hold, sell}}  sum_i  w_i * 1[vote_i(j, t) = y]

Voters are factor signals (quantile votes) and/or LLM analyst agents. Weights w_i track each voter's
historical accuracy and are updated only with returns that are already realised (no look-ahead).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from alphaagent.agents.base import QuantAgent
from alphaagent.agents.types import AgentRole

BUY, HOLD, SELL = 1, 0, -1
LABELS = {BUY: "buy", HOLD: "hold", SELL: "sell"}
_FROM_TEXT = {"buy": BUY, "long": BUY, "hold": HOLD, "neutral": HOLD, "sell": SELL, "short": SELL}


def signal_votes(alpha: pd.Series, quantile: float = 0.2) -> pd.Series:
    """Top quantile -> buy, bottom quantile -> sell, else hold (per date, cross-sectionally)."""
    pct = alpha.groupby(level="date").rank(pct=True)
    votes = pd.Series(HOLD, index=alpha.index, dtype=int)
    votes[pct > 1 - quantile] = BUY
    votes[pct <= quantile] = SELL
    votes[alpha.isna()] = HOLD
    return votes


@dataclass
class EnsembleDecision:
    voters: Sequence[str]
    learning_rate: float = 0.05
    min_weight: float = 0.02
    weights: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = {v: 1.0 / len(self.voters) for v in self.voters}

    def decide(self, votes: Dict[str, pd.Series]) -> pd.Series:
        frame = pd.DataFrame({k: v for k, v in votes.items() if k in self.weights}).fillna(HOLD).astype(int)
        score = {y: sum(self.weights[c] * (frame[c] == y) for c in frame.columns) for y in (BUY, HOLD, SELL)}
        scores = pd.DataFrame(score)[[HOLD, BUY, SELL]]
        best = scores.idxmax(axis=1).astype(int)
        # a tie at the top between different actions resolves to HOLD (the conservative choice)
        tied = scores.eq(scores.max(axis=1), axis=0).sum(axis=1) > 1
        return best.where(~tied, HOLD).rename("decision")

    def update(self, votes: Dict[str, pd.Series], realised: pd.Series) -> None:
        """Accuracy of each voter's non-hold calls against realised returns, blended into its weight."""
        # excess return vs the cross-section, so a rising market does not reward every buy vote
        excess = realised - realised.groupby(level="date").transform("mean")
        for name, v in votes.items():
            if name not in self.weights:
                continue
            df = pd.concat({"v": v, "r": excess}, axis=1).dropna()
            active = df[df["v"] != HOLD]
            if active.empty:
                continue
            hit = float((np.sign(active["r"]) == active["v"]).mean())
            self.weights[name] = (1 - self.learning_rate) * self.weights[name] + self.learning_rate * hit
        total = sum(max(w, self.min_weight) for w in self.weights.values())
        self.weights = {k: max(w, self.min_weight) / total for k, w in self.weights.items()}

    def run(self, votes: Dict[str, pd.Series], forward_returns: pd.Series, horizon: int = 1) -> pd.Series:
        """Online over dates: decide at t with current weights; update with votes from t-h whose returns are realised."""
        dates = sorted(set().union(*[set(v.index.get_level_values("date")) for v in votes.values()]))
        decisions = []
        for i, d in enumerate(dates):
            today = {k: v.xs(d, level="date", drop_level=False) for k, v in votes.items()}
            decisions.append(self.decide(today))
            if i >= horizon:
                past = dates[i - horizon]
                past_votes = {k: v.xs(past, level="date", drop_level=False) for k, v in votes.items()}
                # forward return from past to past+h is fully realised by today
                self.update(past_votes, forward_returns.xs(past, level="date", drop_level=False))
            self.history.append({"date": d, **self.weights})
        return pd.concat(decisions)


class AnalystAgent(QuantAgent):
    """Specialist analyst (sentiment / fundamental / technical / risk) that can cast an LLM vote on a ticker."""

    role = AgentRole.ANALYST

    def __init__(self, *args, specialty: str = "sentiment", **kw):
        kw.setdefault("name", f"{specialty}_analyst")
        super().__init__(*args, **kw)
        self.specialty = specialty
        self.system_prompt = (
            f"You are a {specialty} analyst on a systematic equity team. You form views only from the evidence "
            "provided and say 'hold' when the evidence is thin."
        )

    def vote(self, ticker: str, as_of: Optional[str] = None, context: Optional[str] = None) -> Dict:
        task = f"""From a {self.specialty} perspective, should the team buy, hold, or sell {ticker} for the next 1-4 weeks,
using only documents published on or before {as_of or 'today'}?"""
        decision = self.think(
            task, kind="vote", query=f"{ticker} {self.specialty} outlook", tickers=[ticker], as_of=as_of, context=context,
            expect_json=True, answer_hint=' containing a ```json block: {"vote": "buy|hold|sell", "reason": "..."}',
        )
        out = decision.output if isinstance(decision.output, dict) else {}
        vote = _FROM_TEXT.get(str(out.get("vote", "hold")).lower(), HOLD)
        return {"ticker": ticker, "voter": self.name, "vote": vote, "confidence": decision.confidence,
                "reason": str(out.get("reason", ""))[:300]}


def llm_votes(analysts: Sequence[AnalystAgent], tickers: Sequence[str], as_of: str) -> Dict[str, pd.Series]:
    """Collect one vote per analyst per ticker at ``as_of`` as Series keyed by (date, ticker)."""
    date = pd.Timestamp(as_of)
    out = {}
    for a in analysts:
        idx = pd.MultiIndex.from_product([[date], list(tickers)], names=["date", "ticker"])
        out[a.name] = pd.Series([a.vote(t, as_of)["vote"] for t in tickers], index=idx, dtype=int)
    return out


def describe_votes(votes: Dict[str, pd.Series]) -> str:
    return json.dumps({k: {LABELS[int(x)]: int((v == x).sum()) for x in (BUY, HOLD, SELL)} for k, v in votes.items()})
