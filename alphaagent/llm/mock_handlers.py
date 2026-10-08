"""Deterministic stand-ins for each agent task kind, so the full pipeline runs offline.

The ideas and code here are real, look-ahead-free factor implementations; they just aren't LLM-written.
"""

from __future__ import annotations

import json
import random
import re
from typing import Callable, Dict, List, Optional

TEMPLATES: List[Dict] = [
    {
        "name": "short_term_reversal",
        "description": "Stocks that fell over the past week tend to bounce as liquidity-driven selling pressure fades.",
        "expression": "-sum(returns, 5)",
        "data_fields": ["returns"],
        "code": '''def compute_alpha(df):
    r5 = df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())
    return -r5''',
    },
    {
        "name": "volume_confirmed_reversal",
        "description": "Reversal is stronger when the move happened on abnormally high volume (overreaction).",
        "expression": "-sum(returns,5) * volume / mean(volume,20)",
        "data_fields": ["returns", "volume"],
        "code": '''def compute_alpha(df):
    g = df.groupby(level="ticker")
    r5 = g["returns"].transform(lambda s: s.rolling(5).sum())
    vol_ratio = df["volume"] / g["volume"].transform(lambda s: s.rolling(20).mean())
    return -r5 * vol_ratio''',
    },
    {
        "name": "vwap_gap",
        "description": "Closing far above VWAP signals late-day buying exhaustion; closing below signals capitulation.",
        "expression": "(vwap - close) / close",
        "data_fields": ["vwap", "close"],
        "code": '''def compute_alpha(df):
    return (df["vwap"] - df["close"]) / df["close"]''',
    },
    {
        "name": "low_volatility",
        "description": "Low-volatility names earn higher risk-adjusted returns (lottery-demand mispricing).",
        "expression": "-std(returns, 20)",
        "data_fields": ["returns"],
        "code": '''def compute_alpha(df):
    vol = df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(20).std())
    return -vol''',
    },
    {
        "name": "sentiment_drift",
        "requires": ["sentiment"],
        "description": "Prices under-react to tone in earnings calls and news; fresh positive tone predicts drift.",
        "expression": "rank(sentiment)",
        "data_fields": ["sentiment"],
        "code": '''def compute_alpha(df):
    return df.groupby(level="date")["sentiment"].rank(pct=True)''',
    },
    {
        "name": "guidance_surprise_unpriced",
        "requires": ["guidance"],
        "description": "Guidance raises that the price has not yet reflected (weak recent return) drift upward.",
        "expression": "rank(guidance) - rank(sum(returns,5))",
        "data_fields": ["guidance", "returns"],
        "code": '''def compute_alpha(df):
    g = df.groupby(level="ticker")
    guid = g["guidance"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    r5 = g["returns"].transform(lambda s: s.rolling(5).sum())
    tmp = pd.DataFrame({"g": guid, "r": r5})
    by_date = tmp.groupby(level="date")
    return by_date["g"].rank(pct=True) - by_date["r"].rank(pct=True)''',
    },
    {
        "name": "risk_tone_adjusted_sentiment",
        "requires": ["sentiment", "risk"],
        "description": "Positive tone matters more when management is not simultaneously flagging new risks.",
        "expression": "ts_mean(sentiment,10) * (1 - ts_mean(risk,10))",
        "data_fields": ["sentiment", "risk"],
        "code": '''def compute_alpha(df):
    g = df.groupby(level="ticker")
    s = g["sentiment"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    r = g["risk"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    return s * (1 - r)''',
    },
]

_BY_NAME = {t["name"]: t for t in TEMPLATES}


def _fields_in_prompt(prompt: str) -> List[str]:
    return re.findall(r"^- ([a-z_][a-z0-9_]*):", prompt, flags=re.MULTILINE)


def _wrap(answer: str, confidence: float, reasoning: str = "Reviewed the context and constraints.") -> str:
    return f"## Reasoning\n{reasoning}\n\n## Answer\n{answer}\n\n[Confidence: {confidence:.2f}]"


def ideation_handler(seed: int) -> Callable[[str, Optional[str]], str]:
    rng = random.Random(seed)

    def handle(prompt: str, system: Optional[str]) -> str:
        fields = set(_fields_in_prompt(prompt))
        m = re.search(r"Generate (\d+) novel alpha", prompt)
        n = int(m.group(1)) if m else 3
        tested = set(re.findall(r'"name":\s*"([a-z0-9_]+)"', prompt.split("Previously tested factors")[-1])) if (
            "Previously tested factors" in prompt
        ) else set()
        pool = [
            t for t in TEMPLATES if set(t.get("requires", [])) <= fields and set(t["data_fields"]) <= fields | {"returns"}
        ]
        # text-derived ideas first: that is the point of the system
        pool.sort(key=lambda t: (not t.get("requires"), rng.random()))
        fresh = [t for t in pool if t["name"] not in tested] or pool
        ideas = [
            {k: t[k] for k in ("name", "description", "expression", "data_fields")}
            | {"rationale": t["description"], "expected_behavior": "Works in normal liquidity regimes.", "evidence": ["[1]"]}
            for t in fresh[:n]
        ]
        return _wrap("```json\n" + json.dumps({"ideas": ideas}, indent=1) + "\n```", 0.6)

    return handle


def implementation_handler(prompt: str, system: Optional[str]) -> str:
    m = re.search(r"Name:\s*([a-z0-9_]+)", prompt) or re.search(r"implementation of `([a-z0-9_]+)`", prompt)
    template = _BY_NAME.get(m.group(1) if m else "", TEMPLATES[0])
    return _wrap(f"```python\n{template['code']}\n```", 0.8)


def evaluation_handler(prompt: str, system: Optional[str]) -> str:
    passed = "): PASS" in prompt
    verdict = {
        "verdict": "accept" if passed else "reject",
        "reasons": ["IC is statistically significant"] if passed else ["IC below threshold or insignificant"],
        "suggestions": [] if passed else ["combine with a conditioning variable"],
    }
    return _wrap("```json\n" + json.dumps(verdict) + "\n```", 0.7 if passed else 0.6)


_POS = ("record", "raising", "raised", "beat", "strong", "growth", "expanded", "exceeded", "robust", "accelerat",
        "upgrade", "all-time high", "momentum", "outperform")
_NEG = ("decline", "lowering", "lowered", "cut", "weak", "pressure", "compressed", "delay", "miss", "challenging",
        "impairment", "softness", "headwind", "slowdown")
_RISK = ("litigation", "investigation", "liquidity", "revolver", "impairment", "uncertain", "restatement", "default",
         "covenant", "recall")


def text_scoring_handler(prompt: str, system: Optional[str]) -> str:
    """Lexicon scorer: a transparent stand-in for LLM document scoring."""
    body = prompt.split("---", 1)[-1].lower()
    pos = sum(body.count(w) for w in _POS)
    neg = sum(body.count(w) for w in _NEG)
    sentiment = (pos - neg) / max(pos + neg, 1)
    guidance = 0.0
    if re.search(r"(rais\w*|increas\w*|lift\w*)[^.]{0,30}guidance|guidance[^.]{0,30}(rais|increas|above)", body):
        guidance = 1.0
    elif re.search(r"(lower\w*|cut\w*|reduc\w*)[^.]{0,30}guidance|guidance[^.]{0,30}(lower|cut|below)", body):
        guidance = -1.0
    risk = min(1.0, sum(body.count(w) for w in _RISK) / 3.0)
    out = {"sentiment": round(sentiment, 3), "guidance": guidance, "risk": round(risk, 3), "rationale": "lexicon score"}
    return "```json\n" + json.dumps(out) + "\n```"


def generic_handler(prompt: str, system: Optional[str]) -> str:
    return _wrap("Acknowledged. No further action required.", 0.5)


def default_handlers(seed: int = 0) -> Dict[str, Callable[[str, Optional[str]], str]]:
    return {
        "ideation": ideation_handler(seed),
        "implementation": implementation_handler,
        "repair": implementation_handler,
        "evaluation": evaluation_handler,
        "text_scoring": text_scoring_handler,
        "generic": generic_handler,
    }
