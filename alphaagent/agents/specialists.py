"""Specialised research agents: Ideator, Implementer, Evaluator (Alpha-GPT style loop)."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from alphaagent.agents.base import QuantAgent
from alphaagent.agents.parsing import extract_code
from alphaagent.agents.types import AgentRole, AlphaFactor, AlphaIdea, slugify

PRICE_FIELDS: Dict[str, str] = {
    "open": "daily open price",
    "high": "daily high",
    "low": "daily low",
    "close": "daily close",
    "volume": "shares traded",
    "vwap": "volume-weighted average price",
    "returns": "close-to-close return for the day ending at t",
    "cap": "market capitalisation",
}

ALPHA_CODE_CONTRACT = """Implement exactly one function:

    def compute_alpha(df: pd.DataFrame) -> pd.Series

- `df` has a MultiIndex (date, ticker), sorted by date then ticker. Row (t, i) only contains information
  known at the close of day t.
- Return a float Series with the same index: the alpha score for ticker i at date t (higher = more bullish).
- `pd` and `np` are already imported; do not import anything else, read files, or touch the network.
- Time-series operations MUST be grouped per ticker: df.groupby(level="ticker")[col].rolling(...)/.shift(k>0)
- Cross-sectional operations MUST be grouped per date: df.groupby(level="date")[col].rank(pct=True)
- No look-ahead: never shift(-k) / diff(-k), never bfill / fillna(method="bfill"), never normalise with
  full-sample statistics (mean/std over all dates); use rolling or expanding windows.
- Vectorised pandas only (no Python loops over dates or tickers). Leave NaN where there is not enough history."""


def describe_fields(fields: Dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in fields.items())


class IdeationAgent(QuantAgent):
    role = AgentRole.IDEATOR
    system_prompt = (
        "You are an elite quantitative researcher at a systematic hedge fund. Your expertise spans factor "
        "investing, statistical arbitrage, and alternative data. You generate novel, statistically grounded "
        "alpha ideas from financial documents, favouring economically intuitive relationships that can be "
        "expressed mathematically and computed from the available data fields only."
    )

    def generate_alpha_ideas(
        self,
        theme: str,
        num_ideas: int = 3,
        available_fields: Optional[Dict[str, str]] = None,
        feedback: Optional[List[Dict]] = None,
        tickers: Optional[Sequence[str]] = None,
        as_of: Optional[str] = None,
    ) -> List[AlphaIdea]:
        fields = available_fields or PRICE_FIELDS
        task = f"""Generate {num_ideas} novel alpha factor ideas related to: {theme}

Available data fields (use ONLY these):
{describe_fields(fields)}

For each idea give: name (snake_case), description (economic intuition, 2-3 sentences), expression
(formula using the fields above), data_fields (list), rationale, expected_behavior (when it should and
should not work), evidence (list of retrieved passage ids like "[2]" that motivated it; may be empty).

Prefer ideas that have a clear economic or behavioural rationale, are not trivial re-labelings of
momentum/value/size, and combine information in a way a linear sentiment score would miss."""
        context = None
        if feedback:
            context = (
                "Previously tested factors and their out-of-sample results (learn from them; do not resubmit "
                "near-duplicates of rejected ideas):\n" + json.dumps(feedback, indent=1, default=str)[:4000]
            )
        decision = self.think(
            task,
            kind="ideation",
            context=context,
            query=theme,
            tickers=tickers,
            as_of=as_of,
            expect_json=True,
            answer_hint=' containing a ```json block: {"ideas": [ {...}, ... ]}',
        )
        payload = decision.output
        raw = payload.get("ideas", []) if isinstance(payload, dict) else (payload or [])
        ideas = [AlphaIdea.from_dict(d, theme) for d in raw if isinstance(d, dict)]
        seen, unique = set(), []
        for idea in ideas:
            if idea.name not in seen:
                seen.add(idea.name)
                unique.append(idea)
        for idea in unique:
            self.memory.write(self.name, self.role.value, "idea", f"{idea.name}: {idea.description}", idea.to_dict())
        return unique[:num_ideas]


class ImplementationAgent(QuantAgent):
    role = AgentRole.IMPLEMENTER
    system_prompt = (
        "You are a quantitative implementation engineer. You translate alpha expressions into clean, "
        "vectorised pandas/numpy code that is free of look-ahead bias and robust to missing data."
    )

    def implement_alpha(self, idea: AlphaIdea, available_fields: Optional[Dict[str, str]] = None) -> AlphaFactor:
        fields = available_fields or PRICE_FIELDS
        task = f"""Implement this alpha factor.

Name: {idea.name}
Description: {idea.description}
Expression: {idea.expression}
Rationale: {idea.rationale}

Available columns in df:
{describe_fields(fields)}

{ALPHA_CODE_CONTRACT}"""
        decision = self._write_code(task)
        code = extract_code(decision.content)
        factor = AlphaFactor(
            name=idea.name,
            code=code,
            description=idea.description,
            expression=idea.expression,
            data_fields=[f for f in idea.data_fields if f in fields] or list(fields),
            author_agent=self.role,
            theme=idea.theme,
        )
        self.memory.write(self.name, self.role.value, "factor", f"implemented {factor.name}", {"name": factor.name})
        return factor

    def repair_alpha(self, factor: AlphaFactor, error: str, available_fields: Optional[Dict[str, str]] = None) -> AlphaFactor:
        """Ask for a corrected implementation given a sandbox / bias-check error."""
        fields = available_fields or PRICE_FIELDS
        task = f"""Your implementation of `{factor.name}` failed validation.

Error / findings:
{error}

Previous code:
```python
{factor.code}
```

Available columns:
{describe_fields(fields)}

Fix the problem and return the complete corrected function.

{ALPHA_CODE_CONTRACT}"""
        decision = self._write_code(task, kind="repair")
        factor.code = extract_code(decision.content)
        factor.notes.append(f"repaired after: {error[:200]}")
        return factor

    def _write_code(self, task: str, kind: str = "implementation"):
        # coding needs the spec, not documents: skip retrieval to keep the prompt focused
        return self.think(
            task,
            kind=kind,
            use_retrieval=False,
            answer_hint=" containing exactly one ```python block with the full function",
        )


class EvaluationAgent(QuantAgent):
    role = AgentRole.EVALUATOR
    system_prompt = (
        "You are a quantitative risk analyst specialising in alpha factor evaluation. You are rigorous about "
        "statistical significance, multiple testing, overfitting, transaction costs, and look-ahead bias."
    )

    def __init__(self, *args, ic_threshold: float = 0.02, min_tstat: float = 2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ic_threshold = ic_threshold
        self.min_tstat = min_tstat

    def apply_metrics(self, factor: AlphaFactor, metrics: Dict[str, float]) -> AlphaFactor:
        factor.metrics.update(metrics)
        factor.ic_score = metrics.get("ic")
        factor.rank_ic = metrics.get("rank_ic")
        factor.icir = metrics.get("icir")
        factor.sharpe = metrics.get("sharpe")
        factor.max_drawdown = metrics.get("max_drawdown")
        factor.turnover = metrics.get("turnover")
        factor.status = "evaluated"
        return factor

    def passes_thresholds(self, factor: AlphaFactor) -> bool:
        m = factor.metrics
        strong_ic = max(abs(m.get("ic", 0.0)), abs(m.get("rank_ic", 0.0))) >= self.ic_threshold
        significant = abs(m.get("ic_tstat", 0.0)) >= self.min_tstat
        return strong_ic and significant

    def review(self, factor: AlphaFactor, extra_context: str = "") -> Dict:
        """Quantitative gate + LLM critique. Returns {"verdict": accept|reject|revise, "reasons": [...], ...}."""
        gate = self.passes_thresholds(factor)
        task = f"""Review this alpha factor and decide whether it should advance to portfolio construction.

Factor: {factor.name}
Description: {factor.description}
Expression: {factor.expression}
Code:
```python
{factor.code}
```
Out-of-sample metrics: {json.dumps(factor.metrics, default=str)}
Validation notes: {factor.notes}
Quantitative gate (|IC| >= {self.ic_threshold} and |t| >= {self.min_tstat}): {"PASS" if gate else "FAIL"}
{extra_context}

Assess statistical significance, plausibility of the economic rationale, turnover/cost drag, possible
look-ahead bias or data snooping, and redundancy with well-known factors."""
        decision = self.think(
            task,
            kind="evaluation",
            query=factor.description or factor.name,
            expect_json=True,
            answer_hint=' containing a ```json block: {"verdict": "accept|reject|revise", "reasons": [...], "suggestions": [...]}',
        )
        out = decision.output if isinstance(decision.output, dict) else {}
        verdict = str(out.get("verdict", "reject")).lower()
        if verdict not in {"accept", "reject", "revise"}:
            verdict = "reject"
        if not gate and verdict == "accept":
            verdict = "reject"  # the LLM can veto a factor but never override the statistical gate
            out.setdefault("reasons", []).append("failed quantitative gate")
        out["verdict"] = verdict
        out["confidence"] = decision.confidence
        factor.notes.append(f"evaluator: {verdict} ({decision.confidence:.2f})")
        self.memory.write(
            self.name,
            self.role.value,
            "evaluation",
            f"{factor.name}: {verdict}; IC={factor.ic_score}, RankIC={factor.rank_ic}, Sharpe={factor.sharpe}",
            {"name": factor.name, "verdict": verdict, "metrics": factor.metrics},
        )
        return out


__all__ = [
    "ALPHA_CODE_CONTRACT",
    "PRICE_FIELDS",
    "EvaluationAgent",
    "IdeationAgent",
    "ImplementationAgent",
    "describe_fields",
    "slugify",
]
