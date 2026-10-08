"""AlphaGenerationPipeline: the end-to-end LLM alpha mining loop.

Search -> Analyze -> Finalize (SAF, Kou et al. 2025):

1. Ideation      IdeationAgent proposes factors grounded in retrieved documents + library feedback
2. Implementation ImplementationAgent writes ``compute_alpha``; repaired on failure
3. Safety        AST policy + isolated sandbox execution + look-ahead detection (static + truncation)
4. Evaluation    in-sample / out-of-sample IC, net-of-cost backtest (sign fixed in-sample)
5. Novelty       orthogonalisation vs known style factors and redundancy vs accepted factors
6. Review        EvaluationAgent verdict (can veto, cannot override statistics) -> FactorLibrary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from alphaagent.agents.memory import SharedMemory
from alphaagent.agents.protocol import MessageBus
from alphaagent.agents.specialists import PRICE_FIELDS, EvaluationAgent, IdeationAgent, ImplementationAgent
from alphaagent.agents.types import AlphaFactor, AlphaIdea
from alphaagent.backtest.engine import BacktestConfig, run_backtest
from alphaagent.backtest.metrics import summarize_factor
from alphaagent.features.text_signals import TEXT_FIELDS
from alphaagent.library import FactorLibrary
from alphaagent.llm.base import LLMClient
from alphaagent.sandbox import AlphaSandbox
from alphaagent.validation.factors import compute_known_factors, factor_correlation, orthogonalize
from alphaagent.validation.lookahead import check_lookahead

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    num_ideas: int = 3
    horizon: int = 5  # IC forward-return horizon (trading days)
    oos_fraction: float = 0.3
    ic_threshold: float = 0.02
    min_tstat: float = 2.0
    max_repairs: int = 2
    max_known_corr: float = 0.7
    max_library_corr: float = 0.7
    min_residual_ic: float = 0.01
    cost_bps: float = 5.0
    min_net_sharpe: float = 0.0  # out-of-sample, after transaction costs
    lookahead_cutoffs: int = 2
    review_with_llm: bool = True


class AlphaGenerationPipeline:
    def __init__(
        self,
        llm: LLMClient,
        retriever=None,
        memory: Optional[SharedMemory] = None,
        bus: Optional[MessageBus] = None,
        sandbox: Optional[AlphaSandbox] = None,
        library: Optional[FactorLibrary] = None,
        config: Optional[PipelineConfig] = None,
        weight_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.memory = memory if memory is not None else SharedMemory()
        self.bus = bus
        self.sandbox = sandbox if sandbox is not None else AlphaSandbox()
        self.library = library if library is not None else FactorLibrary()
        self.config = config or PipelineConfig()
        self.weight_fn = weight_fn  # risk overlay hook (week 7-8)

        common = dict(memory=self.memory, bus=bus)
        self.ideator = IdeationAgent(llm, retriever=retriever, **common)
        self.implementer = ImplementationAgent(llm, **common)
        self.evaluator = EvaluationAgent(
            llm, retriever=retriever, ic_threshold=self.config.ic_threshold, min_tstat=self.config.min_tstat, **common
        )
        self.pipeline_history: List[Dict] = []

    # ---- data ----------------------------------------------------------
    @staticmethod
    def prepare_panel(
        price_data: pd.DataFrame, text_panel: Optional[pd.DataFrame] = None, as_of: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        panel = price_data.sort_index()
        fields = {k: v for k, v in PRICE_FIELDS.items() if k in panel.columns}
        if text_panel is not None:
            panel = panel.join(text_panel.reindex(panel.index), how="left")
            for c in text_panel.columns:
                fields[c] = TEXT_FIELDS.get(c, f"text-derived feature {c}")
        if as_of is not None:
            panel = panel[panel.index.get_level_values("date") <= pd.Timestamp(as_of)]
        return panel, fields

    def _split(self, panel: pd.DataFrame) -> pd.Timestamp:
        dates = panel.index.get_level_values("date").unique().sort_values()
        return dates[int(len(dates) * (1 - self.config.oos_fraction)) - 1]

    # ---- stages ----------------------------------------------------------
    def _execute(self, code: str, data: pd.DataFrame) -> pd.Series:
        res = self.sandbox.run(code, data)
        if not res.ok:
            raise RuntimeError(res.error)
        return res.values

    def implement_and_validate(
        self, idea: AlphaIdea, panel: pd.DataFrame, fields: Dict[str, str]
    ) -> Tuple[AlphaFactor, Optional[pd.Series]]:
        factor = self.implementer.implement_alpha(idea, fields)
        factor.name = self.library.unique_name(factor.name)
        for attempt in range(self.config.max_repairs + 1):
            res = self.sandbox.run(factor.code, panel)
            problem = ""
            if not res.ok:
                problem = f"Sandbox execution failed:\n{res.error}"
            else:
                report = check_lookahead(
                    factor.code,
                    compute=lambda d: self._execute(factor.code, d),
                    data=panel,
                    n_cutoffs=self.config.lookahead_cutoffs,
                    full=res.values,
                )
                if report.static_warnings:
                    factor.notes.extend(report.static_warnings)
                if report.ok:
                    factor.notes.extend(res.warnings)
                    return factor, res.values
                problem = "Look-ahead bias detected:\n" + report.summary()
            factor.errors.append(problem[:1000])
            log.info("factor %s attempt %d failed: %s", factor.name, attempt, problem[:200])
            if attempt < self.config.max_repairs:
                factor = self.implementer.repair_alpha(factor, problem, fields)
        factor.status = "invalid"
        return factor, None

    def evaluate(self, factor: AlphaFactor, values: pd.Series, panel: pd.DataFrame) -> AlphaFactor:
        cfg = self.config
        split = self._split(panel)
        dates = panel.index.get_level_values("date")
        is_mask, oos_mask = dates <= split, dates > split
        close = panel["close"]

        m_is = summarize_factor(values[is_mask], close[is_mask], horizon=cfg.horizon)
        sign = -1.0 if m_is["ic"] < 0 else 1.0  # direction chosen in-sample only
        signed = values * sign
        m_oos = summarize_factor(signed[oos_mask], close[oos_mask], horizon=cfg.horizon)
        bt = run_backtest(signed[oos_mask], panel[oos_mask], BacktestConfig(cost_bps=cfg.cost_bps, weight_fn=self.weight_fn))

        self.evaluator.apply_metrics(factor, m_oos)
        factor.metrics.update({f"is_{k}": v for k, v in m_is.items()})
        factor.metrics.update({f"oos_{k}": v for k, v in m_oos.items()})
        factor.metrics.update({f"bt_{k}": v for k, v in bt.stats.items()})
        factor.metrics.update({"sign": sign, "split_date": str(split.date())})
        factor.sharpe = bt.stats.get("sharpe")
        factor.max_drawdown = bt.stats.get("max_drawdown")

        known = compute_known_factors(panel)
        orth = orthogonalize(signed, known)
        m_res = summarize_factor(orth.residual[oos_mask], close[oos_mask], horizon=cfg.horizon)
        factor.metrics.update(
            {
                "known_factor_corr": orth.correlations,
                "known_factor_r2": orth.r2,
                "max_known_corr": orth.max_abs_corr,
                "closest_known_factor": orth.closest_factor,
                "residual_oos_ic": m_res["ic"],
                "residual_oos_ic_tstat": m_res["ic_tstat"],
            }
        )
        lib_corr = {
            name: factor_correlation(signed, v * self.library.factors[name].metrics.get("sign", 1.0))
            for name, v in self.library.values.items()
            if name in self.library.factors and self.library.factors[name].status == "accepted"
        }
        factor.metrics["max_library_corr"] = max((abs(c) for c in lib_corr.values()), default=0.0)
        factor.metrics["closest_library_factor"] = max(lib_corr, key=lambda k: abs(lib_corr[k])) if lib_corr else ""
        return factor

    def decide(self, factor: AlphaFactor) -> AlphaFactor:
        cfg, m = self.config, factor.metrics
        reasons = []
        if not self.evaluator.passes_thresholds(factor):
            reasons.append(f"OOS IC {m.get('oos_ic', 0):.4f} (t={m.get('oos_ic_tstat', 0):.2f}) below gate")
        if m.get("is_ic", 0) * m.get("sign", 1) <= 0 or m.get("oos_ic", 0) <= 0:
            reasons.append("IC sign not stable between in-sample and out-of-sample")
        if m.get("max_known_corr", 0) > cfg.max_known_corr:
            reasons.append(f"{m['max_known_corr']:.2f} correlated with known factor {m.get('closest_known_factor')}")
        if m.get("residual_oos_ic", 0) < cfg.min_residual_ic:
            reasons.append(f"residual IC after orthogonalisation {m.get('residual_oos_ic', 0):.4f} too small")
        if m.get("bt_sharpe", 0) <= cfg.min_net_sharpe:
            reasons.append(f"net-of-cost OOS Sharpe {m.get('bt_sharpe', 0):.2f} (turnover {m.get('bt_turnover', 0):.2f}) too low")
        if m.get("max_library_corr", 0) > cfg.max_library_corr:
            reasons.append(f"redundant with accepted factor {m.get('closest_library_factor')}")

        verdict = "accept" if not reasons else "reject"
        if cfg.review_with_llm:
            review = self.evaluator.review(factor, extra_context="Automated checks: " + ("; ".join(reasons) or "all passed"))
            factor.metrics["review"] = review
            if verdict == "accept" and review.get("verdict") != "accept":
                verdict = "reject"
                reasons.append("evaluator agent vetoed: " + "; ".join(map(str, review.get("reasons", [])))[:300])
        factor.status = "accepted" if verdict == "accept" else "rejected"
        factor.notes.extend(reasons)
        return factor

    # ---- orchestration -----------------------------------------------------
    def run_pipeline(
        self,
        theme: str,
        price_data: pd.DataFrame,
        text_panel: Optional[pd.DataFrame] = None,
        num_ideas: Optional[int] = None,
        tickers=None,
        as_of: Optional[str] = None,
    ) -> List[AlphaFactor]:
        rnd = self.memory.next_round()
        panel, fields = self.prepare_panel(price_data, text_panel, as_of)
        self.memory.write("pipeline", "manager", "note", f"round {rnd}: theme '{theme}' on {panel.shape[0]} rows")

        ideas = self.ideator.generate_alpha_ideas(
            theme, num_ideas or self.config.num_ideas, fields, feedback=self.library.feedback(), tickers=tickers, as_of=as_of
        )
        accepted: List[AlphaFactor] = []
        valid = 0
        for idea in ideas:
            factor, values = self.implement_and_validate(idea, panel, fields)
            if values is not None:
                valid += 1
                self.evaluate(factor, values, panel)
                self.decide(factor)
            self.library.add(factor, values)
            log.info("%s -> %s %s", factor.name, factor.status, factor.notes[-1:] if factor.notes else "")
            if factor.status == "accepted":
                accepted.append(factor)

        self.pipeline_history.append(
            {
                "round": rnd,
                "theme": theme,
                "ideas_generated": len(ideas),
                "valid": valid,
                "accepted": len(accepted),
                "as_of": as_of,
                "llm_calls": self.llm.calls,
            }
        )
        return accepted

    def run(self, theme: str, price_data: pd.DataFrame, text_panel=None, rounds: int = 2, **kw) -> List[AlphaFactor]:
        """Several ideation rounds; each round sees the previous rounds' results (evolutionary feedback)."""
        out: List[AlphaFactor] = []
        for _ in range(rounds):
            out.extend(self.run_pipeline(theme, price_data, text_panel, **kw))
        return out

    def get_factor_report(self) -> pd.DataFrame:
        return self.library.report()

    def factor_values(self, factor: AlphaFactor, price_data: pd.DataFrame, text_panel=None, as_of=None) -> pd.Series:
        """Recompute a factor's signed values on new data (e.g. live) through the sandbox."""
        panel, _ = self.prepare_panel(price_data, text_panel, as_of)
        return self._execute(factor.code, panel) * factor.metrics.get("sign", 1.0)
