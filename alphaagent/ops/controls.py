"""Circuit breakers, kill switch, and human-in-the-loop overrides.

* ``CircuitBreaker``  - classic closed/open/half-open breaker for flaky dependencies (LLM API, data feeds)
* ``ResilientLLM``    - LLMClient guarded by a breaker, with an optional fallback client
* ``TradingGuard``    - pre-trade checks that halt or shrink the book (loss limits, drawdown, stale data,
                        turnover spikes, risk-agent halt, kill switch)
* ``ControlCenter``   - persistent operator state: kill switch, factor pauses, scaling overrides, and an
                        approval queue for trades above a size threshold. Every change is audited.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from alphaagent.llm.base import LLMClient, LLMError, LLMResponse
from alphaagent.ops.audit import APPROVAL, CIRCUIT_BREAKER, HUMAN_OVERRIDE, KILL_SWITCH, AuditTrail


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        reset_timeout: float = 60.0,
        audit: Optional[AuditTrail] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.audit = audit
        self.clock = clock
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._lock = threading.Lock()

    def _transition(self, state: str, reason: str) -> None:
        if state != self.state:
            self.state = state
            if self.audit:
                self.audit.log(CIRCUIT_BREAKER, self.name, {"state": state, "reason": reason[:300]})

    def allow(self) -> bool:
        with self._lock:
            if self.state == self.OPEN and self.clock() - self.opened_at >= self.reset_timeout:
                self._transition(self.HALF_OPEN, "reset timeout elapsed; probing")
            return self.state != self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self._transition(self.CLOSED, "probe succeeded")

    def record_failure(self, reason: str = "") -> None:
        with self._lock:
            self.failures += 1
            if self.state == self.HALF_OPEN or self.failures >= self.failure_threshold:
                self.opened_at = self.clock()
                self._transition(self.OPEN, reason or f"{self.failures} consecutive failures")

    def call(self, fn: Callable, *args, **kwargs):
        if not self.allow():
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            self.record_failure(str(e))
            raise
        self.record_success()
        return out


class ResilientLLM(LLMClient):
    def __init__(self, primary: LLMClient, breaker: Optional[CircuitBreaker] = None, fallback: Optional[LLMClient] = None):
        super().__init__(primary.config)
        self.primary = primary
        self.breaker = breaker or CircuitBreaker("llm")
        self.fallback = fallback

    def _complete(self, prompt, system, max_tokens) -> LLMResponse:
        try:
            return self.breaker.call(self.primary.complete, prompt, system, max_tokens)
        except (CircuitOpenError, LLMError, Exception) as e:
            if self.fallback is None:
                raise
            resp = self.fallback.complete(prompt, system, max_tokens)
            resp.stop_reason = resp.stop_reason or f"fallback after: {type(e).__name__}"
            return resp


# ---------------------------------------------------------------------------------------------------------
@dataclass
class GuardLimits:
    max_daily_loss: float = 0.02  # fraction of capital
    max_drawdown: float = 0.10
    max_turnover: float = 1.0  # sum |delta w| per rebalance
    max_data_age_days: int = 3  # signal/price staleness
    max_gross: float = 2.0


@dataclass
class GuardDecision:
    allowed: bool
    scale: float
    reasons: List[str] = field(default_factory=list)


class TradingGuard:
    """Trading circuit breaker: evaluated before every rebalance. Trips stay latched until a human resets."""

    def __init__(self, limits: Optional[GuardLimits] = None, control: Optional["ControlCenter"] = None,
                 audit: Optional[AuditTrail] = None) -> None:
        self.limits = limits or GuardLimits()
        self.control = control
        self.audit = audit
        self.tripped: Optional[str] = None

    def check(
        self,
        target: pd.Series,
        current: pd.Series,
        pnl_history: pd.Series,
        as_of: pd.Timestamp,
        data_timestamp: pd.Timestamp,
        risk_action: str = "none",
    ) -> GuardDecision:
        lim, reasons = self.limits, []
        if self.control is not None and self.control.state.kill_switch:
            return GuardDecision(False, 0.0, [f"kill switch engaged by {self.control.state.kill_switch_by}"])
        if self.tripped:
            return GuardDecision(False, 0.0, [f"breaker latched: {self.tripped} (needs human reset)"])

        hist = pnl_history.dropna()
        if len(hist) and hist.iloc[-1] <= -lim.max_daily_loss:
            reasons.append(f"daily loss {hist.iloc[-1]:.2%} beyond {lim.max_daily_loss:.0%}")
        if len(hist):
            eq = (1 + hist).cumprod()
            dd = float(eq.iloc[-1] / eq.max() - 1)
            if dd <= -lim.max_drawdown:
                reasons.append(f"drawdown {dd:.1%} beyond {lim.max_drawdown:.0%}")
        age = (pd.Timestamp(as_of) - pd.Timestamp(data_timestamp)).days
        if age > lim.max_data_age_days:
            reasons.append(f"input data is {age} days old")
        if risk_action == "halt":
            reasons.append("risk agent requested halt")
        if reasons:
            self.tripped = "; ".join(reasons)
            if self.audit:
                self.audit.log(CIRCUIT_BREAKER, "trading_guard", {"state": "tripped", "reasons": reasons, "as_of": str(as_of)})
            return GuardDecision(False, 0.0, reasons)

        scale = 1.0
        turnover = float((target.reindex(current.index.union(target.index), fill_value=0)
                          - current.reindex(current.index.union(target.index), fill_value=0)).abs().sum())
        if turnover > lim.max_turnover:
            scale = lim.max_turnover / turnover  # move only part of the way towards the target
            reasons.append(f"turnover {turnover:.2f} capped at {lim.max_turnover:.2f}")
        gross = float(target.abs().sum())
        if gross > lim.max_gross:
            reasons.append(f"gross {gross:.2f} above {lim.max_gross:.2f}")
        return GuardDecision(True, scale, reasons)

    def reset(self, operator: str, reason: str) -> None:
        if self.audit:
            self.audit.log(HUMAN_OVERRIDE, operator, {"action": "reset_trading_guard", "was": self.tripped, "reason": reason})
        self.tripped = None


# ---------------------------------------------------------------------------------------------------------
@dataclass
class Approval:
    approval_id: str
    created_at: str
    description: str
    payload: Dict
    status: str = "pending"  # pending | approved | rejected
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    note: str = ""


@dataclass
class ControlState:
    kill_switch: bool = False
    kill_switch_by: Optional[str] = None
    kill_switch_reason: str = ""
    paused_factors: List[str] = field(default_factory=list)
    book_scale: float = 1.0  # operator multiplier on all positions (<= 1 by policy)
    approvals: Dict[str, Approval] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ControlCenter:
    """Persistent human-override state. Every mutation requires an operator name and is audited."""

    def __init__(self, path: Optional[str] = None, audit: Optional[AuditTrail] = None,
                 approval_threshold: float = 0.02) -> None:
        self.path = Path(path) if path else None
        self.audit = audit
        self.approval_threshold = approval_threshold  # |delta weight| per name that needs a human
        self._lock = threading.Lock()
        self.state = ControlState()
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            raw["approvals"] = {k: Approval(**v) for k, v in raw.get("approvals", {}).items()}
            self.state = ControlState(**raw)

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self.state)
            self.path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")

    def _audit(self, event: str, operator: str, details: Dict) -> None:
        if self.audit:
            self.audit.log(event, operator, details)

    @staticmethod
    def _require(operator: str) -> None:
        if not operator or not operator.strip():
            raise ValueError("an operator name is required for every manual control action")

    def engage_kill_switch(self, operator: str, reason: str) -> None:
        self._require(operator)
        with self._lock:
            self.state.kill_switch, self.state.kill_switch_by, self.state.kill_switch_reason = True, operator, reason
            self._save()
        self._audit(KILL_SWITCH, operator, {"engaged": True, "reason": reason})

    def release_kill_switch(self, operator: str, reason: str) -> None:
        self._require(operator)
        with self._lock:
            self.state.kill_switch, self.state.kill_switch_by, self.state.kill_switch_reason = False, None, ""
            self._save()
        self._audit(KILL_SWITCH, operator, {"engaged": False, "reason": reason})

    def pause_factor(self, name: str, operator: str, reason: str = "") -> None:
        self._require(operator)
        with self._lock:
            if name not in self.state.paused_factors:
                self.state.paused_factors.append(name)
            self._save()
        self._audit(HUMAN_OVERRIDE, operator, {"action": "pause_factor", "factor": name, "reason": reason})

    def resume_factor(self, name: str, operator: str, reason: str = "") -> None:
        self._require(operator)
        with self._lock:
            self.state.paused_factors = [f for f in self.state.paused_factors if f != name]
            self._save()
        self._audit(HUMAN_OVERRIDE, operator, {"action": "resume_factor", "factor": name, "reason": reason})

    def set_book_scale(self, scale: float, operator: str, reason: str = "") -> None:
        self._require(operator)
        scale = float(min(max(scale, 0.0), 1.0))  # operators can de-risk; sizing up goes through config review
        with self._lock:
            self.state.book_scale = scale
            self._save()
        self._audit(HUMAN_OVERRIDE, operator, {"action": "set_book_scale", "scale": scale, "reason": reason})

    # ---- approvals ----------------------------------------------------------------
    def needs_approval(self, target: pd.Series, current: pd.Series) -> pd.Series:
        idx = current.index.union(target.index)
        delta = target.reindex(idx, fill_value=0) - current.reindex(idx, fill_value=0)
        return delta[delta.abs() > self.approval_threshold]

    def request_approval(self, description: str, payload: Dict) -> Approval:
        ap = Approval(uuid.uuid4().hex[:10], _now(), description, payload)
        with self._lock:
            self.state.approvals[ap.approval_id] = ap
            self._save()
        self._audit(APPROVAL, "system", {"approval_id": ap.approval_id, "status": "requested", "description": description})
        return ap

    def decide(self, approval_id: str, approve: bool, operator: str, note: str = "") -> Approval:
        self._require(operator)
        with self._lock:
            ap = self.state.approvals[approval_id]
            if ap.status != "pending":
                raise ValueError(f"approval {approval_id} already {ap.status}")
            ap.status = "approved" if approve else "rejected"
            ap.decided_by, ap.decided_at, ap.note = operator, _now(), note
            self._save()
        self._audit(APPROVAL, operator, {"approval_id": approval_id, "status": ap.status, "note": note})
        return ap

    def pending(self) -> List[Approval]:
        return [a for a in self.state.approvals.values() if a.status == "pending"]
