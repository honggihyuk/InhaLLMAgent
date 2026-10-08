"""Tamper-evident audit trail and regulatory reporting.

Every record is appended to a JSONL file and chained: ``hash = sha256(prev_hash + canonical_json(record))``.
Editing or deleting any past line breaks the chain, which :meth:`AuditTrail.verify` detects.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from alphaagent.llm.base import LLMClient, LLMResponse

GENESIS = "0" * 64

# event types
LLM_CALL = "llm_call"
AGENT_DECISION = "agent_decision"
FACTOR_STATUS = "factor_status"
RISK_ACTION = "risk_action"
CIRCUIT_BREAKER = "circuit_breaker"
KILL_SWITCH = "kill_switch"
HUMAN_OVERRIDE = "human_override"
APPROVAL = "approval"
ORDER = "order"
CONFIG = "config"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditTrail:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash, self._seq = GENESIS, 0
        for rec in self.records():
            self._last_hash, self._seq = rec["hash"], rec["seq"]

    def log(self, event: str, actor: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            body = {
                "seq": self._seq + 1,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event": event,
                "actor": actor,
                "details": details or {},
                "prev_hash": self._last_hash,
            }
            body["hash"] = sha256(self._last_hash + _canonical(body))
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(_canonical(body) + "\n")
            self._last_hash, self._seq = body["hash"], body["seq"]
            return body

    def records(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def verify(self) -> Dict[str, Any]:
        prev, seq = GENESIS, 0
        for rec in self.records():
            claimed = rec.get("hash")
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("prev_hash") != prev or rec.get("seq") != seq + 1 or sha256(prev + _canonical(body)) != claimed:
                return {"ok": False, "records": seq, "broken_at": seq + 1}
            prev, seq = claimed, rec["seq"]
        return {"ok": True, "records": seq, "head": prev}


class AuditedLLM(LLMClient):
    """Wraps any LLMClient and records every call (model, prompt/response hashes, tokens, latency).
    Prompts are hashed, not stored, unless ``store_text=True`` (they can contain licensed data)."""

    def __init__(self, inner: LLMClient, audit: AuditTrail, actor: str = "llm", store_text: bool = False) -> None:
        super().__init__(inner.config)
        self.inner, self.audit, self.actor, self.store_text = inner, audit, actor, store_text

    def _complete(self, prompt, system, max_tokens) -> LLMResponse:
        start = datetime.now(timezone.utc)
        try:
            resp = self.inner.complete(prompt, system, max_tokens)
        except Exception as e:
            self.audit.log(LLM_CALL, self.actor, {"model": self.config.model_name, "prompt_sha256": sha256(prompt),
                                                   "error": str(e)[:500]})
            raise
        details = {
            "model": resp.model or self.config.model_name,
            "kind": _kind(prompt),
            "prompt_sha256": sha256((system or "") + prompt),
            "response_sha256": sha256(resp.text),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "stop_reason": resp.stop_reason,
            "latency_ms": int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
        }
        if self.store_text:
            details.update(prompt=prompt, response=resp.text)
        self.audit.log(LLM_CALL, self.actor, details)
        return resp


def _kind(prompt: str) -> str:
    from alphaagent.llm.mock import task_kind

    return task_kind(prompt)


def regulatory_report(audit: AuditTrail, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    """Summary for algorithmic-trading oversight (e.g. MiFID II Art. 17 / SEC model-risk reviews):
    algorithm inventory, approvals, limit breaches, breaker trips, kill-switch use, human overrides, LLM usage."""
    recs = [r for r in audit.records() if (not start or r["ts"][:10] >= start) and (not end or r["ts"][:10] <= end)]
    by_event = Counter(r["event"] for r in recs)
    llm = [r["details"] for r in recs if r["event"] == LLM_CALL]
    factors: Dict[str, Dict] = {}
    for r in recs:
        if r["event"] == FACTOR_STATUS:
            d = r["details"]
            factors[d.get("name", "?")] = {"status": d.get("status"), "ts": r["ts"], "code_sha256": d.get("code_sha256"),
                                           "approved_by": d.get("approved_by")}
    return {
        "period": {"start": start or (recs[0]["ts"][:10] if recs else None), "end": end or (recs[-1]["ts"][:10] if recs else None)},
        "integrity": audit.verify(),
        "events": dict(by_event),
        "algorithm_inventory": factors,
        "llm_usage": {
            "calls": len(llm),
            "errors": sum(1 for d in llm if "error" in d),
            "input_tokens": sum(int(d.get("input_tokens", 0) or 0) for d in llm),
            "output_tokens": sum(int(d.get("output_tokens", 0) or 0) for d in llm),
            "models": dict(Counter(d.get("model") for d in llm)),
            "by_task": dict(Counter(d.get("kind") for d in llm)),
        },
        "risk_actions": [r for r in recs if r["event"] == RISK_ACTION and r["details"].get("action") != "none"],
        "circuit_breaker_events": [r for r in recs if r["event"] == CIRCUIT_BREAKER],
        "kill_switch_events": [r for r in recs if r["event"] == KILL_SWITCH],
        "human_overrides": [r for r in recs if r["event"] in (HUMAN_OVERRIDE, APPROVAL)],
        "orders": by_event.get(ORDER, 0),
    }


def report_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Algorithmic trading oversight report",
        "",
        f"Period: {report['period']['start']} to {report['period']['end']}",
        f"Audit trail integrity: {'OK' if report['integrity']['ok'] else 'BROKEN at record ' + str(report['integrity'].get('broken_at'))}"
        f" ({report['integrity']['records']} records)",
        "",
        "## Algorithm inventory",
        "| Factor | Status | Code SHA-256 | Approved by | Last change |",
        "|---|---|---|---|---|",
    ]
    for name, f in sorted(report["algorithm_inventory"].items()):
        lines.append(f"| {name} | {f['status']} | {str(f.get('code_sha256') or '')[:12]} | {f.get('approved_by') or ''} | {f['ts']} |")
    u = report["llm_usage"]
    lines += [
        "",
        "## LLM usage",
        f"Calls: {u['calls']} (errors {u['errors']}), tokens in/out: {u['input_tokens']}/{u['output_tokens']}",
        f"Models: {u['models']}",
        "",
        "## Controls",
        f"Risk actions: {len(report['risk_actions'])}",
        f"Circuit-breaker events: {len(report['circuit_breaker_events'])}",
        f"Kill-switch events: {len(report['kill_switch_events'])}",
        f"Human overrides / approvals: {len(report['human_overrides'])}",
        f"Orders recorded: {report['orders']}",
    ]
    for r in report["kill_switch_events"] + report["human_overrides"]:
        lines.append(f"- {r['ts']} {r['event']} by {r['actor']}: {json.dumps(r['details'], default=str)[:200]}")
    return "\n".join(lines) + "\n"
