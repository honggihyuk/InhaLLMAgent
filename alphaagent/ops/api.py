"""Operator API: kill switch, factor pause/resume, book scaling, approvals, audit verification and reports.

Mount next to the retrieval API: ``app.include_router(ops_router(control, audit, guard))``.
Put it behind your SSO / mTLS ingress; the ``X-Operator`` header is recorded in the audit trail.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from alphaagent.ops.audit import AuditTrail, regulatory_report, report_markdown
from alphaagent.ops.controls import ControlCenter, TradingGuard


class Reason(BaseModel):
    reason: str = ""


class Scale(BaseModel):
    scale: float
    reason: str = ""


class Decision(BaseModel):
    approve: bool
    note: str = ""


def ops_router(control: ControlCenter, audit: AuditTrail, guard: Optional[TradingGuard] = None) -> APIRouter:
    r = APIRouter(prefix="/ops", tags=["ops"])

    def op(x_operator: Optional[str]) -> str:
        if not x_operator:
            raise HTTPException(401, "X-Operator header required")
        return x_operator

    @r.get("/status")
    def status():
        s = control.state
        return {
            "kill_switch": s.kill_switch,
            "kill_switch_by": s.kill_switch_by,
            "paused_factors": s.paused_factors,
            "book_scale": s.book_scale,
            "pending_approvals": len(control.pending()),
            "trading_guard_tripped": guard.tripped if guard else None,
        }

    @r.post("/kill-switch/engage")
    def engage(body: Reason, x_operator: Optional[str] = Header(None)):
        control.engage_kill_switch(op(x_operator), body.reason)
        return status()

    @r.post("/kill-switch/release")
    def release(body: Reason, x_operator: Optional[str] = Header(None)):
        control.release_kill_switch(op(x_operator), body.reason)
        return status()

    @r.post("/factors/{name}/pause")
    def pause(name: str, body: Reason, x_operator: Optional[str] = Header(None)):
        control.pause_factor(name, op(x_operator), body.reason)
        return status()

    @r.post("/factors/{name}/resume")
    def resume(name: str, body: Reason, x_operator: Optional[str] = Header(None)):
        control.resume_factor(name, op(x_operator), body.reason)
        return status()

    @r.post("/book-scale")
    def book_scale(body: Scale, x_operator: Optional[str] = Header(None)):
        control.set_book_scale(body.scale, op(x_operator), body.reason)
        return status()

    @r.post("/guard/reset")
    def guard_reset(body: Reason, x_operator: Optional[str] = Header(None)):
        if guard is None:
            raise HTTPException(404, "no trading guard configured")
        guard.reset(op(x_operator), body.reason)
        return status()

    @r.get("/approvals")
    def approvals():
        return [a.__dict__ for a in control.pending()]

    @r.post("/approvals/{approval_id}")
    def decide(approval_id: str, body: Decision, x_operator: Optional[str] = Header(None)):
        try:
            return control.decide(approval_id, body.approve, op(x_operator), body.note).__dict__
        except KeyError:
            raise HTTPException(404, "unknown approval")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @r.get("/audit/verify")
    def verify():
        return audit.verify()

    @r.get("/report")
    def report(start: Optional[str] = None, end: Optional[str] = None):
        return regulatory_report(audit, start, end)

    @r.get("/report.md", response_class=PlainTextResponse)
    def report_md(start: Optional[str] = None, end: Optional[str] = None):
        return report_markdown(regulatory_report(audit, start, end))

    return r
