"""Manager agent: decomposes a research goal into sub-tasks and synthesises analysts' results (FinCon style).

Strategy = Manager( ∪_i Analyst_i(Task_i) )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from alphaagent.agents.base import QuantAgent, _answer_section
from alphaagent.agents.protocol import AgentMessage, MessageType
from alphaagent.agents.types import AgentRole

ANALYST_SPECIALTIES = ("sentiment", "fundamental", "technical", "risk")


@dataclass
class SubTask:
    task_id: str
    theme: str  # research theme handed to the alpha pipeline / analyst
    specialty: str = "sentiment"  # which analyst type owns it
    rationale: str = ""
    priority: int = 1
    tickers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class ManagerAgent(QuantAgent):
    role = AgentRole.MANAGER
    system_prompt = (
        "You are the head of a systematic research team. You break broad research goals into focused, testable "
        "sub-problems, assign each to the right specialist, and synthesise their findings into one coherent "
        "strategy while keeping risk and capacity in mind."
    )

    def decompose(self, goal: str, max_tasks: int = 4, tickers: Optional[List[str]] = None, as_of: Optional[str] = None) -> List[SubTask]:
        task = f"""Research goal: {goal}

Break this goal into at most {max_tasks} focused sub-tasks. Each sub-task must be a concrete research theme that
a specialist can turn into testable alpha factors from earnings calls, filings, news, and price/volume data.

Specialists available: {", ".join(ANALYST_SPECIALTIES)}.
For each sub-task give: task_id (t1, t2, ...), theme, specialty, rationale, priority (1 = highest)."""
        decision = self.think(
            task,
            kind="decomposition",
            query=goal,
            tickers=tickers,
            as_of=as_of,
            expect_json=True,
            answer_hint=' containing a ```json block: {"tasks": [ {...}, ... ]}',
        )
        raw = decision.output.get("tasks", []) if isinstance(decision.output, dict) else (decision.output or [])
        tasks: List[SubTask] = []
        for i, d in enumerate(raw[:max_tasks]):
            if not isinstance(d, dict) or not d.get("theme"):
                continue
            spec = str(d.get("specialty", "sentiment")).lower()
            tasks.append(
                SubTask(
                    task_id=str(d.get("task_id") or f"t{i + 1}"),
                    theme=str(d["theme"]),
                    specialty=spec if spec in ANALYST_SPECIALTIES else "sentiment",
                    rationale=str(d.get("rationale", "")),
                    priority=int(d.get("priority", i + 1) or i + 1),
                    tickers=list(tickers or []),
                )
            )
        if not tasks:  # never stall the team on a malformed plan
            tasks = [SubTask("t1", goal, "sentiment", "fallback: goal used as a single theme")]
        tasks.sort(key=lambda t: t.priority)
        self.memory.write(self.name, self.role.value, "plan", f"{len(tasks)} sub-tasks for '{goal}'",
                          {"tasks": [t.to_dict() for t in tasks]})
        return tasks

    def dispatch(self, tasks: List[SubTask], assignees: Dict[str, str]) -> List[AgentMessage]:
        """Send each sub-task as a TASK message to the agent registered for its specialty."""
        sent = []
        for t in tasks:
            recipient = assignees.get(t.specialty) or next(iter(assignees.values()))
            sent.append(self.send(recipient, MessageType.TASK, t.theme, {"task": t.to_dict(), "kind": "analysis"}))
        return sent

    def synthesize(self, goal: str, results: List[Dict]) -> Dict:
        task = f"""Research goal: {goal}

Results from the specialists (factors found, their validation metrics, and analyst notes):
{json.dumps(results, indent=1, default=str)[:6000]}

Synthesise a strategy: which factors to deploy and with what relative emphasis, what the main risks are,
and what the team should research next."""
        decision = self.think(
            task,
            kind="synthesis",
            use_retrieval=False,
            expect_json=True,
            answer_hint=' containing a ```json block: {"deploy": [names], "emphasis": {name: weight}, "risks": [...], "next_steps": [...]}',
        )
        out = decision.output if isinstance(decision.output, dict) else {}
        out.setdefault("summary", _answer_section(decision.content)[:1500])
        out["confidence"] = decision.confidence
        self.memory.write(self.name, self.role.value, "strategy", f"strategy for '{goal}': deploy {out.get('deploy')}", out)
        return out
