from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(Enum):
    IDEATOR = "ideator"
    IMPLEMENTER = "implementer"
    EVALUATOR = "evaluator"
    PORTFOLIO = "portfolio"
    MANAGER = "manager"
    RISK = "risk"
    ANALYST = "analyst"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:60] or "alpha"


@dataclass
class AlphaIdea:
    name: str
    description: str
    expression: str = ""
    data_fields: List[str] = field(default_factory=list)
    rationale: str = ""
    expected_behavior: str = ""
    theme: str = ""
    evidence: List[str] = field(default_factory=list)  # retrieved-source citations backing the idea

    def __post_init__(self) -> None:
        self.name = slugify(self.name)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], theme: str = "") -> "AlphaIdea":
        fields = d.get("data_fields") or d.get("data_requirements") or []
        if isinstance(fields, str):
            fields = [f.strip() for f in re.split(r"[,\s]+", fields) if f.strip()]
        evidence = d.get("evidence") or []
        return cls(
            name=str(d.get("name") or "alpha"),
            description=str(d.get("description") or ""),
            expression=str(d.get("expression") or d.get("formula") or ""),
            data_fields=list(fields),
            rationale=str(d.get("rationale") or d.get("economic_intuition") or ""),
            expected_behavior=str(d.get("expected_behavior") or ""),
            theme=theme or str(d.get("theme") or ""),
            evidence=[str(e) for e in (evidence if isinstance(evidence, list) else [evidence])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlphaFactor:
    name: str
    code: str
    description: str = ""
    expression: str = ""
    data_fields: List[str] = field(default_factory=list)
    author_agent: AgentRole = AgentRole.IMPLEMENTER
    theme: str = ""
    # performance
    ic_score: Optional[float] = None
    rank_ic: Optional[float] = None
    icir: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    turnover: Optional[float] = None
    # lifecycle
    status: str = "implemented"  # implemented | invalid | evaluated | accepted | rejected
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["author_agent"] = self.author_agent.value
        return d


@dataclass
class AgentDecision:
    agent_role: AgentRole
    timestamp: datetime
    content: str
    confidence: float
    output: Any = None  # parsed structured answer (JSON) when the task asked for one
    metadata: Dict[str, Any] = field(default_factory=dict)
