"""Shared memory M: every agent action is appended per round, M(t) = M(t-1) ∪ {a_1(t), ..., a_n(t)}."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    round: int
    agent: str
    role: str
    kind: str  # e.g. decision | idea | factor | evaluation | message | note
    content: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SharedMemory:
    def __init__(self, path: Optional[str] = None) -> None:
        self._entries: List[MemoryEntry] = []
        self._lock = threading.Lock()
        self.round = 0
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self._entries)

    def next_round(self) -> int:
        with self._lock:
            self.round += 1
            return self.round

    def write(self, agent: str, role: str, kind: str, content: str, payload: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        entry = MemoryEntry(self.round, agent, role, kind, content, dict(payload or {}))
        with self._lock:
            self._entries.append(entry)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")
        return entry

    def read(
        self,
        role: Optional[str] = None,
        kind: Optional[str] = None,
        agent: Optional[str] = None,
        since_round: Optional[int] = None,
        last_n: Optional[int] = None,
    ) -> List[MemoryEntry]:
        with self._lock:
            out = [
                e
                for e in self._entries
                if (role is None or e.role == role)
                and (kind is None or e.kind == kind)
                and (agent is None or e.agent == agent)
                and (since_round is None or e.round >= since_round)
            ]
        return out[-last_n:] if last_n else out

    def summary(self, max_chars: int = 3000, kinds: Optional[List[str]] = None, last_n: int = 20) -> str:
        """Most recent entries rendered compactly for inclusion in an agent prompt."""
        entries = [e for e in self.read(last_n=None) if kinds is None or e.kind in kinds][-last_n:]
        lines: List[str] = []
        used = 0
        for e in reversed(entries):
            line = f"- (round {e.round}, {e.role}/{e.kind}) {' '.join(e.content.split())[:300]}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(reversed(lines))

    @classmethod
    def load(cls, path: str) -> "SharedMemory":
        mem = cls(path=None)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = MemoryEntry(**json.loads(line))
                mem._entries.append(e)
                mem.round = max(mem.round, e.round)
        mem.path = Path(path)
        return mem
