"""Agent communication protocol: typed messages over an in-process bus, mirrored into shared memory.

The bus is synchronous and in-process; :mod:`alphaagent.streaming` swaps in Kafka for production.
"""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from alphaagent.agents.memory import SharedMemory

BROADCAST = "*"


class MessageType(Enum):
    TASK = "task"          # request work from another agent
    RESULT = "result"      # answer to a TASK (same correlation_id)
    CRITIQUE = "critique"  # review/feedback on another agent's output
    VOTE = "vote"          # ensemble decision ballot
    ALERT = "alert"        # risk / circuit-breaker signal
    INFO = "info"


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    type: MessageType
    content: str
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))

    def reply(self, sender: str, type: MessageType, content: str, payload: Optional[Dict[str, Any]] = None) -> "AgentMessage":
        return AgentMessage(sender, self.sender, type, content, dict(payload or {}), correlation_id=self.correlation_id)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


Subscriber = Callable[[AgentMessage], None]


class MessageBus:
    def __init__(self, memory: Optional[SharedMemory] = None) -> None:
        self.memory = memory
        self._inboxes: Dict[str, Deque[AgentMessage]] = defaultdict(deque)
        self._subscribers: Dict[str, List[Subscriber]] = defaultdict(list)
        self._agents: set = set()
        self.history: List[AgentMessage] = []
        self._lock = threading.Lock()

    def register(self, name: str) -> None:
        self._agents.add(name)

    def subscribe(self, name: str, callback: Subscriber) -> None:
        """Push delivery (in addition to the pull inbox); used by monitors and audit loggers."""
        self._subscribers[name].append(callback)

    def send(self, message: AgentMessage) -> AgentMessage:
        targets = sorted(self._agents - {message.sender}) if message.recipient == BROADCAST else [message.recipient]
        with self._lock:
            self.history.append(message)
            for t in targets:
                self._inboxes[t].append(message)
        if self.memory is not None:
            self.memory.write(
                message.sender,
                message.payload.get("role", ""),
                "message",
                f"[{message.type.value} -> {message.recipient}] {message.content}",
                {"message": message.to_dict()},
            )
        for t in targets + [BROADCAST]:
            for cb in self._subscribers.get(t, []):
                cb(message)
        return message

    def receive(self, name: str, type: Optional[MessageType] = None) -> List[AgentMessage]:
        with self._lock:
            box = self._inboxes[name]
            taken = [m for m in box if type is None or m.type == type]
            self._inboxes[name] = deque(m for m in box if m not in taken)
        return taken

    def thread(self, correlation_id: str) -> List[AgentMessage]:
        return [m for m in self.history if m.correlation_id == correlation_id]
