"""Message broker abstraction: Kafka in production, in-memory for tests and single-process runs.

Topics used by the system:
    docs.raw          new documents from pollers (SEC, RSS, transcript drops)
    docs.indexed      documents embedded into the vector store
    signals.text      LLM document scores (sentiment / guidance / risk)
    agents.messages   inter-agent protocol messages (distributed MessageBus)
    ops.alerts        risk alerts, circuit-breaker trips, kill-switch events
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

TOPIC_DOCS_RAW = "docs.raw"
TOPIC_DOCS_INDEXED = "docs.indexed"
TOPIC_SIGNALS = "signals.text"
TOPIC_AGENT_MESSAGES = "agents.messages"
TOPIC_ALERTS = "ops.alerts"


@dataclass
class Record:
    topic: str
    key: Optional[str]
    value: Dict[str, Any]
    offset: int = -1


class MessageBroker(ABC):
    @abstractmethod
    def publish(self, topic: str, value: Dict[str, Any], key: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def poll(self, topics: Iterable[str], group: str, max_records: int = 500, timeout: float = 1.0) -> List[Record]:
        ...

    def commit(self, group: str) -> None:  # at-least-once: commit after processing
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class InMemoryBroker(MessageBroker):
    """Append-only per-topic logs with per-group committed offsets (Kafka semantics, one partition)."""

    def __init__(self) -> None:
        self.logs: Dict[str, List[Record]] = defaultdict(list)
        self._pending: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._committed: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._lock = threading.Lock()

    def publish(self, topic, value, key=None):
        with self._lock:
            log = self.logs[topic]
            # round-trip through JSON so producers can't share mutable state with consumers
            log.append(Record(topic, key, json.loads(json.dumps(value, default=str)), len(log)))

    def poll(self, topics, group, max_records=500, timeout=1.0):
        out: List[Record] = []
        with self._lock:
            for t in topics:
                start = self._pending[group].get(t, self._committed[group].get(t, 0))
                batch = self.logs[t][start : start + max_records - len(out)]
                out.extend(batch)
                self._pending[group][t] = start + len(batch)
        return out

    def commit(self, group):
        with self._lock:
            self._committed[group].update(self._pending[group])

    def rewind(self, group: str) -> None:
        """Drop uncommitted progress (simulates a consumer crash before commit)."""
        with self._lock:
            self._pending[group] = dict(self._committed[group])


class KafkaBroker(MessageBroker):
    """confluent-kafka backed broker. ``pip install confluent-kafka``.

    Producer: idempotent, acks=all. Consumer: manual commit (at-least-once); handlers must be idempotent,
    which the ingestion consumer is (documents are keyed by doc_id and de-duplicated by the store).
    """

    def __init__(self, bootstrap_servers: str, client_id: str = "alphaagent", extra: Optional[Dict] = None) -> None:
        try:
            from confluent_kafka import Consumer, Producer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install confluent-kafka to use KafkaBroker") from e
        self._Consumer = Consumer
        self.conf = {"bootstrap.servers": bootstrap_servers, "client.id": client_id, **(extra or {})}
        self.producer = Producer({**self.conf, "enable.idempotence": True, "acks": "all"})
        self._consumers: Dict[str, Any] = {}

    def publish(self, topic, value, key=None):
        self.producer.produce(topic, json.dumps(value, default=str).encode("utf-8"), key=key.encode() if key else None)
        self.producer.poll(0)

    def _consumer(self, group: str, topics: Iterable[str]):
        if group not in self._consumers:
            c = self._Consumer({**self.conf, "group.id": group, "enable.auto.commit": False, "auto.offset.reset": "earliest"})
            c.subscribe(list(topics))
            self._consumers[group] = c
        return self._consumers[group]

    def poll(self, topics, group, max_records=500, timeout=1.0):
        msgs = self._consumer(group, topics).consume(num_messages=max_records, timeout=timeout)
        out = []
        for m in msgs:
            if m.error():
                continue
            out.append(Record(m.topic(), m.key().decode() if m.key() else None, json.loads(m.value()), m.offset()))
        return out

    def commit(self, group):
        if group in self._consumers:
            self._consumers[group].commit(asynchronous=False)

    def flush(self):
        self.producer.flush(10)

    def close(self):
        self.flush()
        for c in self._consumers.values():
            c.close()


def build_broker(url: Optional[str] = None) -> MessageBroker:
    """``kafka://host:9092,host2:9092`` -> KafkaBroker; ``memory://`` or None -> InMemoryBroker."""
    if not url or url.startswith("memory://"):
        return InMemoryBroker()
    if url.startswith("kafka://"):
        return KafkaBroker(url[len("kafka://"):])
    raise ValueError(f"unsupported broker url {url!r}")


Handler = Callable[[List[Record]], None]
