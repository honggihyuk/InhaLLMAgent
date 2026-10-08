"""Real-time document ingestion over the broker.

producers (SEC / RSS / transcript pollers) --docs.raw--> IngestionConsumer
    -> embed + index (vector store)          --docs.indexed-->
    -> LLM text scoring (optional)           --signals.text-->  feature store / live pipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from alphaagent.agents.protocol import AgentMessage, MessageBus, MessageType
from alphaagent.documents import Document
from alphaagent.ingestion.base import DocumentSource
from alphaagent.ingestion.pipeline import IngestionPipeline
from alphaagent.streaming.broker import (
    TOPIC_AGENT_MESSAGES,
    TOPIC_DOCS_INDEXED,
    TOPIC_DOCS_RAW,
    TOPIC_SIGNALS,
    MessageBroker,
)

log = logging.getLogger(__name__)


class DocumentProducer:
    def __init__(self, broker: MessageBroker, topic: str = TOPIC_DOCS_RAW) -> None:
        self.broker = broker
        self.topic = topic
        self.sent = 0

    def send(self, doc: Document) -> None:
        self.broker.publish(self.topic, doc.to_dict(), key=doc.doc_id)
        self.sent += 1

    def send_all(self, docs: Iterable[Document]) -> int:
        n = 0
        for d in docs:
            self.send(d)
            n += 1
        self.broker.flush()
        return n


def poll_sources(
    sources: List[DocumentSource],
    producer: DocumentProducer,
    interval: float = 300.0,
    iterations: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll each source every ``interval`` seconds and publish what it yields. The consumer de-duplicates."""
    total, i = 0, 0
    while iterations is None or i < iterations:
        for src in sources:
            try:
                total += producer.send_all(src.load())
            except Exception as e:  # one failing source must not stop the others
                log.warning("source %s failed: %s", getattr(src, "name", src), e)
        i += 1
        if iterations is None or i < iterations:
            sleep(interval)
    return total


@dataclass
class ConsumerStats:
    batches: int = 0
    received: int = 0
    indexed: int = 0
    duplicates: int = 0
    scored: int = 0
    errors: int = 0


class IngestionConsumer:
    """At-least-once consumer: offsets are committed only after the whole batch is indexed (and saved)."""

    def __init__(
        self,
        broker: MessageBroker,
        ingestion: IngestionPipeline,
        scorer=None,  # TextSignalExtractor
        group: str = "ingestion",
        on_batch: Optional[Callable[[], None]] = None,  # e.g. persist the vector store
    ) -> None:
        self.broker = broker
        self.ingestion = ingestion
        self.scorer = scorer
        self.group = group
        self.on_batch = on_batch
        self.stats = ConsumerStats()

    def run_once(self, max_records: int = 200, timeout: float = 1.0) -> int:
        records = self.broker.poll([TOPIC_DOCS_RAW], self.group, max_records, timeout)
        if not records:
            return 0
        docs: List[Document] = []
        for r in records:
            try:
                docs.append(Document(**r.value))
            except (TypeError, ValueError) as e:
                self.stats.errors += 1
                log.warning("bad document at offset %s: %s", r.offset, e)
        unique = {d.doc_id: d for d in docs}  # redelivery can repeat a document inside one batch
        new_docs = [d for d in unique.values() if not self.ingestion.store.has_document(d.doc_id)]
        stats = self.ingestion.ingest(new_docs)
        self.stats.duplicates += len(docs) - len(new_docs)
        for d in new_docs:
            self.broker.publish(TOPIC_DOCS_INDEXED, {"doc_id": d.doc_id, "ticker": d.ticker, "published_at": d.published_at})
            if self.scorer is not None:
                s = self.scorer.score_document(d)
                self.broker.publish(TOPIC_SIGNALS, s.__dict__, key=d.ticker)
                self.stats.scored += 1
        if self.on_batch:
            self.on_batch()
        self.broker.commit(self.group)
        self.stats.batches += 1
        self.stats.received += len(records)
        self.stats.indexed += stats.documents_added
        return len(records)

    def run_forever(self, idle_sleep: float = 1.0, stop: Optional[Callable[[], bool]] = None) -> None:
        while not (stop and stop()):
            if not self.run_once():
                time.sleep(idle_sleep)


class BrokerMessageBus(MessageBus):
    """MessageBus that also mirrors every agent message to the broker, so agents in different pods share it."""

    def __init__(self, broker: MessageBroker, memory=None, topic: str = TOPIC_AGENT_MESSAGES) -> None:
        super().__init__(memory)
        self.broker = broker
        self.topic = topic

    def send(self, message: AgentMessage) -> AgentMessage:
        self.broker.publish(self.topic, message.to_dict(), key=message.correlation_id)
        return super().send(message)

    def pull_remote(self, group: str) -> int:
        """Deliver messages published by other processes into the local inboxes."""
        n = 0
        for r in self.broker.poll([self.topic], group):
            v = dict(r.value)
            if any(m.message_id == v.get("message_id") for m in self.history):
                continue  # our own message echoed back
            v["type"] = MessageType(v["type"])
            super().send(AgentMessage(**v))
            n += 1
        self.broker.commit(group)
        return n
