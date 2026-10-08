"""QuantAgent: retrieval-grounded, chain-of-thought LLM agent with shared memory and messaging."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from alphaagent.agents.memory import SharedMemory
from alphaagent.agents.parsing import extract_confidence, extract_json
from alphaagent.agents.protocol import AgentMessage, MessageBus, MessageType
from alphaagent.agents.types import AgentDecision, AgentRole
from alphaagent.llm.base import LLMClient

COT_INSTRUCTIONS = """## Instructions
Work through this step by step before answering:
1. Identify which retrieved passages are relevant and what they actually say (cite them as [n]).
2. Apply quantitative reasoning and financial domain knowledge; state assumptions explicitly.
3. Check your answer for look-ahead bias, unavailable data, and claims the context does not support.
4. Give your final answer, then a confidence score (0.0-1.0) reflecting the strength of the evidence.

Format: a "## Reasoning" section, then a "## Answer" section{answer_hint}, then a final line exactly like
[Confidence: 0.72]"""


class QuantAgent:
    """Base class. ``retriever`` is optional so agents also work on pure memory/metrics input."""

    role: AgentRole = AgentRole.ANALYST
    system_prompt: str = "You are a rigorous quantitative research agent at a systematic hedge fund."

    def __init__(
        self,
        llm: LLMClient,
        retriever=None,
        memory: Optional[SharedMemory] = None,
        bus: Optional[MessageBus] = None,
        name: Optional[str] = None,
        top_k: int = 8,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.memory = memory if memory is not None else SharedMemory()
        self.bus = bus
        self.name = name or self.role.value
        self.top_k = top_k
        if system_prompt:
            self.system_prompt = system_prompt
        if bus is not None:
            bus.register(self.name)

    # ---- retrieval ---------------------------------------------------
    def retrieve_context(
        self, query: str, tickers: Optional[Sequence[str]] = None, as_of: Optional[str] = None, k: Optional[int] = None
    ) -> str:
        if self.retriever is None:
            return ""
        return self.retriever.context(query, k=k or self.top_k, tickers=tickers, end_date=as_of)

    # ---- reasoning ---------------------------------------------------
    def build_prompt(
        self,
        task: str,
        kind: str,
        retrieved: str,
        context: Optional[str] = None,
        answer_hint: str = "",
        include_memory: bool = True,
    ) -> str:
        parts = [
            f"TASK_KIND: {kind}",
            f"You are the {self.role.value} agent ({self.name}) in a multi-agent quant research team.",
            f"## Task\n{task}",
            f"## Retrieved Financial Context\n{retrieved or 'No relevant documents retrieved.'}",
        ]
        if include_memory:
            mem = self.memory.summary(max_chars=2500)
            if mem:
                parts.append(f"## Shared Team Memory (most recent last)\n{mem}")
        if context:
            parts.append(f"## Additional Context\n{context}")
        parts.append(COT_INSTRUCTIONS.format(answer_hint=answer_hint))
        return "\n\n".join(parts)

    def think(
        self,
        task: str,
        kind: str = "generic",
        context: Optional[str] = None,
        query: Optional[str] = None,
        tickers: Optional[Sequence[str]] = None,
        as_of: Optional[str] = None,
        expect_json: bool = False,
        answer_hint: str = "",
        max_tokens: Optional[int] = None,
        use_retrieval: bool = True,
    ) -> AgentDecision:
        retrieved = self.retrieve_context(query or task, tickers=tickers, as_of=as_of) if use_retrieval else ""
        prompt = self.build_prompt(task, kind, retrieved, context, answer_hint)
        resp = self.llm.complete(prompt, system=self.system_prompt, max_tokens=max_tokens)
        decision = AgentDecision(
            agent_role=self.role,
            timestamp=datetime.utcnow(),
            content=resp.text,
            confidence=extract_confidence(resp.text),
            output=extract_json(resp.text) if expect_json else None,
            metadata={
                "kind": kind,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "retrieved": bool(retrieved),
                "as_of": as_of,
            },
        )
        self.memory.write(
            self.name,
            self.role.value,
            "decision",
            f"{kind}: {_answer_section(resp.text)[:600]}",
            {"confidence": decision.confidence, "kind": kind},
        )
        return decision

    # ---- messaging ---------------------------------------------------
    def send(self, recipient: str, type: MessageType, content: str, payload: Optional[Dict[str, Any]] = None) -> AgentMessage:
        if self.bus is None:
            raise RuntimeError(f"agent {self.name} is not attached to a MessageBus")
        payload = {"role": self.role.value, **(payload or {})}
        return self.bus.send(AgentMessage(self.name, recipient, type, content, payload))

    def handle(self, message: AgentMessage) -> AgentMessage:
        """Default TASK handler: reason about the request and reply with a RESULT."""
        decision = self.think(message.content, kind=message.payload.get("kind", "generic"))
        reply = message.reply(
            self.name, MessageType.RESULT, _answer_section(decision.content), {"confidence": decision.confidence}
        )
        if self.bus is not None:
            self.bus.send(reply)
        return reply

    def process_inbox(self) -> int:
        if self.bus is None:
            return 0
        tasks = self.bus.receive(self.name, MessageType.TASK)
        for m in tasks:
            self.handle(m)
        return len(tasks)


def _answer_section(text: str) -> str:
    marker = "## Answer"
    idx = text.find(marker)
    body = text[idx + len(marker) :] if idx >= 0 else text
    return body.split("[Confidence")[0].strip()
