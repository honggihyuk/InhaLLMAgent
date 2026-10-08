import json
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from alphaagent.agents import (
    AgentMessage,
    AgentRole,
    AlphaFactor,
    AlphaIdea,
    EvaluationAgent,
    IdeationAgent,
    ImplementationAgent,
    MessageBus,
    MessageType,
    QuantAgent,
    SharedMemory,
)
from alphaagent.agents.parsing import extract_code, extract_confidence, extract_json
from alphaagent.backtest import summarize_factor
from alphaagent.data import create_synthetic_market_data
from alphaagent.llm import LLMConfig, MockLLM, ScriptedLLM, build_llm
from alphaagent.llm.anthropic_client import FALLBACK_BETA, AnthropicLLM
from alphaagent.llm.base import LLMError
from alphaagent.llm.vllm_client import VLLMClient


# ---- parsing ---------------------------------------------------------
def test_extract_helpers():
    text = 'blah\n```json\n{"a": [1, 2]}\n```\n```python\ndef f():\n    return 1\n```\n[Confidence: 0.83]'
    assert extract_json(text) == {"a": [1, 2]}
    assert extract_code(text).startswith("def f()")
    assert extract_confidence(text) == pytest.approx(0.83)
    assert extract_confidence("[Confidence: 85]") == pytest.approx(0.85)
    assert extract_confidence("none") == 0.5
    assert extract_json('Answer: {"verdict": "accept", "x": "a}b"} trailing') == {"verdict": "accept", "x": "a}b"}


# ---- memory & protocol -----------------------------------------------
def test_shared_memory_rounds_and_persistence(tmp_path):
    path = tmp_path / "mem.jsonl"
    mem = SharedMemory(str(path))
    mem.write("a", "ideator", "idea", "first")
    mem.next_round()
    mem.write("b", "evaluator", "evaluation", "second")
    assert [e.round for e in mem.read()] == [0, 1]
    assert len(mem.read(role="evaluator")) == 1
    assert "second" in mem.summary()
    restored = SharedMemory.load(str(path))
    assert len(restored) == 2 and restored.round == 1


def test_message_bus_routing_and_broadcast():
    mem = SharedMemory()
    bus = MessageBus(mem)
    for n in ("manager", "ideator", "evaluator"):
        bus.register(n)
    seen = []
    bus.subscribe("*", seen.append)
    task = bus.send(AgentMessage("manager", "ideator", MessageType.TASK, "find ideas"))
    bus.send(AgentMessage("manager", "*", MessageType.INFO, "new round"))
    inbox = bus.receive("ideator")
    assert [m.type for m in inbox] == [MessageType.TASK, MessageType.INFO]
    assert [m.type for m in bus.receive("evaluator")] == [MessageType.INFO]
    assert bus.receive("ideator") == []
    reply = task.reply("ideator", MessageType.RESULT, "done")
    bus.send(reply)
    assert [m.content for m in bus.thread(task.correlation_id)] == ["find ideas", "done"]
    assert len(seen) == 3 and len(mem.read(kind="message")) == 3


def test_agent_handles_task_via_bus():
    bus = MessageBus(SharedMemory())
    agent = QuantAgent(MockLLM(), bus=bus, name="analyst")
    bus.register("manager")
    bus.send(AgentMessage("manager", "analyst", MessageType.TASK, "summarise"))
    assert agent.process_inbox() == 1
    (result,) = bus.receive("manager")
    assert result.type == MessageType.RESULT and "Acknowledged" in result.content


# ---- QuantAgent CoT + retrieval ----------------------------------------
def test_cot_prompt_contains_retrieval_memory_and_point_in_time(loaded):
    retriever, _ = loaded
    llm = ScriptedLLM(["## Reasoning\n...\n## Answer\nok\n[Confidence: 0.9]"])
    mem = SharedMemory()
    mem.write("x", "manager", "note", "focus on guidance cuts")
    agent = QuantAgent(llm, retriever=retriever, memory=mem)
    decision = agent.think("guidance cut liquidity", tickers=["GLBX"], as_of="2026-02-04")
    prompt = llm.prompts[0]
    assert "TASK_KIND: generic" in prompt and "step by step" in prompt
    assert "Globex" in prompt and "focus on guidance cuts" in prompt
    assert "2026-02-05" not in prompt  # news published after as_of must not leak
    assert decision.confidence == pytest.approx(0.9)
    assert mem.read(kind="decision")[-1].payload["confidence"] == pytest.approx(0.9)


# ---- specialist agents ---------------------------------------------------
def test_ideation_parses_json_and_uses_feedback(loaded):
    retriever, _ = loaded
    llm = MockLLM()
    ideator = IdeationAgent(llm, retriever=retriever)
    ideas = ideator.generate_alpha_ideas("price reversal", num_ideas=2)
    assert len(ideas) == 2 and all(isinstance(i, AlphaIdea) for i in ideas)
    again = ideator.generate_alpha_ideas("price reversal", num_ideas=2, feedback=[{"name": i.name} for i in ideas])
    assert {i.name for i in again}.isdisjoint({i.name for i in ideas})
    assert "Previously tested factors" in llm.prompts[-1]


def test_implementation_produces_runnable_code():
    llm = MockLLM()
    impl = ImplementationAgent(llm)
    factor = impl.implement_alpha(AlphaIdea(name="short_term_reversal", description="reversal"))
    assert "def compute_alpha" in factor.code
    ns = {"pd": pd, "np": np}
    exec(factor.code, ns)
    df = create_synthetic_market_data(10, 30)
    out = ns["compute_alpha"](df)
    assert out.index.equals(df.index)
    repaired = impl.repair_alpha(factor, "KeyError: 'foo'")
    assert "def compute_alpha" in repaired.code and repaired.notes


def test_evaluator_cannot_override_statistical_gate():
    always_accept = ScriptedLLM(lambda p, s: '```json\n{"verdict": "accept", "reasons": []}\n```\n[Confidence: 0.99]')
    ev = EvaluationAgent(always_accept, ic_threshold=0.02, min_tstat=2.0)
    weak = ev.apply_metrics(AlphaFactor("weak", "code"), {"ic": 0.001, "rank_ic": 0.0, "ic_tstat": 0.3})
    assert ev.review(weak)["verdict"] == "reject"
    strong = ev.apply_metrics(AlphaFactor("strong", "code"), {"ic": 0.05, "rank_ic": 0.04, "ic_tstat": 4.0})
    assert ev.review(strong)["verdict"] == "accept"


def test_metrics_find_planted_reversal_and_not_noise():
    signal = create_synthetic_market_data(40, 200, planted_signal="reversal", seed=1)
    noise = create_synthetic_market_data(40, 200, planted_signal=None, seed=1)

    def reversal(df):
        return -df.groupby(level="ticker")["returns"].transform(lambda s: s.rolling(5).sum())

    m_sig = summarize_factor(reversal(signal), signal["close"], horizon=1)
    m_noise = summarize_factor(reversal(noise), noise["close"], horizon=1)
    assert m_sig["ic"] > 0.05 and m_sig["ic_tstat"] > 3 and m_sig["sharpe"] > 1
    assert abs(m_noise["ic"]) < 0.03


# ---- LLM backends ----------------------------------------------------------
class _FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.message


def _fake_anthropic(stop_reason="end_turn"):
    calls = []
    message = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="hello")],
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
    )

    def stream(**kwargs):
        calls.append(kwargs)
        return _FakeStream(message)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    return client, calls


def test_anthropic_backend_request_shape():
    client, calls = _fake_anthropic()
    llm = AnthropicLLM(LLMConfig(effort="high"), client=client)
    assert llm.generate("hi", system_prompt="sys") == "hello"
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "sys"
    assert kw["output_config"] == {"effort": "high"}
    assert kw["fallbacks"] == "default" and kw["betas"] == [FALLBACK_BETA]
    assert "thinking" not in kw and "temperature" not in kw
    assert llm.total_input_tokens == 10 and llm.calls == 1


def test_anthropic_backend_surfaces_refusal():
    client, _ = _fake_anthropic("refusal")
    with pytest.raises(LLMError):
        AnthropicLLM(LLMConfig(), client=client).generate("x")


def test_vllm_backend():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "meta-llama/Meta-Llama-3-70B-Instruct"
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "local answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    http = httpx.Client(base_url="http://vllm/v1", transport=httpx.MockTransport(handler))
    llm = VLLMClient(LLMConfig(provider="vllm", model_name="meta-llama/Meta-Llama-3-70B-Instruct"), http=http)
    assert llm.generate("q", system_prompt="s") == "local answer"


def test_build_llm_factory():
    assert isinstance(build_llm(LLMConfig(provider="mock")), MockLLM)
    with pytest.raises(ValueError):
        build_llm(LLMConfig(provider="nope"))
