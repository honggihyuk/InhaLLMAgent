from __future__ import annotations

from typing import Optional

from alphaagent.llm.base import LLMClient, LLMConfig, LLMError, LLMResponse
from alphaagent.llm.mock import MockLLM, ScriptedLLM


def build_llm(config: Optional[LLMConfig] = None) -> LLMClient:
    config = config or LLMConfig()
    provider = config.provider.lower()
    if provider in {"anthropic", "claude"}:
        from alphaagent.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(config)
    if provider in {"vllm", "openai_compatible", "local"}:
        from alphaagent.llm.vllm_client import VLLMClient

        return VLLMClient(config)
    if provider == "mock":
        return MockLLM(config=config)
    raise ValueError(f"unknown LLM provider {config.provider!r}")


__all__ = ["LLMClient", "LLMConfig", "LLMError", "LLMResponse", "MockLLM", "ScriptedLLM", "build_llm"]
