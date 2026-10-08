"""Provider-agnostic LLM interface. Agents only ever talk to :class:`LLMClient`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    # "anthropic" (default), "vllm" (any OpenAI-compatible local server, e.g. Llama-3-70B via vLLM), "mock"
    provider: str = "anthropic"
    model_name: str = "claude-opus-5-5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 16000
    # Claude: reasoning effort (low|medium|high|xhigh|max). vLLM: ignored.
    effort: str = "high"
    temperature: float = 0.1  # used by vLLM only; current Claude models fix sampling server-side
    timeout: float = 600.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    raw: Any = None


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls = 0

    @abstractmethod
    def _complete(self, prompt: str, system: Optional[str], max_tokens: int) -> LLMResponse:
        ...

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: Optional[int] = None) -> LLMResponse:
        resp = self._complete(prompt, system, max_tokens or self.config.max_tokens)
        self.calls += 1
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        return resp

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        return self.complete(prompt, system_prompt, max_tokens).text
