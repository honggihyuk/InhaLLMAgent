"""Offline LLM backends for tests, demos, and CI (no API key, no network).

``ScriptedLLM`` replays canned responses. ``MockLLM`` routes on the ``TASK_KIND:`` marker that every
agent prompt carries and answers with handler functions; :mod:`alphaagent.llm.mock_handlers` supplies
handlers that emit well-formed ideas, alpha code, and reviews so the whole pipeline can run offline.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Union

from alphaagent.llm.base import LLMClient, LLMConfig, LLMResponse

Handler = Callable[[str, Optional[str]], str]
_KIND = re.compile(r"TASK_KIND:\s*([a-z_]+)")


def task_kind(prompt: str) -> str:
    m = _KIND.search(prompt)
    return m.group(1) if m else "generic"


class ScriptedLLM(LLMClient):
    def __init__(self, responses: Union[Iterable[str], Handler], config: Optional[LLMConfig] = None) -> None:
        super().__init__(config or LLMConfig(provider="mock", model_name="scripted"))
        self._fn = responses if callable(responses) else None
        self._queue: List[str] = [] if callable(responses) else list(responses)
        self.prompts: List[str] = []

    def _complete(self, prompt, system, max_tokens):
        self.prompts.append(prompt)
        if self._fn is not None:
            text = self._fn(prompt, system)
        else:
            if not self._queue:
                raise RuntimeError("ScriptedLLM ran out of responses")
            text = self._queue.pop(0)
        return LLMResponse(text=text, model="scripted", input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)


class MockLLM(LLMClient):
    def __init__(self, handlers: Optional[Dict[str, Handler]] = None, config: Optional[LLMConfig] = None, seed: int = 0):
        super().__init__(config or LLMConfig(provider="mock", model_name="mock"))
        if handlers is None:
            from alphaagent.llm.mock_handlers import default_handlers

            handlers = default_handlers(seed)
        self.handlers = handlers
        self.prompts: List[str] = []

    def _complete(self, prompt, system, max_tokens):
        self.prompts.append(prompt)
        kind = task_kind(prompt)
        handler = self.handlers.get(kind) or self.handlers.get("generic")
        text = handler(prompt, system) if handler else "No handler.\n[Confidence: 0.50]"
        return LLMResponse(text=text, model="mock", input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)
