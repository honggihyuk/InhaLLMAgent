"""Local open-weight models (e.g. Llama-3-70B-Instruct) served by vLLM's OpenAI-compatible HTTP server.

Start the server with e.g.::

    vllm serve meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 4 --port 8001
"""

from __future__ import annotations

from typing import Optional

import httpx

from alphaagent.llm.base import LLMClient, LLMConfig, LLMError, LLMResponse


class VLLMClient(LLMClient):
    def __init__(self, config: LLMConfig, http: Optional[httpx.Client] = None) -> None:
        super().__init__(config)
        base = (config.base_url or "http://localhost:8001/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.http = http or httpx.Client(base_url=base, headers=headers, timeout=config.timeout)

    def _complete(self, prompt: str, system: Optional[str], max_tokens: int) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = self.http.post(
            "/chat/completions",
            json={
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
            },
        )
        if resp.status_code >= 400:
            raise LLMError(f"vLLM error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        return LLMResponse(
            text=choice["message"]["content"] or "",
            model=data.get("model", self.config.model_name),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            stop_reason=choice.get("finish_reason") or "",
            raw=data,
        )
