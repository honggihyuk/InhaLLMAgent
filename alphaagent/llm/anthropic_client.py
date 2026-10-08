"""Claude backend via the official Anthropic SDK."""

from __future__ import annotations

from typing import Optional

from alphaagent.llm.base import LLMClient, LLMConfig, LLMError, LLMResponse

# Server-side refusal fallback: on a policy decline the API re-runs the request on a fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM(LLMClient):
    def __init__(self, config: LLMConfig, client=None) -> None:
        super().__init__(config)
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install 'alphaagent[llm]' to use the Claude backend") from e
            # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile.
            kwargs = {"timeout": config.timeout}
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            client = anthropic.Anthropic(**kwargs)
        self.client = client

    def _complete(self, prompt: str, system: Optional[str], max_tokens: int) -> LLMResponse:
        kwargs = dict(
            model=self.config.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.config.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if system:
            kwargs["system"] = system
        # Streaming keeps long generations clear of HTTP timeouts.
        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMError(f"model declined the request ({getattr(details, 'category', None)})")
        text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            stop_reason=message.stop_reason or "",
            raw=message,
        )
