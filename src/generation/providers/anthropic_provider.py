"""Anthropic Claude LLM provider."""

from typing import Optional

import anthropic
from loguru import logger

from src.generation.providers.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    Calls Anthropic's Claude API.

    Requires ANTHROPIC_API_KEY in environment.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return f"anthropic/{self.model}"

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        logger.debug(f"Calling {self.provider_name} | tokens={max_tokens} temp={temperature}")

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text
