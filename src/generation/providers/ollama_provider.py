"""Ollama LLM provider — local models via Ollama."""

from typing import Optional

import ollama
from loguru import logger

from src.generation.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Runs generation against a locally running Ollama instance.

    Default model: llama3 (change via constructor or LLM_MODEL env var).
    Ollama must be running: `ollama serve`
    """

    def __init__(self, model: str = "llama3"):
        self.model = model
        self._client = ollama.AsyncClient()

    @property
    def provider_name(self) -> str:
        return f"ollama/{self.model}"

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Calling {self.provider_name} | tokens={max_tokens} temp={temperature}")

        response = await self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return response["message"]["content"]
