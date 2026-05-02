"""OpenAI LLM provider."""

from typing import Optional

from loguru import logger

from src.generation.providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    Calls OpenAI's Chat Completion API.

    Requires OPENAI_API_KEY in environment.
    Install: pip install openai
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    @property
    def provider_name(self) -> str:
        return f"openai/{self.model}"

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        client = self._get_client()
        logger.debug(f"Calling {self.provider_name} | tokens={max_tokens} temp={temperature}")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
