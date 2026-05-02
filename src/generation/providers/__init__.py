"""LLM provider factory — swap providers via LLM_PROVIDER env var."""

import os
from typing import Optional

from src.generation.providers.base import BaseLLMProvider


def get_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Return the configured LLM provider.

    Resolution order:
      1. `provider` argument
      2. LLM_PROVIDER env var  (ollama | anthropic | openai)
      3. Default: ollama

    Args:
        provider: Override provider name
        model: Override model name

    Returns:
        Instantiated provider ready for async generate() calls
    """
    name = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()

    if name == "ollama":
        from src.generation.providers.ollama_provider import OllamaProvider
        return OllamaProvider(model=model or os.getenv("LLM_MODEL", "llama3"))

    if name == "anthropic":
        from src.generation.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    if name == "openai":
        from src.generation.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{name}'. Choose: ollama | anthropic | openai"
    )
