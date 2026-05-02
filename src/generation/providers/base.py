"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """
    Pluggable LLM provider interface.

    Implement this to swap between Ollama, Anthropic, OpenAI, etc.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: User prompt text
            system: Optional system/instruction prefix
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0=deterministic, 1=creative)

        Returns:
            Generated text string
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name, e.g. 'ollama/llama3' or 'anthropic/claude-sonnet'."""
