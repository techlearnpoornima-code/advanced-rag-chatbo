"""
LLM Client for Text Generation
Handles communication with Claude API
"""
import anthropic
from typing import Optional
from loguru import logger
from app.config import settings


class LLMClient:
    """Client for Claude LLM"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> str:
        """
        Generate text using Claude
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            system: System prompt
            
        Returns:
            Generated text
        """
        try:
            # Use provided params or defaults
            temp = temperature if temperature is not None else settings.TEMPERATURE
            tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS
            
            # Build request
            kwargs = {
                "model": self.model,
                "max_tokens": tokens,
                "temperature": temp,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system:
                kwargs["system"] = system
            
            # Call API
            response = self.client.messages.create(**kwargs)
            
            # Extract text
            text = response.content[0].text
            
            logger.debug(f"Generated {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Generate text with streaming
        
        Yields:
            Text chunks as they're generated
        """
        try:
            temp = temperature if temperature is not None else settings.TEMPERATURE
            tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS
            
            with self.client.messages.stream(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
