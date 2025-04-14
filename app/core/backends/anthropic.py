"""Anthropic backend implementation."""

import asyncio
from typing import Dict, Any, AsyncGenerator, Optional
import anthropic
from app.core.backends.base import LLMBackend
from app.core.models import ModelRecord
from app.core.logging import logger

class AnthropicBackend(LLMBackend):
    """Anthropic API backend implementation."""
    
    def __init__(self, model: ModelRecord):
        """Initialize the Anthropic backend."""
        super().__init__(model)
        self.client = anthropic.AsyncAnthropic(
            api_key=self.config.get("api_key")
        )
        
    async def generate(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate a single response from Anthropic."""
        try:
            merged_params = self.merge_parameters(parameters)
            final_prompt = self.prepare_prompt(prompt, parameters)
            
            response = await self.client.messages.create(
                model=self.model.model_id,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=merged_params["temperature"],
                max_tokens=merged_params["max_tokens"],
                top_p=merged_params["top_p"],
                stop_sequences=merged_params["stop"] if merged_params["stop"] else None,
                stream=False
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response from Anthropic: {str(e)}")
            raise
            
    async def generate_stream(self, prompt: str, parameters: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream responses from Anthropic."""
        try:
            merged_params = self.merge_parameters(parameters)
            final_prompt = self.prepare_prompt(prompt, parameters)
            
            stream = await self.client.messages.create(
                model=self.model.model_id,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=merged_params["temperature"],
                max_tokens=merged_params["max_tokens"],
                top_p=merged_params["top_p"],
                stop_sequences=merged_params["stop"] if merged_params["stop"] else None,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.type == "content_block" and chunk.delta.text:
                    yield chunk.delta.text
                    
        except Exception as e:
            logger.error(f"Error streaming response from Anthropic: {str(e)}")
            raise
            
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Anthropic models."""
        params = super().get_default_parameters()
        # Anthropic doesn't support frequency/presence penalties
        params.pop("frequency_penalty", None)
        params.pop("presence_penalty", None)
        return params 