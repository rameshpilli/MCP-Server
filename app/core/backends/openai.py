"""OpenAI backend implementation."""

import asyncio
from typing import Dict, Any, AsyncGenerator, Optional
import openai
from openai import AsyncOpenAI
from app.core.backends.base import LLMBackend
from app.core.models import ModelRecord
from app.core.logging import logger

class OpenAIBackend(LLMBackend):
    """OpenAI API backend implementation."""
    
    def __init__(self, model: ModelRecord):
        """Initialize the OpenAI backend."""
        super().__init__(model)
        self.client = AsyncOpenAI(
            api_key=self.config.get("api_key"),
            organization=self.config.get("organization")
        )
        
    async def generate(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate a single response from OpenAI."""
        try:
            merged_params = self.merge_parameters(parameters)
            final_prompt = self.prepare_prompt(prompt, parameters)
            
            response = await self.client.chat.completions.create(
                model=self.model.model_id,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=merged_params["temperature"],
                max_tokens=merged_params["max_tokens"],
                top_p=merged_params["top_p"],
                frequency_penalty=merged_params["frequency_penalty"],
                presence_penalty=merged_params["presence_penalty"],
                stop=merged_params["stop"],
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {str(e)}")
            raise
            
    async def generate_stream(self, prompt: str, parameters: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream responses from OpenAI."""
        try:
            merged_params = self.merge_parameters(parameters)
            final_prompt = self.prepare_prompt(prompt, parameters)
            
            stream = await self.client.chat.completions.create(
                model=self.model.model_id,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=merged_params["temperature"],
                max_tokens=merged_params["max_tokens"],
                top_p=merged_params["top_p"],
                frequency_penalty=merged_params["frequency_penalty"],
                presence_penalty=merged_params["presence_penalty"],
                stop=merged_params["stop"],
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming response from OpenAI: {str(e)}")
            raise
            
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for OpenAI models."""
        params = super().get_default_parameters()
        # Add any OpenAI-specific defaults here
        return params 