"""Base class for LLM backend implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional
from app.core.models import ModelRecord

class LLMBackend(ABC):
    """Abstract base class for LLM backend implementations."""

    def __init__(self, model: ModelRecord):
        """Initialize the LLM backend with model configuration."""
        self.model = model
        self.config = model.config or {}

    @abstractmethod
    async def generate(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate a single response from the model."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, parameters: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream responses from the model."""
        pass

    def prepare_prompt(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Prepare the prompt with system message and examples if provided."""
        final_prompt = ""
        
        # Add system prompt if provided
        if system_prompt := parameters.get("system_prompt"):
            final_prompt += f"System: {system_prompt}\n\n"
            
        # Add few-shot examples if provided
        if examples := parameters.get("examples"):
            for example in examples:
                if isinstance(example, dict):
                    final_prompt += f"User: {example.get('input', '')}\n"
                    final_prompt += f"Assistant: {example.get('output', '')}\n\n"
                else:
                    final_prompt += f"{example}\n\n"
                    
        # Add the actual prompt
        final_prompt += f"User: {prompt}\nAssistant: "
        
        return final_prompt

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the model."""
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None
        }

    def merge_parameters(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user parameters with defaults."""
        params = self.get_default_parameters()
        params.update(user_params)
        return params 