"""Model client implementations."""

import httpx
import json
import time
from typing import Dict, Any, Optional, List
import asyncio
import base64
from datetime import datetime, UTC

from app.core.logger import logger
from app.core.config import ModelBackend, get_settings

settings = get_settings()

class ModelClient:
    """Base class for model clients."""
    
    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a text."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def update_metrics(self, successful: bool, tokens: int, latency: float) -> None:
        """Update model metrics."""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIClient(ModelClient):
    """Client for OpenAI API."""
    
    def __init__(self, api_key: str, model_id: str, model_name: str = "gpt-4", api_base: str = None, timeout: int = 60):
        """Initialize OpenAI client."""
        self.api_key = api_key
        self.model_id = model_id
        self.model_name = model_name or "gpt-4"
        self.api_base = api_base or "https://api.openai.com/v1"
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout
        )
    
    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> str:
        """Generate text from a prompt using OpenAI API."""
        if options is None:
            options = {}
        
        start_time = time.time()
        successful = False
        tokens = 0
        
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": options.get("temperature", 0.7),
                    "max_tokens": options.get("max_tokens", 1000),
                    "top_p": options.get("top_p", 1.0),
                    "frequency_penalty": options.get("frequency_penalty", 0.0),
                    "presence_penalty": options.get("presence_penalty", 0.0)
                }
            )
            response.raise_for_status()
            result = response.json()
            
            text = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            successful = True
            
            return text
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
        finally:
            latency = time.time() - start_time
            await self.update_metrics(successful, tokens, latency)
    
    async def embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a text using OpenAI API."""
        try:
            response = await self.client.post(
                "/embeddings",
                json={
                    "model": "text-embedding-ada-002",
                    "input": text
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return result["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            raise
    
    async def update_metrics(self, successful: bool, tokens: int, latency: float) -> None:
        """Update model metrics in MCP database."""
        from mcp.database import get_session, models
        
        try:
            async with get_session() as session:
                model = await models.get_model(session, self.model_id)
                if not model:
                    return
                
                # Get current metrics or initialize
                metrics = model.metrics or {}
                
                # Update metrics
                metrics["total_requests"] = metrics.get("total_requests", 0) + 1
                
                if successful:
                    metrics["successful_requests"] = metrics.get("successful_requests", 0) + 1
                else:
                    metrics["failed_requests"] = metrics.get("failed_requests", 0) + 1
                
                metrics["total_tokens"] = metrics.get("total_tokens", 0) + tokens
                
                # Update average latency
                current_latency = metrics.get("average_latency", 0.0)
                total_requests = metrics.get("total_requests", 1)
                metrics["average_latency"] = ((current_latency * (total_requests - 1)) + latency) / total_requests
                
                # Update last used timestamp
                metrics["last_used"] = datetime.now(UTC).isoformat()
                
                # Save updated metrics
                model.metrics = metrics
                await models.update_model(session, model)
                
        except Exception as e:
            logger.error(f"Error updating model metrics: {str(e)}")


class AnthropicClient(ModelClient):
    """Client for Anthropic API."""
    
    def __init__(self, api_key: str, model_id: str, model_name: str = "claude-2", api_base: str = None, timeout: int = 60):
        """Initialize Anthropic client."""
        self.api_key = api_key
        self.model_id = model_id
        self.model_name = model_name or "claude-2"
        self.api_base = api_base or "https://api.anthropic.com/v1"
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            timeout=self.timeout
        )
    
    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> str:
        """Generate text from a prompt using Anthropic API."""
        if options is None:
            options = {}
        
        start_time = time.time()
        successful = False
        tokens = 0
        
        try:
            response = await self.client.post(
                "/messages",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": options.get("max_tokens", 1000),
                    "temperature": options.get("temperature", 0.7),
                    "top_p": options.get("top_p", 1.0)
                }
            )
            response.raise_for_status()
            result = response.json()
            
            text = result["content"][0]["text"]
            # Anthropic doesn't return token count, estimate it
            tokens = len(prompt.split()) + len(text.split())
            successful = True
            
            return text
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {str(e)}")
            raise
        finally:
            latency = time.time() - start_time
            await self.update_metrics(successful, tokens, latency)
    
    async def update_metrics(self, successful: bool, tokens: int, latency: float) -> None:
        """Update model metrics in MCP database."""
        from mcp.database import get_session, models
        
        try:
            async with get_session() as session:
                model = await models.get_model(session, self.model_id)
                if not model:
                    return
                
                # Get current metrics or initialize
                metrics = model.metrics or {}
                
                # Update metrics
                metrics["total_requests"] = metrics.get("total_requests", 0) + 1
                
                if successful:
                    metrics["successful_requests"] = metrics.get("successful_requests", 0) + 1
                else:
                    metrics["failed_requests"] = metrics.get("failed_requests", 0) + 1
                
                metrics["total_tokens"] = metrics.get("total_tokens", 0) + tokens
                
                # Update average latency
                current_latency = metrics.get("average_latency", 0.0)
                total_requests = metrics.get("total_requests", 1)
                metrics["average_latency"] = ((current_latency * (total_requests - 1)) + latency) / total_requests
                
                # Update last used timestamp
                metrics["last_used"] = datetime.now(UTC).isoformat()
                
                # Save updated metrics
                model.metrics = metrics
                await models.update_model(session, model)
                
        except Exception as e:
            logger.error(f"Error updating model metrics: {str(e)}")


class MockClient(ModelClient):
    """Mock client for testing and development."""
    
    def __init__(self, model_id: str, model_name: str = "mock"):
        """Initialize mock client."""
        self.model_id = model_id
        self.model_name = model_name
    
    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> str:
        """Generate a mock response."""
        # Add a small delay to simulate processing
        await asyncio.sleep(0.5)
        
        # Return a simple mock response
        return f"This is a mock response from {self.model_name} for the prompt: '{prompt[:50]}...'"
    
    async def embeddings(self, text: str) -> List[float]:
        """Generate mock embeddings."""
        # Return a vector of 128 random values
        import random
        return [random.random() for _ in range(128)]
    
    async def update_metrics(self, successful: bool, tokens: int, latency: float) -> None:
        """Update mock metrics."""
        logger.info(f"Mock update metrics: successful={successful}, tokens={tokens}, latency={latency}")


class ModelClientFactory:
    """Factory for creating model clients."""
    
    def create_client_from_mcp_model(self, model):
        """Create a client from an MCP model object."""
        # Extract the API key from the model's configuration
        api_key = model.configuration.get("api_key", "")
        model_name = model.configuration.get("model_name", model.name)
        
        # Create the appropriate client based on backend type
        if model.backend.lower() == ModelBackend.OPENAI.value:
            return OpenAIClient(
                api_key=api_key,
                model_id=model.id,
                model_name=model_name,
                api_base=model.api_base
            )
        elif model.backend.lower() == ModelBackend.ANTHROPIC.value:
            return AnthropicClient(
                api_key=api_key,
                model_id=model.id,
                model_name=model_name,
                api_base=model.api_base
            )
        elif model.backend.lower() == ModelBackend.LOCAL.value:
            # For local models, use the mock client for now
            return MockClient(model_id=model.id, model_name=model_name)
        else:
            # Default to mock client for unsupported backends
            logger.warning(f"Unsupported model backend {model.backend}, using mock client")
            return MockClient(model_id=model.id, model_name=model_name)
    
    def create_client(self, model_id: str, backend: str, api_key: str, model_name: str = None, api_base: str = None):
        """Create a client based on backend type."""
        backend_type = backend.lower()
        
        if backend_type == ModelBackend.OPENAI.value:
            return OpenAIClient(
                api_key=api_key,
                model_id=model_id,
                model_name=model_name,
                api_base=api_base
            )
        elif backend_type == ModelBackend.ANTHROPIC.value:
            return AnthropicClient(
                api_key=api_key,
                model_id=model_id,
                model_name=model_name,
                api_base=api_base
            )
        elif backend_type == ModelBackend.LOCAL.value:
            return MockClient(model_id=model_id, model_name=model_name)
        else:
            logger.warning(f"Unsupported model backend {backend}, using mock client")
            return MockClient(model_id=model_id, model_name=model_name) 