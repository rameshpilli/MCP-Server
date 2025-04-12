from typing import Dict, Any, Optional, List
import httpx
from pydantic import BaseModel
import os
from enum import Enum

class ModelBackend(str, Enum):
    """Supported model backend types"""
    CUSTOM = "custom"
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"

class ModelConfig(BaseModel):
    """Model configuration"""
    model_id: str
    backend: ModelBackend
    api_base: str
    api_version: str = "v1"
    timeout: int = 30
    max_tokens: int = 2000
    temperature: float = 0.7
    additional_params: Dict[str, Any] = {}

class MCPClient:
    """MCP Client for model registration and management"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize MCP client.
        
        Args:
            base_url: MCP server URL (default: from MCP_SERVER_URL env var)
            api_key: API key for authentication (default: from MCP_API_KEY env var)
        """
        self.base_url = base_url or os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("MCP_API_KEY")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
    
    async def register_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Register a new model with MCP.
        
        Args:
            config: Model configuration
            
        Returns:
            Dict containing registration response
            
        Example:
            ```python
            from mcp_client import MCPClient, ModelConfig, ModelBackend
            
            # Initialize client
            client = MCPClient()
            
            # Create model config
            config = ModelConfig(
                model_id="my-custom-model",
                backend=ModelBackend.CUSTOM,
                api_base="http://my-model:8000",
                additional_params={"model_type": "text-generation"}
            )
            
            # Register model
            response = await client.register_model(config)
            ```
        """
        response = await self.client.post(
            "/models/register",
            json=config.model_dump()
        )
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        response = await self.client.get("/models")
        response.raise_for_status()
        return response.json()
    
    async def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """Get usage statistics for a model."""
        response = await self.client.get(f"/models/{model_id}/stats")
        response.raise_for_status()
        return response.json()
    
    async def health_check(self, model_id: str) -> Dict[str, Any]:
        """Check health status of a model."""
        response = await self.client.get(f"/models/{model_id}/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client session."""
        await self.client.aclose() 