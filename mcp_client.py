import os
import httpx
import asyncio
from typing import Optional, Dict, Any


class MCPClient:
    """Simple client for interacting with the MCP server over HTTP."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("MCP_SERVER_URL", "http://localhost:8000")

    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Send a query to the MCP server and return the response."""
        payload = {"message": message}
        if context:
            payload["context"] = context
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/api/v1/chat", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", resp.json())

    def query_sync(self, message: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Synchronous wrapper around :meth:`query`."""
        return asyncio.run(self.query(message, context))

