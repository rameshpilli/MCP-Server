"""Simple Graphiti HTTP client used by MCP."""
from typing import List, Dict, Any
import requests

from .config import GRAPHITI_BASE_URL, GRAPHITI_COLLECTION_ID


class GraphitiClient:
    """Lightweight wrapper around the Graphiti REST API."""

    def __init__(self, base_url: str = GRAPHITI_BASE_URL, collection_id: str = GRAPHITI_COLLECTION_ID):
        self.base_url = base_url.rstrip("/")
        self.collection_id = collection_id

    def ingest_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest nodes into the Graphiti collection."""
        url = f"{self.base_url}/api/v1/collections/{self.collection_id}/nodes"
        response = requests.post(url, json={"nodes": nodes}, timeout=10)
        response.raise_for_status()
        return response.json()

    def query(self, query: str) -> Dict[str, Any]:
        """Send a free-form query to Graphiti."""
        url = f"{self.base_url}/api/v1/query"
        payload = {"query": query, "collection_id": self.collection_id}
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
