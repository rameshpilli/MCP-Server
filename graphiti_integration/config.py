import os

# Base URL of the local Graphiti server
GRAPHITI_BASE_URL = os.getenv("GRAPHITI_BASE_URL", "http://localhost:9100")
# Collection ID to store MCP client data
GRAPHITI_COLLECTION_ID = os.getenv("GRAPHITI_COLLECTION_ID", "mcp_clients")
