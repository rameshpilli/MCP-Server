import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mcp_client import MCPClient


def test_base_url_env(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    client = MCPClient()
    assert client.base_url == "http://localhost:8000"

    monkeypatch.setenv("MCP_SERVER_URL", "http://example.com")
    client2 = MCPClient()
    assert client2.base_url == "http://example.com"

