"""Ingest top client data from MCP tools into Graphiti."""
import asyncio
from typing import List, Dict

from fastmcp import Context
from app.mcp_server import mcp

from .client_graphiti import GraphitiClient
from .utils import markdown_table_to_dicts


def _build_nodes(records: List[Dict[str, str]]) -> List[Dict[str, Dict]]:
    nodes = []
    for rec in records:
        nodes.append({
            "text": str(rec),
            "metadata": rec,
        })
    return nodes


async def fetch_top_clients() -> List[Dict[str, str]]:
    tools = await mcp.get_tools()
    tool = tools.get("get_top_clients")
    if not tool:
        raise RuntimeError("get_top_clients tool not found")

    ctx = Context({})
    result = await tool.fn(ctx)
    return markdown_table_to_dicts(result)


async def main() -> None:
    records = await fetch_top_clients()
    if not records:
        print("No client data retrieved")
        return

    client = GraphitiClient()
    nodes = _build_nodes(records)
    resp = client.ingest_nodes(nodes)
    print("Graphiti response:", resp)


if __name__ == "__main__":
    asyncio.run(main())
