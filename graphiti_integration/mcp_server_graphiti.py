"""Minimal MCP server that uses Graphiti as the backend."""
import asyncio
import json
from fastapi import FastAPI, Request
from fastmcp import FastMCP, Context

from .client_graphiti import GraphitiClient

mcp = FastMCP("GraphitiMCP", description="MCP Server backed by Graphiti")
app = FastAPI()
client = GraphitiClient()


async def process_query(message: str) -> str:
    """Send the query to Graphiti and return the answer."""
    result = await asyncio.to_thread(client.query, message)
    if isinstance(result, dict):
        return result.get("answer") or json.dumps(result)
    return str(result)


@app.post("/mcp")
async def handle_mcp(request: Request):
    data = await request.json()
    prompt = data.get("message") or data.get("prompt")
    if not prompt:
        return {"error": "Missing 'message' field"}
    answer = await process_query(prompt)
    return {"response": answer}
