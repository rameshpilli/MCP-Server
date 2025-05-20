"""Server-Sent Events (SSE) endpoint for MCP protocol.

This module exposes utility functions to handle Chainlit's SSE communication. It
performs a handshake, lists available tools and keeps the connection alive with
periodic pings. JSON parsing errors result in HTTP 400 responses.
Clients should reconnect if the connection drops.
"""

import asyncio
import json
import time
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("mcp_server.sse")


async def sse_endpoint(request: Request, process_message_func, get_tools_func):
    """Main SSE endpoint used by Chainlit."""

    async def event_generator():
        try:
            handshake = {
                "jsonrpc": "2.0",
                "method": "handshake",
                "params": {"protocol_version": "1.0", "server_info": {"name": "MCP Server", "version": "1.0.0"}},
            }
            yield {"event": "message", "data": json.dumps(handshake)}
            tools = await get_tools_func()
            formatted = []
            for name, tool in tools.items():
                schema = getattr(tool, "input_schema", {})
                formatted.append({"name": name, "description": getattr(tool, "description", ""), "inputSchema": schema})
            yield {"event": "message", "data": json.dumps({"jsonrpc": "2.0", "method": "tools_list", "params": {"tools": formatted}, "id": 0})}
            msg_id = 1
            while True:
                if await request.is_disconnected():
                    logger.info("SSE client disconnected")
                    break
                await asyncio.sleep(30)
                ping = {"jsonrpc": "2.0", "method": "ping", "id": msg_id, "params": {"timestamp": time.time()}}
                try:
                    yield {"event": "message", "data": json.dumps(ping)}
                except Exception as e:
                    logger.warning(f"Ping failed: {e}")
                    break
                msg_id += 1
        except Exception as e:
            logger.error(f"SSE Error: {e}")
            error = {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": None}
            yield {"event": "message", "data": json.dumps(error)}

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


async def sse_list_tools(get_tools_func):
    """Return available tools in Chainlit-compatible format."""
    try:
        tools = await get_tools_func()
        formatted = []
        for name, tool in tools.items():
            schema = getattr(tool, "input_schema", {})
            formatted.append({"name": name, "description": getattr(tool, "description", ""), "inputSchema": schema})
        return {"status": "success", "tools": formatted}
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return {"status": "error", "error": str(e)}


async def sse_process(request: Request, process_message_func):
    """Process a message from an SSE client."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"status": "error", "error": "Invalid JSON"})

    message = data.get("message", "")
    context = data.get("context", {})
    try:
        result = await process_message_func(message, context)
        return {"status": "success", "response": result}
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {"status": "error", "error": str(e)}


async def sse_call_tool(request: Request, get_tools_func):
    """Execute a specific tool with parameters."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"status": "error", "error": "Invalid JSON"})

    tool_name = data.get("tool", "")
    parameters = data.get("parameters", {})
    if not tool_name:
        return JSONResponse(status_code=400, content={"status": "error", "error": "Missing tool name"})

    tools = await get_tools_func()
    if tool_name not in tools:
        return JSONResponse(status_code=404, content={"status": "error", "error": f"Tool '{tool_name}' not found"})

    from fastmcp import Context
    ctx = Context({})
    try:
        result = await tools[tool_name].fn(ctx, **parameters)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return {"status": "error", "error": str(e)}
