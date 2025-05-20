"""
Server-Sent Events (SSE) endpoint for MCP protocol

This is the main entry point for Chainlit's SSE connection. When Chainlit
attempts to connect to our MCP server, this is the endpoint it hits first.

The communication flow works like this:

1. Chainlit connects to /sse via GET request
2. We send a handshake event with server info (name, version)
3. We keep the connection open with regular ping events
4. If the connection drops, Chainlit will attempt to reconnect

While this connection is open, Chainlit will make separate HTTP requests to:
- /tools to get available tools
- /process to send messages for processing
- /call_tool to execute specific tools

The SSE connection is just for the initial handshake and keepalive.
It doesn't carry the actual tool calls or processing requests.

Args:
    request: The FastAPI request object
    process_message_func: Function to process messages (not used in this endpoint)

Returns:
    EventSourceResponse: An SSE response that keeps the connection open
"""
# app/sse_handler.py
import asyncio
import json
import time
import logging
from fastapi import Request
from sse_starlette.sse import EventSourceResponse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger("mcp_server.sse")

async def sse_endpoint(request: Request, process_message_func, get_tools_func):
    """Server-Sent Events (SSE) endpoint for MCP protocol"""

    logger.info(f"SSE endpoint called, starting handshake")

    async def event_generator():
        try:
            # Step 1: Send handshake event as a JSON-RPC notification
            jsonrpc_handshake = {
                "jsonrpc": "2.0",
                "method": "handshake",
                "params": {
                    "protocol_version": "1.0",
                    "server_info": {
                        "name": "MCP Server",
                        "version": "1.0.0"
                    }
                }
            }

            # Important: Chainlit expects the event name to be "message"
            yield {
                "event": "message",
                "data": json.dumps(jsonrpc_handshake)
            }
            logger.info("Sent handshake event")

            # List tools immediately after handshake
            tools = await get_tools_func()
            formatted_tools = []
            for name, tool in tools.items():
                # Get input schema
                input_schema = getattr(tool, 'input_schema', {})

                # Handle possible function call schema format
                if not input_schema and hasattr(tool, 'fn'):
                    # Extract parameter info from function
                    import inspect
                    sig = inspect.signature(tool.fn)
                    input_schema = {
                        "type": "object",
                        "properties": {}
                    }

                    # Skip first parameter (usually ctx/context)
                    params = list(sig.parameters.items())
                    if params:
                        params = params[1:]  # Skip first parameter (ctx)

                    for param_name, param in params:
                        param_info = {
                            "type": "string"  # Default to string
                        }
                        # Get annotation if available
                        if param.annotation != inspect.Parameter.empty:
                            if param.annotation == str:
                                param_info["type"] = "string"
                            elif param.annotation == int:
                                param_info["type"] = "integer"
                            elif param.annotation == float:
                                param_info["type"] = "number"
                            elif param.annotation == bool:
                                param_info["type"] = "boolean"

                        # Get default if available
                        if param.default != inspect.Parameter.empty:
                            param_info["default"] = param.default

                        input_schema["properties"][param_name] = param_info

                formatted_tools.append({
                    "name": name,
                    "description": getattr(tool, 'description', 'No description'),
                    "inputSchema": input_schema
                })

            # Send tools_list as JSON-RPC notification with id field
            # Chainlit might expect an ID even for notifications
            jsonrpc_tools_list = {
                "jsonrpc": "2.0",
                "method": "tools_list",
                "params": {
                    "tools": formatted_tools
                },
                "id": 0  # Add an ID field
            }

            yield {
                "event": "message",
                "data": json.dumps(jsonrpc_tools_list)
            }
            logger.info("Sent tools_list event")

            # Keep connection alive with ping
            message_id = 1
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected")
                    break

                # Chainlit expects periodic ping events as JSON-RPC requests
                await asyncio.sleep(30)

                jsonrpc_ping = {
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": message_id,
                    "params": {
                        "timestamp": time.time()
                    }
                }

                yield {
                    "event": "message",
                    "data": json.dumps(jsonrpc_ping)
                }
                logger.info(f"Sent ping event #{message_id}")
                message_id += 1

        except Exception as e:
            logger.error(f"SSE Error: {str(e)}")
            jsonrpc_error = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e)
                },
                "id": None
            }
            yield {
                "event": "message",
                "data": json.dumps(jsonrpc_error)
            }
    return EventSourceResponse(event_generator(), media_type="text/event-stream")

async def sse_list_tools(get_tools_func):
    """List available tools in Chainlit-compatible format"""
    try:
        tools = await get_tools_func()

        # Format tools in the way Chainlit expects them
        formatted_tools = []
        for name, tool in tools.items():
            # Get input schema
            input_schema = getattr(tool, 'input_schema', {})

            # Handle possible function call schema format
            if not input_schema and hasattr(tool, 'fn'):
                # Extract parameter info from function
                import inspect

                sig = inspect.signature(tool.fn)
                input_schema = {
                    "type": "object",
                    "properties": {}
                }

                # Skip first parameter (usually ctx/context)
                params = list(sig.parameters.items())
                if params:
                    params = params[1:]  # Skip first parameter (ctx)

                for param_name, param in params:
                    param_info = {
                        "type": "string"  # Default to string
                    }
                    # Get annotation if available
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == str:
                            param_info["type"] = "string"
                        elif param.annotation == int:
                            param_info["type"] = "integer"
                        elif param.annotation == float:
                            param_info["type"] = "number"
                        elif param.annotation == bool:
                            param_info["type"] = "boolean"

                    # Get default if available
                    if param.default != inspect.Parameter.empty:
                        param_info["default"] = param.default

                    input_schema["properties"][param_name] = param_info

            formatted_tools.append({
                "name": name,
                "description": getattr(tool, 'description', 'No description'),
                "inputSchema": input_schema
            })

        # Return Chainlit-compatible format
        return {
            "status": "success",  # Add status field
            "tools": formatted_tools
        }
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        return {
            "status": "error",  # Add status field
            "error": str(e)
        }

async def sse_process(request: Request, process_message_func):
    """Process MCP requests from SSE clients"""
    try:
        # Parse the request body
        data = await request.json()

        # Extract message and context
        message = data.get("message", "")
        context = data.get("context", {})

        logger.info(f"Processing message: {message[:50]}...")

        # Process the message
        result = await process_message_func(message, context)

        # Return properly formatted response
        return {
            "status": "success",
            "response": result
        }
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def sse_call_tool(request: Request, get_tools_func):
    """Execute a specific tool with parameters"""
    try:
        # Parse the request body
        data = await request.json()

        # Extract tool name and parameters
        tool_name = data.get("tool", "")
        parameters = data.get("parameters", {})

        if not tool_name:
            return {
                "status": "error",
                "error": "Missing tool name"
            }

        logger.info(f"Calling tool: {tool_name} with parameters: {parameters}")

        # Get available tools
        tools = await get_tools_func()

        if tool_name not in tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not found"
            }

        # Import Context to avoid circular imports
        from fastmcp import Context

        # Create a context for the tool
        ctx = Context({})

        # Execute the tool
        result = await tools[tool_name].fn(ctx, **parameters)

        # Return properly formatted response
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
