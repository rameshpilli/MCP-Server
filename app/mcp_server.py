# app/mcp_server.py
import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from fastmcp import FastMCP, Context
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import Dict, Any
from dotenv import load_dotenv

# Set up paths
sys.path.append(str(Path(__file__).parent.parent))
from app.config import config
from app.stdio_handler import run_stdio_mode
from app.sse_handler import sse_endpoint, sse_process, sse_list_tools, sse_call_tool

# Environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("mcp_server")
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# MCP server
mcp = FastMCP(
    "MCP Server",
    description="MCP Server for tool execution"
)


# Register a basic tool
@mcp.tool()
async def health_check(ctx: Context) -> str:
    """Health check endpoint"""
    return "MCP Server is healthy"


@mcp.tool()
async def server_info(ctx: Context) -> str:
    """Get server information"""
    return f"""
# MCP Server Information
**Host**: {config.MCP_SERVER_HOST}
**Port**: {config.MCP_SERVER_PORT}
"""

# Auto-discover tools in app.tools package
try:
    from app.registry.tools import autodiscover_tools
    autodiscover_tools(mcp)
    logger.info("Auto-discovered tool modules")
except Exception as e:
    logger.error(f"Tool auto-discovery failed: {e}")

# Add debugging tool
@mcp.tool()
async def list_tools(ctx: Context) -> str:
    """List all available tools and their schemas"""
    try:
        tools = await mcp.get_tools()
        result = "# Available Tools\n\n"

        for name, tool in tools.items():
            result += f"## {name}\n"
            result += f"Description: {getattr(tool, 'description', 'No description')}\n\n"

            # Try to get schema
            schema = None
            for attr in ['input_schema', 'schema', 'parameters']:
                if hasattr(tool, attr):
                    schema = getattr(tool, attr)
                    break

            if schema:
                result += f"Parameters:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            else:
                result += "No parameter schema available.\n\n"

        return result
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return f"Error listing tools: {str(e)}"

# Setup the bridge
try:
    from app.mcp_bridge import MCPBridge

    # Create a bridge instance
    bridge = MCPBridge(mcp)
    logger.info("MCP-Bridge initialized Successfully")
except Exception as e:
    logger.error(f"MCP-Bridge initialization failed: {e}")
    bridge = None

# Process message using the bridge
async def process_message(message: str, context: Dict[str, Any] = None) -> str:
    """Process a message using the MCPBridge"""
    try:
        logger.info(f"Processing message: {message}")

        # Route the message using the bridge
        routing_result = await bridge.route_request(message, context or {})
        logger.info(f"Routing result: {routing_result}")

        # Execute tools from endpoints
        results = []
        for endpoint in routing_result["endpoints"]:
            if endpoint["type"] == "tool":
                tool_name = endpoint["name"]
                params = endpoint.get("params", {})

                logger.info(f"Executing tool: {tool_name} with params: {params}")

                # Get tools
                tools = await mcp.get_tools()
                if tool_name in tools:
                    # Create context
                    ctx = Context(context or {})

                    try:
                        # Execute tool with parameters
                        if params:
                            # Tools expect named parameters
                            result = await tools[tool_name].fn(ctx, **params)
                        else:
                            # No parameters
                            result = await tools[tool_name].fn(ctx)

                        results.append(result)
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        results.append(error_msg)
                else:
                    logger.warning(f"Tool not found: {tool_name}")

        # Format response based on results
        if results:
            if len(results) == 1:
                # Just return the single result directly
                return results[0]
            else:
                # Format multiple results
                response_parts = ["Multiple tool results:"]
                for i, result in enumerate(results, 1):
                    response_parts.append(f"{i}. {result}")
                return "\n\n".join(response_parts)
        else:
            return "I couldn't find any tools to execute for your request."

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return f"Error processing message: {str(e)}"


def main_factory():
    """Create and return the FastAPI application"""
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # Keep your existing endpoints
    @app.get("/ping")
    async def ping():
        return {"status": "ok", "server": "MCP Server"}


    @app.post("/mcp")
    async def handle_mcp_request(request: Request):
        try:
            data = await request.json()
            if "message" not in data:
                return {"error": "Missing 'message' field"}

            message = data["message"]
            context = data.get("context", {})

            result = await process_message(message, context)
            return {"response": result}
        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return {"error": f"Error processing request: {str(e)}"}


    # Add SSE endpoint (this is what Chainlit connects to)
    @app.get("/sse")
    async def handle_sse(request: Request):
        return await sse_endpoint(request, process_message, get_tools_func=mcp.get_tools)


    # Add endpoints for tool listing and execution
    @app.get("/tools")
    async def handle_list_tools():
        try:
            tools = await mcp.get_tools()

            # Format tools in the way Chainlit expects them
            formatted_tools = []
            for name, tool in tools.items():
                # Get input schema
                input_schema = getattr(tool, 'input_schema', {})

                # Handle possible function call schema format
                if not input_schema and hasattr(tool, 'fn'):
                    # Try to extract parameter info from function
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

            # Return JSON-RPC response
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": formatted_tools
                },
                "id": None  # Chainlit will send an ID, but we're handling GET requests here
            }
        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e)
                },
                "id": None
            }

    # Add process endpoint for message handling
    @app.post("/process")
    async def handle_process_message(request: Request):
        try:
            # Parse the request body
            data = await request.json()

            # Extract JSON-RPC fields
            if "jsonrpc" not in data or data["jsonrpc"] != "2.0" or "method" not in data:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request"
                    },
                    "id": data.get("id")
                }

            # Check if the method is 'process'
            if data["method"] != "process":
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method '{data['method']}' not found"
                    },
                    "id": data.get("id")
                }

            # Extract params
            params = data.get("params", {})
            message = params.get("message", "")
            context = params.get("context", {})

            logger.info(f"Processing message: {message[:50]}...")

            # Process the message
            result = await process_message(message, context)

            # Return JSON-RPC response
            return {
                "jsonrpc": "2.0",
                "result": {
                    "response": result
                },
                "id": data.get("id")
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e)
                },
                "id": None
            }

    # Add call_tool endpoint for tool execution
    @app.post("/call_tool")
    async def handle_call_tool(request: Request):
        try:
            # Parse the request body
            data = await request.json()

            # Extract JSON-RPC fields
            if "jsonrpc" not in data or data["jsonrpc"] != "2.0" or "method" not in data:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request"
                    },
                    "id": data.get("id")
                }

            # Check if the method is 'call_tool'
            if data["method"] != "call_tool":
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method '{data['method']}' not found"
                    },
                    "id": data.get("id")
                }

            # Extract params
            params = data.get("params", {})
            tool_name = params.get("tool", "")
            parameters = params.get("parameters", {})

            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: Missing tool name"
                    },
                    "id": data.get("id")
                }

            logger.info(f"Calling tool: {tool_name} with parameters: {parameters}")

            # Get available tools
            tools = await mcp.get_tools()

            if tool_name not in tools:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32602,
                        "message": f"Tool '{tool_name}' not found"
                    },
                    "id": data.get("id")
                }

            # Import Context to avoid circular imports
            from fastmcp import Context

            # Create a context for the tool
            ctx = Context({})

            # Execute the tool
            result = await tools[tool_name].fn(ctx, **parameters)

            # Return JSON-RPC response
            return {
                "jsonrpc": "2.0",
                "result": {
                    "result": result
                },
                "id": data.get("id")
            }
        except Exception as e:
            logger.error(f"Error calling tool: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e)
                },
                "id": data.get("id", None)
            }

    # Test SSE endpoint
    @app.get("/test-sse")
    async def test_sse(request: Request):
        """Simple SSE test endpoint"""

        async def event_generator():
            yield {"event": "message", "data": "Connected to MCP Server"}
            await asyncio.sleep(1)
            yield {"event": "message", "data": "Test message 1"}
            await asyncio.sleep(1)
            yield {"event": "message", "data": "Test message 2"}

        return EventSourceResponse(event_generator())


    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Server")
    parser.add_argument(
        "--server_type", "--mode",
        dest="mode",
        choices=["http", "stdio"],
        default="http",
        help="Server mode: 'http' for HTTP/SSE or 'stdio' for STDIO transport",
    )
    args = parser.parse_args()

    if args.mode == "stdio":
        # Run in STDIO mode
        asyncio.run(run_stdio_mode(mcp, process_message))
    else:
        # Run in HTTP/SSE mode
        import uvicorn
        uvicorn.run(main_factory(), host="0.0.0.0", port=8080)
