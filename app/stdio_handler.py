# app/stdio_handler.py
import sys
import json
import asyncio
import logging
from fastmcp import Context

logger = logging.getLogger("mcp_server.stdio")

async def run_stdio_mode(mcp, process_message_func):
    """
    Run MCP server in STDIO mode for Chainlit integration

    Args:
        mcp: The FastMCP instance
        process_message_func: Function to process messages
    """
    logger.info("Starting MCP server in STDIO mode")

    def safe_json_dumps(obj):
        """Safely convert an object to JSON, handling non-serializable types"""
        try:
            return json.dumps(obj)
        except TypeError:
            # Try to make the object serializable by converting problematic parts to strings
            if isinstance(obj, dict):
                return json.dumps({k: str(v) if not isinstance(v, (dict, list)) else v for k, v in obj.items()})
            elif isinstance(obj, list):
                return json.dumps([str(item) if not isinstance(item, (dict, list)) else item for item in obj])
            else:
                return json.dumps(str(obj))

    # Reading from standard input (for Chainlit STDIO transport)
    while True:
        try:
            # Read a line from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break  # EOF

            logger.debug(f"Received input: {line.strip()}")

            # Parse JSON request
            request = json.loads(line.strip())

            # Determine the request type and handle accordingly
            if "type" in request:
                if request["type"] == "ping":
                    # Handle ping request
                    response = {"type": "pong"}
                elif request["type"] == "list_tools":
                    # List available tools
                    tools = await mcp.get_tools()
                    formatted_tools = []
                    for name, tool in tools.items():
                        formatted_tools.append({
                            "name": name,
                            "description": getattr(tool, 'description', 'No description'),
                            "inputSchema": getattr(tool, 'input_schema', {})
                        })
                    response = {"type": "tools_list", "tools": formatted_tools}
                elif request["type"] == "call_tool":
                    # Call a specific tool
                    tool_name = request.get("tool", "")
                    params = request.get("parameters", {})

                    tools = await mcp.get_tools()
                    if tool_name in tools:
                        try:
                            # Create a context for the tool
                            ctx = Context({})
                            # Execute the tool
                            result = await tools[tool_name].fn(ctx, **params)
                            response = {"type": "tool_result", "result": result}
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            response = {"type": "error", "error": str(e)}
                    else:
                        logger.warning(f"Tool '{tool_name}' not found")
                        response = {"type": "error", "error": f"Tool '{tool_name}' not found"}
                elif request["type"] == "process":
                    # Process a message
                    message = request.get("message", "")
                    context = request.get("context", {})

                    logger.info(f"Processing message: {message}")
                    result = await process_message_func(message, context)
                    response = {"type": "response", "response": result}
                else:
                    logger.warning(f"Unknown request type: {request['type']}")
                    response = {"type": "error", "error": f"Unknown request type: {request['type']}"}
            else:
                logger.warning("Missing request type")
                response = {"type": "error", "error": "Missing request type"}

            # Write response to stdout
            response_json = safe_json_dumps(response)
            logger.debug(f"Sending response: {response_json[:100]}...")
            print(response_json, flush=True)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            print(json.dumps({"type": "error", "error": f"Invalid JSON: {str(e)}"}), flush=True)
        except Exception as e:
            logger.error(f"Error in STDIO handler: {e}")
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
