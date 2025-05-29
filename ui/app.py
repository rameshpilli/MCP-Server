"""
Chainlit UI App for MCP Server Integration

This module provides a web-based UI for interacting with the MCP server using OpenAI API.

1. Architecture Overview:
    - Chainlit web UI frontend
    - Integration with MCP server via HTTP endpoints
    - OpenAI-compatible API handling
    - Memory management with Redis (optional)

2. Authentication Flow:
    - OAuth token acquisition and caching
    - Fallback to API key authentication
    - Proper token expiry handling

3. Message Processing Flow:
    - Receive user message via Chainlit
    - Process through OpenAI-compatible API with tool definitions
    - Handle response in non-streaming mode:
        - Text responses rendered directly
        - Tool calls processed sequentially
        - Follow-up responses after tool execution
    - Format and display processing time statistics

4. Tool Handling:
    - Dynamically collect tools from connected MCP servers via HTTP
    - Format tools into OpenAI function calling format
    - Execute tool calls in separate steps with proper error handling
    - Display tool results with appropriate formatting

5. Memory Management:
    - Optional Redis-backed conversation memory
    - Fallback to in-memory storage when Redis unavailable
    - Persistent session tracking

6. Error Handling:
    - Comprehensive error handling for API calls
    - User-friendly error messages
    - Detailed logging
    - Troubleshooting suggestions for common issues

Key Components:
    - Authentication (OAuth/API key)
    - LLM API Interface
    - MCP Tool Registration and Execution
    - Response Processing and Formatting
    - Session and Memory Management
    - Performance Monitoring

Usage:
    - Run with: chainlit run ui/app.py
    - Connect to MCP servers via UI
    - Interact with AI assistant with access to all connected tools
"""
import chainlit as cl
from typing import Dict, Any, List, Optional, Union
import os
import sys
import httpx
import asyncio
from pathlib import Path
import time
import json
import logging
import pandas as pd
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from app modules
try:
    from app.config import config
    from app.utils.logger import logger, log_interaction, log_error
except ImportError:
    # Fallback to basic logging if app modules not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("chainlit_ui")
    
    def log_interaction(step, message, session_id=None, **kwargs):
        logger.info(f"{step}: {message} (session: {session_id})")
        
    def log_error(step, error, session_id=None):
        logger.error(f"{step} error: {error} (session: {session_id})")
    
    # Basic config
    class Config:
        LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
        LLM_OAUTH_ENDPOINT = os.getenv("LLM_OAUTH_ENDPOINT", "")
        LLM_OAUTH_CLIENT_ID = os.getenv("LLM_OAUTH_CLIENT_ID", "")
        LLM_OAUTH_CLIENT_SECRET = os.getenv("LLM_OAUTH_CLIENT_SECRET", "")
        LLM_OAUTH_GRANT_TYPE = os.getenv("LLM_OAUTH_GRANT_TYPE", "client_credentials")
        LLM_OAUTH_SCOPE = os.getenv("LLM_OAUTH_SCOPE", "read")
        MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
        
    config = Config()

# Try to import memory module, but make it optional
try:
    from app.memory import ShortTermMemory

    # Test Redis connection
    test_memory = ShortTermMemory()
    if test_memory.redis_client is None:
        logger.warning("Redis connection not available, memory will use in-memory storage")
        MEMORY_AVAILABLE = False
    else:
        logger.info("Redis connection successful, memory management enabled")
        MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory module not available: {e}")
    MEMORY_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error initializing memory module: {e}")
    MEMORY_AVAILABLE = False

# OAuth token cache
oauth_token = None
oauth_token_expiry = 0

# Cache for MCP tools
mcp_tools_cache = {}

# MCP server connection cache
mcp_servers = {}

# Model settings
settings = {
    "model": getattr(config, "LLM_MODEL", "gpt-3.5-turbo"),
    "temperature": 0.7,
    "stream": False,  # Non-streaming mode
    "max_tokens": 4000  # Request a larger response
}


# ===== MCP Server Connection Module =====
async def connect_to_mcp_server(server_url: str, server_name: str = None) -> bool:
    """
    Connect to an MCP server via HTTP
    """
    if not server_name:
        server_name = f"mcp-{len(mcp_servers) + 1}"
        
    try:
        # Check if server is reachable
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try to ping the server
            response = await client.get(f"{server_url}/ping")
            if response.status_code != 200:
                logger.error(f"Failed to connect to MCP server at {server_url}: {response.status_code}")
                return False
                
            # Get server information
            info_response = await client.get(f"{server_url}/")
            server_info = info_response.json()
            
            # Store server connection
            mcp_servers[server_name] = {
                "url": server_url,
                "info": server_info,
                "connected_at": time.time()
            }
            
            # Fetch tools from the server
            await fetch_tools_from_server(server_name)
            
            return True
    except Exception as e:
        logger.error(f"Error connecting to MCP server at {server_url}: {e}")
        return False


async def fetch_tools_from_server(server_name: str) -> List[Dict[str, Any]]:
    """
    Fetch tools from an MCP server
    """
    if server_name not in mcp_servers:
        logger.error(f"MCP server {server_name} not found")
        return []
        
    server = mcp_servers[server_name]
    server_url = server["url"]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{server_url}/tools")
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch tools from {server_name}: {response.status_code}")
                return []
                
            # Parse tools from response
            tools_data = response.json()
            
            # Handle different response formats
            tools = []
            if "tools" in tools_data:
                # Standard format
                tools = tools_data["tools"]
            elif "jsonrpc" in tools_data and "result" in tools_data:
                # JSON-RPC format
                tools = tools_data["result"].get("tools", [])
                
            # Cache tools
            mcp_tools_cache[server_name] = tools
            
            logger.info(f"Fetched {len(tools)} tools from {server_name}")
            return tools
    except Exception as e:
        logger.error(f"Error fetching tools from {server_name}: {e}")
        return []


async def execute_tool_on_server(server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
    """
    Execute a tool on an MCP server
    """
    if server_name not in mcp_servers:
        logger.error(f"MCP server {server_name} not found")
        return {"error": f"MCP server {server_name} not found"}
        
    server = mcp_servers[server_name]
    server_url = server["url"]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Use the /execute endpoint
            response = await client.post(
                f"{server_url}/execute",
                json={
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "context": {}  # Add context if needed
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to execute tool {tool_name} on {server_name}: {response.status_code}")
                return {"error": f"Failed to execute tool: {response.text}"}
                
            result = response.json()
            
            # Check for error in result
            if result.get("error"):
                logger.error(f"Error executing tool {tool_name}: {result['error']}")
                return {"error": result["error"]}
                
            return result.get("result", result)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name} on {server_name}: {e}")
        return {"error": f"Error executing tool: {str(e)}"}


# ===== Authentication Module =====
async def get_oauth_token() -> str:
    """
    Get OAuth token for LLM authentication
    """
    global oauth_token, oauth_token_expiry

    # Return cached token if still valid
    current_time = time.time()
    if oauth_token and current_time < oauth_token_expiry:
        logger.info("Using cached OAuth token")
        return oauth_token

    try:
        oauth_endpoint = getattr(config, "LLM_OAUTH_ENDPOINT", "")
        client_id = getattr(config, "LLM_OAUTH_CLIENT_ID", "")
        client_secret = getattr(config, "LLM_OAUTH_CLIENT_SECRET", "")

        if not all([oauth_endpoint, client_id, client_secret]):
            # If OAuth is not configured, return empty string
            logger.warning("OAuth not fully configured")
            return ""

        logger.info(f"Requesting new OAuth token from {oauth_endpoint}")

        # Disable SSL verification for internal endpoints
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.post(
                oauth_endpoint,
                data={
                    "grant_type": getattr(config, "LLM_OAUTH_GRANT_TYPE", "client_credentials"),
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": getattr(config, "LLM_OAUTH_SCOPE", "read")
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()

            token_data = response.json()
            token = token_data.get("access_token", "")

            # Update cache
            if token:
                oauth_token = token
                expires_in = token_data.get("expires_in", 3600)
                oauth_token_expiry = current_time + expires_in - 300  # 5 minutes buffer
                logger.info(f"OAuth token obtained, valid for {expires_in} seconds")
            else:
                logger.warning("OAuth response did not contain access_token")

            return token

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error in get_oauth_token: {e.response.status_code} - {e.response.text}")
        return ""
    except Exception as e:
        logger.error(f"Failed to get OAuth token: {e}")
        return ""


# ===== LLM API Module =====
async def call_llm(messages: List[Dict[str, str]], params: Dict[str, Any] = None) -> Union[Dict[str, Any], str]:
    """
    Call the configured LLM using OpenAI API
    """
    logger.info("call_llm called")
    if params is None:
        params = {}

    try:
        # Get OAuth token first (if configured)
        token = await get_oauth_token()
        logger.info(f"Got OAuth token: {'Yes' if token else 'No'}")

        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }

        # Add token if available, or use API key from environment
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif os.getenv('OPENAI_API_KEY'):
            headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
            logger.info("Using OPENAI_API_KEY from environment")
        else:
            logger.warning("No authentication method available")
            return "I'm experiencing authentication issues. Please check your OAuth configuration or API key."

        # Prepare full request payload
        request_body = {
            "model": params.get("model", getattr(config, "LLM_MODEL", "gpt-3.5-turbo")),
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "stream": False,  # Always use non-streaming mode
            "max_tokens": params.get("max_tokens", 4000)  # Request a larger response by default
        }

        # Add tools if needed
        if "tools" in params:
            request_body["tools"] = params["tools"]

        if "tool_choice" in params:
            request_body["tool_choice"] = params["tool_choice"]

        # Call the LLM with proper error handling and timeouts
        logger.info(f"Calling LLM at {getattr(config, 'LLM_BASE_URL', 'https://api.openai.com/v1/chat/completions')}")

        # Ensure the URL is correctly formed
        base_url = getattr(config, "LLM_BASE_URL", "https://api.openai.com/v1/chat/completions").rstrip('/')
        url = f"{base_url}"
        logger.info(f"Making request to {url}")

        # Need to disable SSL verification for internal endpoints
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            response = await client.post(
                url,
                headers=headers,
                json=request_body,
                timeout=60.0
            )

            # Log response status for debugging
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")

            if response.status_code != 200:
                logger.error(f"Error response: {response.text}")
                return f"I'm experiencing connection issues with my knowledge service: {response.status_code}. Please try again in a moment."

            # Try to parse the response
            try:
                result = response.json()
                logger.debug(f"Response content preview: {str(result)[:500]}...")
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {response.text[:500]}")
                return "I received an invalid response format from the API. Please try again."

    except httpx.ReadTimeout:
        logger.error("LLM request timed out")
        return "I'm sorry, but the response is taking longer than expected. Please try again with a simpler question."

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error in call_llm: {e.response.status_code} - {e.response.text}")
        log_error("call_llm_http", e)
        return f"I'm experiencing connection issues with my knowledge service: {e.response.status_code}. Please try again in a moment."

    except Exception as e:
        logger.error(f"Error in call_llm: {e}")
        log_error("call_llm", e)
        return f"I understand your question, but I'm having trouble processing it right now: {str(e)}. Please try again."


# ===== Tool Helper Functions =====
async def format_tools_for_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format MCP tools into OpenAI tools format"""
    openai_tools = []

    for tool in tools:
        # Get tool name and description
        name = tool.get("name", "")
        description = tool.get("description", "")
        
        # Get parameters schema
        parameters = {}
        if "parameters" in tool:
            parameters = tool["parameters"]
        elif "inputSchema" in tool:
            parameters = tool["inputSchema"]
        elif "input_schema" in tool:
            parameters = tool["input_schema"]
            
        # Ensure parameters has the proper structure
        if not parameters:
            parameters = {"type": "object", "properties": {}}
        elif "type" not in parameters:
            parameters = {"type": "object", "properties": parameters}
            
        # Create OpenAI tool format
        openai_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def format_table_output(result: str) -> str:
    """
    Format table outputs for better display in Chainlit
    
    This function detects and formats markdown tables in the result
    """
    # Check if result contains a markdown table
    if "| " in result and " |" in result:
        try:
            # Try to extract and format the table
            table_pattern = r"(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)"
            tables = re.findall(table_pattern, result)
            
            if tables:
                formatted_result = result
                for table in tables:
                    # Convert the markdown table to HTML for better display
                    try:
                        # Parse the markdown table
                        lines = table.strip().split('\n')
                        header_line = lines[0]
                        separator_line = lines[1]
                        data_lines = lines[2:]
                        
                        # Extract headers
                        headers = [h.strip() for h in header_line.split('|') if h.strip()]
                        
                        # Create a list of dictionaries for the data
                        data = []
                        for line in data_lines:
                            values = [v.strip() for v in line.split('|') if v.strip()]
                            if len(values) == len(headers):
                                data.append(dict(zip(headers, values)))
                        
                        # Create DataFrame and convert to HTML
                        df = pd.DataFrame(data)
                        html_table = df.to_html(index=False, classes=["table", "table-striped"])
                        
                        # Replace the markdown table with HTML
                        formatted_result = formatted_result.replace(table, f"\n{html_table}\n")
                    except Exception as e:
                        logger.error(f"Error formatting table: {e}")
                        # If formatting fails, keep the original table
                
                return formatted_result
        except Exception as e:
            logger.error(f"Error processing tables: {e}")
    
    # If no tables found or formatting failed, return the original result
    return result


def format_tool_result(result: Any) -> str:
    """Format tool result for display"""
    if isinstance(result, dict):
        # Check for error
        if "error" in result:
            return f"Error: {result['error']}"
            
        # Check for structured output
        if "output" in result:
            result = result["output"]
            
    # Convert to string if not already
    if not isinstance(result, str):
        try:
            # Try to format as JSON
            if isinstance(result, (dict, list)):
                result = json.dumps(result, indent=2)
            else:
                result = str(result)
        except Exception:
            result = str(result)
            
    # Format tables if present
    result = format_table_output(result)
    
    return result


# ===== Chainlit Event Handlers =====
@cl.on_chat_start
async def start():
    """Initialize chat session"""
    session_id = f"chat_{int(time.time())}"

    # Initialize message history
    cl.user_session.set(
        "message_history",
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant with MCP integration. You can access tools using MCP servers."
            }
        ]
    )

    # Initialize memory if available
    if MEMORY_AVAILABLE:
        # Create memory instance with session ID
        memory = ShortTermMemory(session_id=session_id)
        cl.user_session.set("memory", memory)
        cl.user_session.set("session_id", session_id)

        # Test memory by adding a system message
        await memory.add_message("system", "Chat session initialized")
        logger.info(f"Memory initialized for session {session_id}")

    # Pre-fetch OAuth token to verify it works
    token = await get_oauth_token()
    if token:
        logger.info("Successfully authenticated with OAuth")
    else:
        logger.warning("OAuth authentication failed or not configured")
        if os.getenv("OPENAI_API_KEY"):
            logger.info("Using OPENAI_API_KEY from environment as fallback")
        else:
            logger.warning("No authentication method available for LLM API")

    # Try to connect to default MCP server
    default_mcp_url = getattr(config, "MCP_SERVER_URL", "http://localhost:8080")
    connected = await connect_to_mcp_server(default_mcp_url, "default")
    
    if connected:
        await cl.Message(
            content=f"Connected to MCP server at {default_mcp_url}"
        ).send()
    else:
        await cl.Message(
            content=f"Failed to connect to MCP server at {default_mcp_url}. Please connect manually."
        ).send()

    # Send welcome message
    model_name = getattr(config, "LLM_MODEL", "gpt-3.5-turbo")
    await cl.Message(
        content=f"Welcome! I'm using {model_name} with MCP integration. Here's what I can help you with:\n\n"
                "1. Chat and answer questions\n"
                "2. Access tools and execute commands\n"
                "3. Process financial data and documents\n\n"
                "You can connect to an MCP server using the 'Connect MCP' button below."
    ).send()

    # Add connect button
    await cl.Message(content="").send(
        actions=[
            cl.Action(
                name="connect_mcp",
                label="Connect MCP",
                description="Connect to an MCP server",
                value={"default_url": default_mcp_url}
            ),
            cl.Action(
                name="clear_chat",
                label="Clear Chat",
                description="Clear the chat history"
            )
        ]
    )

    # Log interaction
    log_interaction(
        step="chat_start",
        message="Chat session initialized",
        session_id=session_id,
        status="success"
    )


@cl.action_callback("connect_mcp")
async def on_connect_mcp(action):
    """Handle MCP server connection action"""
    # Get the default URL from the action value
    default_url = "http://localhost:8080"
    if action.value and "default_url" in action.value:
        default_url = action.value["default_url"]
        
    # Show input form
    res = await cl.AskUserMessage(
        content="Enter the URL of the MCP server you want to connect to:",
        content_format="markdown",
        default=default_url
    ).send()
    
    if not res:
        await cl.Message("Connection cancelled").send()
        return
        
    server_url = res.strip()
    
    # Show server name input
    name_res = await cl.AskUserMessage(
        content="Enter a name for this MCP server connection:",
        content_format="markdown",
        default=f"mcp-{len(mcp_servers) + 1}"
    ).send()
    
    if not name_res:
        server_name = f"mcp-{len(mcp_servers) + 1}"
    else:
        server_name = name_res.strip()
    
    # Show connecting message
    connecting_msg = cl.Message(content=f"Connecting to MCP server at {server_url}...")
    await connecting_msg.send()
    
    # Try to connect
    connected = await connect_to_mcp_server(server_url, server_name)
    
    if connected:
        connecting_msg.content = f"✅ Connected to MCP server '{server_name}' at {server_url}"
        await connecting_msg.update()
        
        # Get tools count
        tools_count = len(mcp_tools_cache.get(server_name, []))
        
        await cl.Message(
            content=f"Found {tools_count} tools from '{server_name}' MCP server."
        ).send()
    else:
        connecting_msg.content = f"❌ Failed to connect to MCP server at {server_url}"
        await connecting_msg.update()


@cl.action_callback("clear_chat")
async def on_clear_chat(action):
    """Clear chat history"""
    session_id = cl.user_session.get("session_id")
    try:
        # Clear message history
        cl.user_session.set("message_history", [
            {
                "role": "system",
                "content": "You are a helpful AI assistant powered by the MCP server. You can access tools and resources through the MCP integration."
            }
        ])

        if MEMORY_AVAILABLE:
            memory = cl.user_session.get("memory")
            if memory:
                await memory.clear()

        log_interaction(
            step="clear_chat",
            message="Chat history cleared",
            session_id=session_id,
            status="success"
        )

        await cl.Message(content="Chat history cleared").send()
    except Exception as e:
        log_error("clear_chat", e, session_id)
        await cl.Message(content=f"Error clearing chat: {str(e)}").send()


@cl.step(type="tool")
async def execute_tool(tool_name: str, tool_input: Dict[str, Any]):
    """Execute a specific tool in an MCP server"""
    logger.info(f"Executing tool: {tool_name}")
    logger.info(f"Tool input: {tool_input}")

    # Find which MCP server has this tool
    server_name = None
    for name, tools in mcp_tools_cache.items():
        if any(tool["name"] == tool_name for tool in tools):
            server_name = name
            break

    if not server_name:
        return {"error": f"Tool '{tool_name}' not found in any connected MCP server"}

    try:
        # Execute the tool on the server
        result = await execute_tool_on_server(server_name, tool_name, tool_input)
        
        # Format the result
        formatted_result = format_tool_result(result)
        return formatted_result
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {"error": f"Error calling tool '{tool_name}': {str(e)}"}


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages"""
    # Get session ID for tracking
    session_id = cl.user_session.get("session_id", f"chat_{int(time.time())}")
    start_time = time.time()

    try:
        # Get message history
        message_history = cl.user_session.get("message_history", [])
        message_history.append({"role": "user", "content": message.content})

        # Log message receipt
        log_interaction(
            step="message_received",
            message=message.content,
            session_id=session_id,
            status="processing"
        )

        # Store in memory if available
        if MEMORY_AVAILABLE:
            memory = cl.user_session.get("memory")
            if memory:
                await memory.add_message("user", message.content)

        # Create initial message for response
        initial_msg = cl.Message(content="Thinking...")
        await initial_msg.send()

        # Get all available tools from connected MCP servers
        all_tools = []
        for server_name, tools in mcp_tools_cache.items():
            all_tools.extend(tools)

        # Prepare parameters for chat completion
        chat_params = {**settings}  # Use default settings which has stream=False
        if all_tools:
            openai_tools = await format_tools_for_openai(all_tools)
            chat_params["tools"] = openai_tools
            chat_params["tool_choice"] = "auto"
            logger.info(f"Including {len(openai_tools)} tools in request")

        # Call the model with tools
        response = await call_llm(message_history, chat_params)

        # Handle different response types
        if isinstance(response, str):
            # If response is a string, it's an error message
            initial_msg.content = response
            await initial_msg.update()
        else:
            # Response is a dict with the full response
            initial_response = ""
            tool_calls = []

            # Process the non-streaming response
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]

                if "message" in choice:
                    message_obj = choice["message"]

                    # Handle regular text response
                    if "content" in message_obj and message_obj["content"]:
                        initial_response = message_obj["content"]
                        
                        # Format tables in the response
                        formatted_response = format_table_output(initial_response)
                        initial_msg.content = formatted_response
                        await initial_msg.update()

                        # Add to message history
                        message_history.append({"role": "assistant", "content": initial_response})

                        # Store in memory if available
                        if MEMORY_AVAILABLE:
                            memory = cl.user_session.get("memory")
                            if memory:
                                await memory.add_message("assistant", initial_response)

                    # Handle tool calls
                    if "tool_calls" in message_obj:
                        tool_calls = message_obj["tool_calls"]

                        # Add tool call to message history
                        message_history.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls
                        })

                        # Process each tool call
                        for tool_call in tool_calls:
                            if tool_call["type"] == "function":
                                function = tool_call["function"]
                                tool_name = function["name"]

                                try:
                                    # Parse tool arguments
                                    tool_args = json.loads(function["arguments"])

                                    # Execute the tool
                                    with cl.Step(name=f"Executing tool: {tool_name}", type="tool"):
                                        tool_result = await execute_tool(tool_name, tool_args)

                                    # Format and display the tool result
                                    formatted_result = format_tool_result(tool_result)
                                    tool_result_msg = cl.Message(
                                        content=f"Tool Result from {tool_name}:\n{formatted_result}",
                                        author="Tool"
                                    )
                                    await tool_result_msg.send()

                                    # Add tool result to message history
                                    message_history.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": str(tool_result)
                                    })

                                except Exception as e:
                                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                                    logger.error(error_msg)
                                    await cl.Message(content=error_msg).send()

                        # Get a follow-up response after all tools have executed
                        if tool_calls:
                            follow_up_params = {**settings}
                            follow_up_result = await call_llm(message_history, follow_up_params)

                            if isinstance(follow_up_result, str):
                                await cl.Message(content=follow_up_result).send()
                            else:
                                follow_up_text = ""
                                if "choices" in follow_up_result and follow_up_result["choices"]:
                                    choice = follow_up_result["choices"][0]
                                    if "message" in choice and "content" in choice["message"]:
                                        follow_up_text = choice["message"]["content"]

                                if follow_up_text:
                                    # Format tables in the response
                                    formatted_follow_up = format_table_output(follow_up_text)
                                    follow_up_msg = cl.Message(content=formatted_follow_up)
                                    await follow_up_msg.send()

                                    # Add to message history
                                    message_history.append({"role": "assistant", "content": follow_up_text})

                                    # Store in memory if available
                                    if MEMORY_AVAILABLE:
                                        memory = cl.user_session.get("memory")
                                        if memory:
                                            await memory.add_message("assistant", follow_up_text)

        # Calculate processing time
        processing_time = time.time() - start_time
        processing_time_ms = processing_time * 1000

        # Log the processing time to the UI in a subtle way
        processing_time_msg = cl.Message(
            content=f"_Processing time: {processing_time:.2f} seconds ({processing_time_ms:.0f}ms)_",
            author="System"
        )
        await processing_time_msg.send()

        # Update session message history
        cl.user_session.set("message_history", message_history)

        # Log successful processing
        log_interaction(
            step="message_processed",
            message=message.content,
            session_id=session_id,
            response_length=len(initial_response) if 'initial_response' in locals() else 0,
            processing_time_ms=processing_time_ms,
            status="success"
        )

    except Exception as e:
        # Log error
        log_error("message_processing", e, session_id)
        error_message = f"Error: {str(e)}"
        await cl.Message(content=error_message).send()

        # Provide appropriate troubleshooting tips
        troubleshooting = (
            "Troubleshooting tips:\n"
            "1. Check that the OAuth configuration is correct\n"
            "2. Verify the LLM API endpoint is accessible\n"
            "3. Check that your API key or OAuth credentials are valid\n"
            "4. Ensure the model supports function calling\n"
            "5. Check network connectivity to all services"
        )
        await cl.Message(content=troubleshooting).send()


if __name__ == "__main__":
    logger.info("Starting Chainlit app with MCP integration...")
