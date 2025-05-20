"""
Chainlit UI App for MCP Server Integration

This module provides a web-based UI for interacting with the MCP server using OpenAI API.

1. Architecture Overview:
    - Chainlit web UI frontend
    - Integration with MCP (Model Context Protocol) servers
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
    - Dynamically collect tools from connected MCP servers
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
    - Run with: chainlit run chainlit_app.py
    - Connect to MCP servers via UI
    - Interact with AI assistant with access to all connected tools

Communications:
    - SSE: This is HTTP-based transport that requires the endpoints
    - STDIO: This is command-based transport where Chainlit spawns a sub-process and communicates via stdio/stdout.
"""
import chainlit as cl
from typing import Dict, Any, List, Optional, Union
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
import os
import sys
import httpx
from pathlib import Path
import time
import json
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from app modules
from app.config import config
from app.utils.logger import logger, log_interaction, log_error


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

# Model settings
settings = {
    "model": config.LLM_MODEL,
    "temperature": 0.7,
    "stream": False,  # Non-streaming mode
    "max_tokens": 4000  # Request a larger response
}


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
        oauth_endpoint = config.LLM_OAUTH_ENDPOINT
        client_id = config.LLM_OAUTH_CLIENT_ID
        client_secret = config.LLM_OAUTH_CLIENT_SECRET

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
                    "grant_type": config.LLM_OAUTH_GRANT_TYPE or "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": config.LLM_OAUTH_SCOPE or "read"
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
            "model": params.get("model", config.LLM_MODEL),
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
        logger.info(f"Calling LLM at {config.LLM_BASE_URL}")

        # Ensure the URL is correctly formed
        base_url = config.LLM_BASE_URL.rstrip('/')
        url = f"{base_url}"
        logger.info(f"Making request to {url}")

        # Need to disable SSL verification for internal RBC endpoints
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
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def format_calltoolresult_content(result):
    """Extract text content from a CallToolResult object"""
    text_contents = []

    if isinstance(result, CallToolResult):
        for content_item in result.content:
            # Extract text from TextContent type items
            if isinstance(content_item, TextContent):
                text_contents.append(content_item.text)

    if text_contents:
        return "\n".join(text_contents)
    return str(result)


# ===== Response Handling Module =====
async def process_llm_response(result: Union[Dict[str, Any], str], initial_msg: cl.Message,
                               message_history: List[Dict[str, Any]],
                               memory: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Process the LLM response and update UI
    """
    # Handle error response (string)
    if isinstance(result, str):
        initial_msg.content = result
        await initial_msg.update()
        return message_history

    logger.info(f"Processing LLM response")

    # Process the response
    if "choices" in result and result["choices"]:
        choice = result["choices"][0]

        if "message" in choice:
            message_obj = choice["message"]

            # Handle regular text response
            if "content" in message_obj and message_obj["content"]:
                content = message_obj["content"]

                # Log content length for debugging
                logger.info(f"Received content of length: {len(content)}")

                # Create a new message instead of updating (helps with display issues)
                await initial_msg.remove()

                new_msg = cl.Message(content=content)
                await new_msg.send()

                # Add to message history
                message_history.append({"role": "assistant", "content": content})

                # Store in memory if available
                if memory:
                    await memory.add_message("assistant", content)

            # Handle tool calls
            if "tool_calls" in message_obj and message_obj["tool_calls"]:
                await process_tool_calls(
                    message_obj["tool_calls"],
                    message_history,
                    memory
                )
        else:
            logger.warning("Response did not contain a 'message' field")
            initial_msg.content = "Received an unexpected response from the model."
            await initial_msg.update()
    else:
        logger.warning(f"Unexpected response format: {result}")
        initial_msg.content = "Received an unexpected response format from the model."
        await initial_msg.update()

    return message_history


async def process_tool_calls(tool_calls, message_history, memory=None):
    """
    Process tool calls from the LLM response
    """
    # Add assistant message with tool calls to history
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
            tool_args = json.loads(function["arguments"])

            # Execute the tool
            with cl.Step(name=f"Executing tool: {tool_name}", type="tool"):
                tool_result = await execute_tool(tool_name, tool_args)

            # Display tool result
            tool_result_content = format_calltoolresult_content(tool_result)
            tool_result_msg = cl.Message(
                content=f"Tool Result from {tool_name}:\n{tool_result_content}",
                author="Tool"
            )
            await tool_result_msg.send()

            # Add tool result to message history
            message_history.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result_content
            })

    # Get a follow-up response after tool execution
    follow_up_result = await call_llm(message_history, settings)

    if isinstance(follow_up_result, str):
        # Error response
        await cl.Message(content=follow_up_result).send()
    else:
        # Process follow-up response
        follow_up_choice = follow_up_result["choices"][0]
        follow_up_message = follow_up_choice.get("message", {})
        follow_up_content = follow_up_message.get("content", "")

        if follow_up_content:
            follow_up_msg = cl.Message(content=follow_up_content)
            await follow_up_msg.send()

            # Add to message history
            message_history.append({"role": "assistant", "content": follow_up_content})

            # Store in memory if available
            if memory:
                await memory.add_message("assistant", follow_up_content)


# ===== Chainlit Event Handlers =====
@cl.on_chat_start
async def start():
    """Initialize chat session and MCP connection"""
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

    # Send welcome message
    await cl.Message(
        content=f"Welcome! I'm using {config.LLM_MODEL} with MCP integration. Here's what I can help you with:\n\n"
                "1. Chat and answer questions\n"
                "2. Access tools and execute commands\n"
                "3. Process financial data and documents"
    ).send()

    # Log interaction
    log_interaction(
        step="chat_start",
        message="Chat session initialized",
        session_id=session_id,
        status="success"
    )


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Handle successful MCP server connection"""
    await cl.Message(f"Connected to MCP server: {connection.name}").send()

    try:
        # Get list of available tools
        result = await session.list_tools()

        # Format tools for storage
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

        # Cache tools both globally and in session
        mcp_tools_cache[connection.name] = tools
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)

        await cl.Message(
            f"Found {len(tools)} tools from {connection.name} MCP server."
        ).send()
    except Exception as e:
        logger.error(f"MCP connection error: {e}")
        await cl.Message(f"Error listing tools from MCP server: {str(e)}").send()


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Handle MCP server disconnection"""
    if name in mcp_tools_cache:
        del mcp_tools_cache[name]

    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)

    await cl.Message(f"Disconnected from MCP server: {name}").send()


@cl.step(type="tool")
async def execute_tool(tool_name: str, tool_input: Dict[str, Any]):
    """Execute a specific tool in an MCP server"""
    logger.info(f"Executing tool: {tool_name}")
    logger.info(f"Tool input: {tool_input}")

    # Find which MCP server has this tool
    mcp_name = None
    mcp_tools = cl.user_session.get("mcp_tools", {})

    for conn_name, tools in mcp_tools.items():
        if any(tool["name"] == tool_name for tool in tools):
            mcp_name = conn_name
            break

    if not mcp_name:
        return {"error": f"Tool '{tool_name}' not found in any connected MCP server"}

    # Get the session for this MCP server
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

    try:
        # Call the tool with the provided input
        result = await mcp_session.call_tool(tool_name, tool_input)
        return result
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
        mcp_tools = cl.user_session.get("mcp_tools", {})
        all_tools = []
        for connection_tools in mcp_tools.values():
            all_tools.extend(connection_tools)

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
                        initial_msg.content = initial_response
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
                                    tool_result_content = format_calltoolresult_content(tool_result)
                                    tool_result_msg = cl.Message(
                                        content=f"Tool Result from {tool_name}:\n{tool_result_content}",
                                        author="Tool"
                                    )
                                    await tool_result_msg.send()

                                    # Add tool result to message history
                                    message_history.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": tool_result_content
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
                                    follow_up_msg = cl.Message(content=follow_up_text)
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

@cl.on_stop
async def on_stop():
    """Clean up resources when the chat session ends"""
    global oauth_token, oauth_token_expiry

    # Log that we're stopping
    logger.info("Chat session ending")

    # Clear token cache
    oauth_token = None
    oauth_token_expiry = 0


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


if __name__ == "__main__":
    logger.info("Starting Chainlit app with MCP integration...")

