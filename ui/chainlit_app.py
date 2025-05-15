"""
Chainlit UI for MCP Server

This module provides a web-based UI for interacting with the MCP server.
"""

import os
import chainlit as cl
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config as app_config
from app.client import mcp_client
from app.utils.logger import logger, log_interaction, log_error

# Try to import memory module, but make it optional
try:
    from app.memory import ShortTermMemory
    # Test Redis connection by creating a temporary instance
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

# Load environment variables
load_dotenv()

# Initialize MCP connection and tools cache
mcp_connection = None
mcp_tools_cache = {}

# Model settings
settings = {
    "model": app_config.LLM_MODEL,
    "temperature": 0.3,
    "stream": True
}

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session and MCP connection"""
    global mcp_connection
    session_id = f"chat_{int(time.time())}"
    
    try:
        # Initialize message history
        await cl.user_session.set(
            "message_history",
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant powered by the MCP server. You can access tools and resources through the MCP integration."
                }
            ]
        )
        
        # Initialize memory if available
        if MEMORY_AVAILABLE:
            # Create memory instance with session ID
            memory = ShortTermMemory(session_id=session_id)
            await cl.user_session.set("memory", memory)
            await cl.user_session.set("session_id", session_id)
            
            # Test memory by adding a system message
            await memory.add_message("system", "Chat session initialized")
            
            logger.info(f"Memory initialized for session {session_id}")
        
        # Set up MCP connection
        try:
            # Try SSE connection first
            mcp_connection = await cl.connect_mcp(
                transport="sse",
                url=f"http://{app_config.MCP_SERVER_HOST}:8081"
            )
            logger.info("MCP SSE connection established")
        except Exception as e:
            logger.warning(f"Failed to establish SSE connection: {e}")
            try:
                # Fall back to stdio connection
                mcp_connection = await cl.connect_mcp(
                    transport="stdio",
                    command=["python", "-m", "app.mcp_server"]
                )
                logger.info("MCP stdio connection established")
            except Exception as e:
                logger.error(f"Failed to establish stdio connection: {e}")
                mcp_connection = None
        
        if mcp_connection is None:
            logger.warning("MCP server is not accessible")
            await cl.Message(
                content="Warning: MCP server is not accessible. Some features may be limited."
            ).send()
        else:
            # Store connection in user session
            await cl.user_session.set("mcp_connection", mcp_connection)
            
            # Get available tools
            tools = await mcp_connection.get_tools()
            if tools:
                logger.info(f"Available tools: {list(tools.keys())}")
                # Cache tools for this connection
                mcp_tools_cache[mcp_connection.name] = tools
                await cl.user_session.set("mcp_tools", mcp_tools_cache)
        
        # Send welcome message with model info
        await cl.Message(
            content=f"Welcome to the MCP Server UI! I'm using {settings['model']} and I'm ready to help you with your tasks.\n\n"
                   f"Available features:\n"
                   f"1. Chat with AI assistant\n"
                   f"2. Execute tools and commands\n"
                   f"3. View tool execution history\n"
                   f"4. Session management"
        ).send()
        
        log_interaction(
            step="chat_start",
            message="Chat session initialized",
            session_id=session_id,
            status="success"
        )
    except Exception as e:
        log_error("chat_start", e, session_id)
        await cl.Message(
            content="Welcome! Note: Memory management is not available."
        ).send()

@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    """Handle MCP server connection"""
    await cl.Message(f"Connected to MCP server: {connection.name}").send()
    
    try:
        # Get available tools
        tools = await session.list_tools()
        tool_list = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema
            }
            for t in tools.tools
        ]
        
        # Cache tools
        mcp_tools_cache[connection.name] = tool_list
        await cl.user_session.set("mcp_tools", mcp_tools_cache)
        
        await cl.Message(
            f"Found {len(tool_list)} tools from {connection.name} MCP server."
        ).send()
    except Exception as e:
        await cl.Message(f"Error listing tools from MCP server: {str(e)}").send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session):
    """Handle MCP server disconnection"""
    if name in mcp_tools_cache:
        del mcp_tools_cache[name]
        await cl.user_session.set("mcp_tools", mcp_tools_cache)
    
    await cl.Message(f"Disconnected from MCP server: {name}").send()

@cl.step(type="tool")
async def execute_tool(tool_name: str, tool_input: Dict[str, Any]):
    """Execute a tool through MCP"""
    logger.info(f"Executing tool: {tool_name}")
    logger.info(f"Tool input: {tool_input}")
    
    mcp_name = None
    mcp_tools = await cl.user_session.get("mcp_tools", {})
    
    # Find which MCP server has this tool
    for conn_name, tools in mcp_tools.items():
        if any(tool["name"] == tool_name for tool in tools):
            mcp_name = conn_name
            break
    
    if not mcp_name:
        return {"error": f"Tool '{tool_name}' not found in any connected MCP server"}
    
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    
    try:
        result = await mcp_session.call_tool(tool_name, tool_input)
        return result
    except Exception as e:
        return {"error": f"Error calling tool '{tool_name}': {str(e)}"}

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages"""
    session_id = cl.user_session.get("session_id")
    mcp_connection = cl.user_session.get("mcp_connection")
    start_time = time.time()
    
    try:
        # Get message history
        message_history = await cl.user_session.get("message_history", [])
        message_history.append({"role": "user", "content": message.content})
        
        log_interaction(
            step="message_received",
            message=message.content,
            session_id=session_id,
            status="processing"
        )
        
        # Store message in memory if available
        if MEMORY_AVAILABLE:
            memory = cl.user_session.get("memory")
            if memory:
                await memory.add_message("user", message.content)
        
        # Create initial message for streaming
        initial_msg = cl.Message(content="")
        await initial_msg.send()
        
        # Process message through MCP connection if available
        if mcp_connection:
            try:
                # Get available tools
                mcp_tools = await cl.user_session.get("mcp_tools", {})
                all_tools = []
                for connection_tools in mcp_tools.values():
                    all_tools.extend(connection_tools)
                
                # Process message through MCP
                response = await mcp_connection.process_message(
                    message.content,
                    tools=all_tools if all_tools else None
                )
                
                # Stream the response
                if isinstance(response, str):
                    await initial_msg.stream_token(response)
                else:
                    # Handle structured response with tool calls
                    if hasattr(response, 'tool_calls'):
                        for tool_call in response.tool_calls:
                            # Execute tool in a step
                            with cl.Step(name=f"Executing tool: {tool_call.name}", type="tool"):
                                tool_result = await execute_tool(
                                    tool_call.name,
                                    json.loads(tool_call.arguments)
                                )
                            
                            # Display tool result
                            tool_result_msg = cl.Message(
                                content=f"Tool Result from {tool_call.name}:\n{tool_result}",
                                author="Tool"
                            )
                            await tool_result_msg.send()
                            
                            # Add to message history
                            message_history.append({
                                "role": "tool",
                                "tool_call_id": f"call_{len(message_history)}",
                                "content": str(tool_result)
                            })
                    
                    # Stream any follow-up response
                    if hasattr(response, 'content'):
                        await initial_msg.stream_token(response.content)
                
                # Store response in memory if available
                if MEMORY_AVAILABLE:
                    memory = cl.user_session.get("memory")
                    if memory:
                        await memory.add_message("assistant", str(response))
                
                # Log successful interaction
                log_interaction(
                    step="message_processed",
                    message=message.content,
                    session_id=session_id,
                    response_length=len(str(response)),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    status="success"
                )
                
            except Exception as e:
                logger.error(f"Error processing message through MCP: {e}")
                # Fall back to MCP client
                result = await mcp_client.process_message(message.content, session_id)
                await cl.Message(content=result["response"]).send()
        else:
            # Use MCP client if no MCP connection
            result = await mcp_client.process_message(message.content, session_id)
            await cl.Message(content=result["response"]).send()
        
        # Update message history
        await cl.user_session.set("message_history", message_history)
        
    except Exception as e:
        log_error("message_processing", e, session_id)
        await cl.Message(
            content=f"Error processing message: {str(e)}"
        ).send()

@cl.on_stop
async def on_stop():
    """Clean up when chat session ends"""
    global mcp_connection
    if mcp_connection:
        try:
            await mcp_connection.close()
            mcp_connection = None
            logger.info("MCP connection closed")
        except Exception as e:
            logger.error(f"Error closing MCP connection: {e}")

@cl.action_callback("clear_chat")
async def on_clear_chat(action):
    """Clear chat history"""
    session_id = cl.user_session.get("session_id")
    try:
        # Clear message history
        await cl.user_session.set("message_history", [
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