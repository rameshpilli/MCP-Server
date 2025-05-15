"""
Chainlit app for the CRM MCP Server.
Provides a chat interface for interacting with the CRM tools.
"""

import os
import chainlit as cl
import httpx
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = f"http://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8000')}"
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Set up the chat UI
    await cl.Message(
        content="Welcome to the CRM MCP Server! I can help you with:\n"
                "1. Getting top clients by revenue\n"
                "2. Analyzing client value by product\n"
                "3. Tracking client value over time\n\n"
                "What would you like to know?",
        author="Assistant"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    try:
        # Show thinking message
        msg = cl.Message(content="Thinking...")
        await msg.send()

        # Call the MCP server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}{API_PREFIX}/chat",
                json={
                    "message": message.content,
                    "session_id": cl.user_session.get("id")
                }
            )
            
            if response.status_code != 200:
                await msg.update(content=f"Error: {response.text}")
                return

            result = response.json()
            
            # Update message with the response
            await msg.update(
                content=result.get("response", "No response received"),
                elements=[
                    cl.Text(
                        name="tools_executed",
                        content=f"Tools used: {', '.join(result.get('tools_executed', []))}"
                    ) if result.get("tools_executed") else None,
                    cl.Text(
                        name="intent",
                        content=f"Intent: {result.get('intent', 'unknown')}"
                    ) if result.get("intent") else None
                ]
            )

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")

@cl.on_stop
async def stop():
    """Clean up when the chat session ends"""
    # Add any cleanup code here
    pass 