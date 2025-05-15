"""
Agent Management Page for Chainlit UI

This page provides a user interface for managing and interacting with agents.
"""

import chainlit as cl
from typing import Dict, List, Optional
import httpx
import json
from app.config import config

# Constants
API_BASE_URL = f"http://{config.API_SERVER_HOST}:{config.API_SERVER_PORT}"

async def get_agents() -> List[Dict]:
    """Get list of registered agents"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/agents")
        if response.status_code == 200:
            return response.json()
        return []

async def route_request(request: str, context: Optional[Dict] = None) -> Dict:
    """Route a request to an appropriate agent"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/route",
            json={"request": request, "context": context or {}}
        )
        if response.status_code == 200:
            return response.json()
        return {"message": f"Error: {response.text}"}

@cl.on_chat_start
async def start():
    """Initialize the agent management page"""
    # Create a welcome message
    await cl.Message(
        content="Welcome to the Agent Management Interface! Here you can:\n"
                "1. View all registered agents and their capabilities\n"
                "2. Route requests to specialized agents\n"
                "3. Get detailed information about specific agents\n\n"
                "Type 'list agents' to see available agents or describe your request to route it to an appropriate agent."
    ).send()

    # Add action buttons
    actions = [
        cl.Action(name="list_agents", label="List Agents", description="Show all registered agents"),
        cl.Action(name="refresh", label="Refresh", description="Refresh agent list")
    ]
    await cl.Message(content="Available actions:", actions=actions).send()

@cl.action_callback("list_agents")
async def on_list_agents(action):
    """Handle list agents action"""
    agents = await get_agents()
    if not agents:
        await cl.Message(content="No agents currently registered.").send()
        return

    # Create a message with agent information
    content = "## Registered Agents\n\n"
    for agent in agents:
        capabilities = "\n".join([
            f"- {cap['name']}: {cap['description']}"
            for cap in agent["capabilities"]
        ])
        content += (
            f"### {agent['name']} (ID: {agent['id']})\n"
            f"Description: {agent['description']}\n"
            f"Endpoint: {agent['endpoint']}\n"
            f"Capabilities:\n{capabilities}\n\n"
        )

    await cl.Message(content=content).send()

@cl.action_callback("refresh")
async def on_refresh(action):
    """Handle refresh action"""
    await on_list_agents(action)

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    # Check for specific commands
    if message.content.lower() == "list agents":
        await on_list_agents(None)
        return

    # Try to route the request to an appropriate agent
    response = await route_request(message.content)
    
    if response.get("agent_id"):
        # Create a message with routing information
        content = (
            f"Request routed to agent: {response['agent_name']}\n"
            f"Confidence: {response['confidence']:.2f}\n\n"
            f"Message: {response['message']}"
        )
        await cl.Message(content=content).send()
        
        # Add a button to get more information about the agent
        actions = [
            cl.Action(
                name=f"agent_info_{response['agent_id']}",
                label=f"Get {response['agent_name']} Info",
                description=f"Get detailed information about {response['agent_name']}"
            )
        ]
        await cl.Message(content="Available actions:", actions=actions).send()
    else:
        await cl.Message(content=response["message"]).send()

@cl.action_callback("agent_info_*")
async def on_agent_info(action):
    """Handle agent info action"""
    agent_id = action.name.split("_")[-1]
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/agents/{agent_id}")
        if response.status_code == 200:
            agent = response.json()
            
            # Format capabilities
            capabilities = "\n".join([
                f"### {cap['name']}\n"
                f"Description: {cap['description']}\n"
                f"Parameters: {json.dumps(cap['parameters'], indent=2)}\n"
                f"Examples: {json.dumps(cap['examples'], indent=2)}"
                for cap in agent["capabilities"]
            ])
            
            # Format metadata
            metadata = "\n".join([
                f"- {key}: {value}"
                for key, value in agent["metadata"].items()
            ]) if agent["metadata"] else "No metadata available"
            
            content = (
                f"# Agent Information: {agent['name']}\n\n"
                f"ID: {agent['id']}\n"
                f"Description: {agent['description']}\n"
                f"Endpoint: {agent['endpoint']}\n\n"
                f"## Capabilities\n{capabilities}\n\n"
                f"## Metadata\n{metadata}"
            )
            
            await cl.Message(content=content).send()
        else:
            await cl.Message(content=f"Error getting agent information: {response.text}").send() 