"""
Agent interaction tools for the MCP server
"""

import logging
from typing import Dict, List, Optional, Any
from fastmcp import Context
from app.agents.manager import agent_registry, Agent, AgentCapability

logger = logging.getLogger(__name__)

def register_tools(mcp):
    """Register agent interaction tools with the MCP server"""

    @mcp.tool()
    async def list_agents(ctx: Context) -> str:
        """
        List all registered agents and their capabilities.

        Call this tool when you need to know what agents are available and what they can do.

        Args:
            ctx: The MCP server provided context.

        Returns:
            A formatted string containing information about all registered agents.
        """
        try:
            agents = agent_registry.list_agents()
            if not agents:
                return "No agents currently registered."

            # Format agent information
            agent_info = []
            for agent in agents:
                capabilities = "\n".join([
                    f"  - {cap.name}: {cap.description}"
                    for cap in agent.capabilities
                ])
                agent_info.append(
                    f"## {agent.name} (ID: {agent.id})\n"
                    f"Description: {agent.description}\n"
                    f"Endpoint: {agent.endpoint}\n"
                    f"Capabilities:\n{capabilities}\n"
                )

            return "\n".join(agent_info)
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return f"Error listing agents: {e}"

    @mcp.tool()
    async def route_to_agent(ctx: Context, request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Route a request to the most appropriate agent using Cohere Compass.

        Call this tool when you need to delegate a request to a specialized agent.

        Args:
            ctx: The MCP server provided context.
            request: The request to route to an agent.
            context: Optional context information for routing.

        Returns:
            Information about the selected agent or an error message.
        """
        try:
            agent = await agent_registry.route_request(request, context)
            if agent:
                capabilities = "\n".join([
                    f"- {cap.name}: {cap.description}"
                    for cap in agent.capabilities
                ])
                return (
                    f"Request routed to agent: {agent.name} (ID: {agent.id})\n"
                    f"Description: {agent.description}\n"
                    f"Capabilities:\n{capabilities}\n"
                    f"Endpoint: {agent.endpoint}"
                )
            else:
                return "No suitable agent found for the request."
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return f"Error routing request: {e}"

    @mcp.tool()
    async def get_agent_info(ctx: Context, agent_id: str) -> str:
        """
        Get detailed information about a specific agent.

        Call this tool when you need detailed information about a particular agent.

        Args:
            ctx: The MCP server provided context.
            agent_id: The ID of the agent to get information about.

        Returns:
            Detailed information about the specified agent.
        """
        try:
            agent = agent_registry.get_agent(agent_id)
            if not agent:
                return f"Agent {agent_id} not found."

            capabilities = "\n".join([
                f"  - {cap.name}: {cap.description}\n"
                f"    Parameters: {cap.parameters}\n"
                f"    Examples: {cap.examples}"
                for cap in agent.capabilities
            ])

            metadata = "\n".join([
                f"  - {key}: {value}"
                for key, value in agent.metadata.items()
            ]) if agent.metadata else "  No metadata available"

            return (
                f"# Agent Information: {agent.name} (ID: {agent.id})\n\n"
                f"Description: {agent.description}\n\n"
                f"Endpoint: {agent.endpoint}\n\n"
                f"## Capabilities\n{capabilities}\n\n"
                f"## Metadata\n{metadata}"
            )
        except Exception as e:
            logger.error(f"Error getting agent info: {e}")
            return f"Error getting agent info: {e}" 