"""
Agent Registration API

This module provides functions for registering agents with the MCP server.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from dataclasses import dataclass, field

from app.agent import registry, Agent
from app.registry.tools import registry as tool_registry
from app.registry.resources import registry as resource_registry
from app.registry.prompts import registry as prompt_registry

# Get logger
logger = logging.getLogger('mcp_server.agent.registration')

# Create FastAPI router
router = APIRouter()

@dataclass
class Agent:
    """Agent information"""
    id: str
    name: str
    description: str
    namespace: str
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Registry for external agents"""
    def __init__(self):
        self._agents: Dict[str, Agent] = {}
    
    def register(self, agent: Agent) -> bool:
        """Register a new agent"""
        if agent.id in self._agents:
            return False
        self._agents[agent.id] = agent
        return True
    
    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        return True
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> Dict[str, Agent]:
        """List all registered agents"""
        return self._agents.copy()


# Create global registry instance
registry = AgentRegistry()

@router.post("/agents/register")
async def register_agent(
    name: str,
    description: str,
    namespace: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Register a new agent with the MCP server"""
    try:
        # Generate a unique ID if not provided
        agent_id = str(uuid.uuid4())
        
        # Use name as namespace if not provided
        if not namespace:
            namespace = name.lower().replace(" ", "_")
        
        # Create agent
        agent = Agent(
            id=agent_id,
            name=name,
            description=description,
            namespace=namespace,
            capabilities=capabilities or [],
            metadata=metadata or {}
        )
        
        # Register agent
        if not registry.register(agent):
            raise HTTPException(status_code=400, detail=f"Agent with ID {agent_id} already exists")
        
        logger.info(f"Registered agent: {agent.name} (ID: {agent.id}, Namespace: {agent.namespace})")
        
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "namespace": agent.namespace,
            "capabilities": agent.capabilities,
            "message": "Agent registered successfully"
        }
    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error registering agent: {str(e)}")

@router.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents"""
    try:
        agents = registry.list_agents()
        return {
            "count": len(agents),
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "description": agent.description,
                    "namespace": agent.namespace,
                    "capabilities": agent.capabilities
                }
                for agent in agents.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> Dict[str, Any]:
    """Get agent information by ID"""
    try:
        agent = registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        
        # Get components in the agent's namespace
        tools = tool_registry.list_tools(namespace=agent.namespace)
        resources = resource_registry.list_resources(namespace=agent.namespace)
        prompts = prompt_registry.list_prompts(namespace=agent.namespace)
        
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "namespace": agent.namespace,
            "capabilities": agent.capabilities,
            "metadata": agent.metadata,
            "components": {
                "tools": tools,
                "resources": resources,
                "prompts": prompts
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting agent: {str(e)}")

@router.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str) -> Dict[str, Any]:
    """Unregister an agent"""
    try:
        if not registry.unregister(agent_id):
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        
        logger.info(f"Unregistered agent with ID: {agent_id}")
        
        return {
            "message": f"Agent with ID {agent_id} unregistered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error unregistering agent: {str(e)}") 