"""
Agent Registration API

This module provides FastAPI endpoints for agent registration and management.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.agent import registry, Agent

# Create FastAPI router
router = APIRouter()


class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration"""
    name: str
    description: str
    namespace: str
    capabilities: List[str] = []
    metadata: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Response model for agent information"""
    id: str
    name: str
    description: str
    namespace: str
    capabilities: List[str]
    metadata: Dict[str, Any]


@router.post("/agents/register", response_model=AgentResponse)
async def register_agent(request: AgentRegistrationRequest) -> AgentResponse:
    """
    Register a new agent

    Args:
        request: Agent registration request

    Returns:
        Registered agent information

    Raises:
        HTTPException: If registration fails
    """
    try:
        # Generate unique ID for the agent
        agent_id = f"{request.namespace}-{request.name.lower().replace(' ', '-')}"

        # Check if agent already exists
        if registry.get_agent(agent_id):
            raise HTTPException(status_code=409, detail="Agent already registered")

        # Create and register agent
        agent = Agent(
            id=agent_id,
            name=request.name,
            description=request.description,
            namespace=request.namespace,
            capabilities=request.capabilities,
            metadata=request.metadata
        )

        if not registry.register(agent):
            raise HTTPException(status_code=500, detail="Failed to register agent")

        return AgentResponse(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            namespace=agent.namespace,
            capabilities=agent.capabilities,
            metadata=agent.metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents", response_model=List[AgentResponse])
async def list_agents() -> List[AgentResponse]:
    """
    List all registered agents

    Returns:
        List of registered agents
    """
    try:
        agents = registry.list_agents()
        return [
            AgentResponse(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                namespace=agent.namespace,
                capabilities=agent.capabilities,
                metadata=agent.metadata
            )
            for agent in agents.values()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    """
    Get agent information

    Args:
        agent_id: ID of the agent to retrieve

    Returns:
        Agent information

    Raises:
        HTTPException: If agent not found
    """
    try:
        agent = registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return AgentResponse(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            namespace=agent.namespace,
            capabilities=agent.capabilities,
            metadata=agent.metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str) -> Dict[str, Any]:
    """
    Unregister an agent

    Args:
        agent_id: ID of the agent to unregister

    Returns:
        Success message

    Raises:
        HTTPException: If agent not found or unregistration fails
    """
    try:
        if not registry.unregister(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")

        return {"message": f"Agent {agent_id} unregistered successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 