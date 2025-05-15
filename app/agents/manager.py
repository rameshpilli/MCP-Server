"""
Agent Manager Module

This module handles agent registration, discovery, and routing using Cohere Compass
for intelligent request routing.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel
import cohere
from app.config import config

logger = logging.getLogger(__name__)

class AgentCapability(BaseModel):
    """Represents a capability that an agent can perform"""
    name: str
    description: str
    parameters: Dict[str, Any]
    examples: List[Dict[str, Any]]

class Agent(BaseModel):
    """Represents a registered agent"""
    id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    endpoint: str
    metadata: Dict[str, Any]

@dataclass
class AgentRegistry:
    """Registry for managing agents and their capabilities"""
    agents: Dict[str, Agent] = None
    cohere_client: Optional[cohere.Client] = None

    def __post_init__(self):
        self.agents = {}
        if hasattr(config, 'COHERE_API_KEY'):
            self.cohere_client = cohere.Client(config.COHERE_API_KEY)

    def register_agent(self, agent: Agent) -> None:
        """Register a new agent"""
        if agent.id in self.agents:
            logger.warning(f"Agent {agent.id} already registered, updating...")
        self.agents[agent.id] = agent
        logger.info(f"Registered agent {agent.id} with {len(agent.capabilities)} capabilities")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
        else:
            logger.warning(f"Agent {agent_id} not found")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[Agent]:
        """List all registered agents"""
        return list(self.agents.values())

    async def route_request(self, request: str, context: Dict[str, Any] = None) -> Optional[Agent]:
        """
        Route a request to the most appropriate agent using Cohere Compass
        
        Args:
            request: The user request to route
            context: Additional context for routing
            
        Returns:
            The most appropriate agent for handling the request, or None if no suitable agent found
        """
        if not self.cohere_client or not self.agents:
            logger.warning("Cohere client not configured or no agents registered")
            return None

        try:
            # Prepare agent descriptions for routing
            agent_descriptions = []
            for agent in self.agents.values():
                capabilities_text = "\n".join([
                    f"- {cap.name}: {cap.description}"
                    for cap in agent.capabilities
                ])
                agent_descriptions.append(
                    f"Agent: {agent.name}\n"
                    f"Description: {agent.description}\n"
                    f"Capabilities:\n{capabilities_text}"
                )

            # Use Cohere to find the most relevant agent
            response = self.cohere_client.rerank(
                query=request,
                documents=agent_descriptions,
                top_n=1,
                model="rerank-english-v2.0"
            )

            if response.results:
                # Get the index of the best matching agent
                best_match_idx = response.results[0].index
                best_match_agent = list(self.agents.values())[best_match_idx]
                
                # Only return the agent if the relevance score is high enough
                if response.results[0].relevance_score > 0.7:
                    logger.info(f"Routed request to agent {best_match_agent.id} "
                              f"with score {response.results[0].relevance_score}")
                    return best_match_agent
                else:
                    logger.info("No agent found with sufficient relevance score")
                    return None

        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return None

# Create global agent registry instance
agent_registry = AgentRegistry() 