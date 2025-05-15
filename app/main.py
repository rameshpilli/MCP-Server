from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from .config import Config
from .client import mcp_client
from .agent.registration import router as agent_router
from .mcp_bridge import MCPBridge
from app.agents.manager import Agent, AgentCapability, agent_registry

# Setup logging
logger = logging.getLogger('mcp_app')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting MCP API server')

# Create config instance
config = Config()

app = FastAPI(
    title="MCP API Server",
    description="API Server for MCP",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include agent registration router
app.include_router(agent_router, prefix="/api/v1", tags=["Agent Registration"])

# Initialize the bridge
mcp_bridge = MCPBridge()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tools_executed: List[str] = []

class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration"""
    id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    endpoint: str
    metadata: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    """Response model for agent information"""
    id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    endpoint: str
    metadata: Dict[str, Any]

class RouteRequestRequest(BaseModel):
    """Request model for routing a request to an agent"""
    request: str
    context: Optional[Dict[str, Any]] = None

class RouteRequestResponse(BaseModel):
    """Response model for routed request"""
    agent_id: Optional[str]
    agent_name: Optional[str]
    confidence: Optional[float]
    message: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "config": config.get_safe_config()
    }

@app.post("/api/v1/chat")
async def chat(request: Request):
    """
    Chat endpoint that uses the MCPBridge for intelligent routing
    """
    try:
        # Parse request body
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Use the bridge to determine routing
        routing = await mcp_bridge.route_request(message, {"session_id": session_id})
        
        # Process the request based on routing information
        response = await process_with_routing(message, routing, session_id)
        
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_with_routing(message: str, routing: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Process a message based on routing information"""
    endpoints = routing["endpoints"]
    results = []
    
    for endpoint in endpoints:
        if endpoint["type"] == "tool":
            # Execute a tool
            tool_name = endpoint["name"]
            tool_params = endpoint["params"]
            tool_params["message"] = message  # Add the original message
            
            # Call the tool with the appropriate namespace if applicable
            if ":" in tool_name:
                namespace, name = tool_name.split(":", 1)
                result = await execute_tool(name, tool_params, namespace)
            else:
                result = await execute_tool(tool_name, tool_params)
                
            results.append(result)
    
    # Combine results and generate final response
    return {
        "response": combine_results(results, routing["intent"]),
        "tools_executed": [ep["name"] for ep in endpoints if ep["type"] == "tool"],
        "intent": routing["intent"],
        "confidence": routing["confidence"]
    }

@app.post(f"{config.API_PREFIX}/register")
async def register_tool():
    """
    Register a new tool with the MCP system
    """
    # TODO: Implement tool registration
    return {"status": "not implemented"}

@app.post(f"{config.API_PREFIX}/tool/{{tool_name}}")
async def execute_tool(tool_name: str):
    """
    Execute a specific tool
    """
    # TODO: Implement tool execution
    return {"status": "not implemented"}

@app.get(f"{config.API_PREFIX}/config")
async def get_config():
    """
    Get the current configuration (excluding sensitive information)
    """
    return config.get_safe_config()

@app.post("/agents", response_model=AgentResponse)
async def register_agent(request: AgentRegistrationRequest):
    """Register a new agent"""
    try:
        agent = Agent(**request.dict())
        agent_registry.register_agent(agent)
        return agent
    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent"""
    try:
        agent_registry.unregister_agent(agent_id)
        return {"message": f"Agent {agent_id} unregistered successfully"}
    except Exception as e:
        logger.error(f"Error unregistering agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents():
    """List all registered agents"""
    try:
        return agent_registry.list_agents()
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get information about a specific agent"""
    try:
        agent = agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route", response_model=RouteRequestResponse)
async def route_request(request: RouteRequestRequest):
    """Route a request to the most appropriate agent"""
    try:
        agent = await agent_registry.route_request(request.request, request.context)
        if agent:
            return RouteRequestResponse(
                agent_id=agent.id,
                agent_name=agent.name,
                confidence=0.8,  # TODO: Get actual confidence from Cohere
                message=f"Request routed to agent {agent.name}"
            )
        else:
            return RouteRequestResponse(
                message="No suitable agent found for the request"
            )
    except Exception as e:
        logger.error(f"Error routing request: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 