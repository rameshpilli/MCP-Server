# # app/main.py
# from fastapi import FastAPI, HTTPException, Request, Response, status
# from fastapi.middleware.cors import CORSMiddleware
# import logging
# import os
# from typing import Dict, Any, List, Optional
# from pydantic import BaseModel
#
# from .config import Config, config
#
# # Setup proper logging for container environments
# logger = logging.getLogger('mcp_app')
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#
# # Always log to stdout for container environments
# handler = logging.StreamHandler()
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# # Create FastAPI application with detailed documentation
# app = FastAPI(
#     title="MCP API Server",
#     description="API Server for Model Context Protocol (MCP)",
#     version="0.1.0",
#     docs_url="/docs",
#     redoc_url="/redoc",
#     openapi_url="/openapi.json"
# )
#
# # Add CORS middleware with configurable origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=config.CORS_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Include agent registration router
# from .agent.registration import router as agent_router
# app.include_router(agent_router, prefix="/api/v1", tags=["Agent Registration"])
#
# # Define request and response models for better API documentation
# class ChatRequest(BaseModel):
#     message: str
#     session_id: Optional[str] = None
#
#     class Config:
#         schema_extra = {
#             "example": {
#                 "message": "What documents do you have available?",
#                 "session_id": "user-123456"
#             }
#         }
#
# class ChatResponse(BaseModel):
#     response: str
#     tools_executed: List[str] = []
#     intent: Optional[str] = None
#     processing_time_ms: Optional[float] = None
#
# class HealthResponse(BaseModel):
#     status: str
#     config: Dict[str, Any]
#     kubernetes: Optional[Dict[str, Any]] = None
#     components: Dict[str, str] = {}
#
# # Add Kubernetes-specific health probes
# @app.get("/livez", status_code=200, tags=["Kubernetes"])
# async def liveness_probe():
#     """
#     Kubernetes liveness probe endpoint
#
#     This endpoint lets Kubernetes know if the application is alive.
#     A successful response means the server is running.
#     """
#     return {"status": "alive"}
#
# @app.get("/readyz", status_code=200, tags=["Kubernetes"])
# async def readiness_probe():
#     """
#     Kubernetes readiness probe endpoint
#
#     This endpoint lets Kubernetes know if the application is ready to receive traffic.
#     Checks if the MCP server and other dependencies are available.
#     """
#     # Import here to avoid circular imports
#     from .client import mcp_client
#
#     try:
#         # Check if we can connect to MCP server
#         mcp_server_port = await mcp_client._discover_mcp_server_port()
#         if not mcp_server_port:
#             # If MCP server is not available, return 503 Service Unavailable
#             return Response(
#                 content="MCP server not available",
#                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE
#             )
#
#         # You could add more dependency checks here (Redis, Cohere, etc.)
#
#         return {"status": "ready", "mcp_server_port": mcp_server_port}
#     except Exception as e:
#         logger.error(f"Readiness check failed: {e}")
#         return Response(
#             content=f"Readiness check failed: {str(e)}",
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE
#         )
#
# @app.get("/", tags=["Information"])
# async def root():
#     """Root endpoint with basic server information"""
#     return {
#         "message": "MCP API Server",
#         "version": "0.1.0",
#         "docs_url": "/docs",
#         "kubernetes": config.IN_KUBERNETES
#     }
#
# @app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
# async def health_check():
#     """
#     Health check endpoint
#
#     Returns detailed information about the server's health and configuration.
#     """
#     # Basic health information
#     health_info = {
#         "status": "ok",
#         "config": config.get_safe_config(),
#         "components": {}
#     }
#
#     # Add Kubernetes-specific information if running in Kubernetes
#     if config.IN_KUBERNETES:
#         health_info["kubernetes"] = {
#             "pod_name": config.POD_NAME,
#             "namespace": config.NAMESPACE,
#             "node": os.getenv("NODE_NAME", "unknown")
#         }
#
#     # Check MCP server connection
#     try:
#         from .client import mcp_client
#         mcp_server_port = await mcp_client._discover_mcp_server_port()
#         health_info["components"]["mcp_server"] = "connected" if mcp_server_port else "disconnected"
#     except Exception as e:
#         health_info["components"]["mcp_server"] = f"error: {str(e)}"
#
#     # Check other components as needed
#     # For example, check Redis connection if you're using it
#     if hasattr(config, "REDIS_HOST") and config.REDIS_HOST:
#         try:
#             import redis
#             r = redis.Redis(
#                 host=config.REDIS_HOST,
#                 port=config.REDIS_PORT,
#                 password=config.REDIS_PASSWORD,
#                 db=config.REDIS_DB
#             )
#             if r.ping():
#                 health_info["components"]["redis"] = "connected"
#             else:
#                 health_info["components"]["redis"] = "disconnected"
#         except Exception as e:
#             health_info["components"]["redis"] = f"error: {str(e)}"
#
#     return health_info
#
# @app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
# async def chat(request: ChatRequest):
#     """
#     Chat endpoint that processes messages using the MCP pipeline
#
#     This endpoint:
#     1. Takes a message and optional session_id
#     2. Routes the request through the MCP bridge
#     3. Executes appropriate tools based on the intent
#     4. Returns a response with the results
#     """
#     try:
#         # Import here to avoid circular imports
#         from .client import mcp_client
#
#         # Add pod name to session_id in Kubernetes for tracing
#         session_id = request.session_id
#         if config.IN_KUBERNETES and session_id:
#             session_id = f"{session_id}-{config.POD_NAME}"
#
#         # Process the message
#         result = await mcp_client.process_message(
#             message=request.message,
#             session_id=session_id
#         )
#
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
# if __name__ == "__main__":
#     import uvicorn
#     # Use 0.0.0.0 in container environments
#     host = "0.0.0.0" if config.IN_KUBERNETES else config.HOST
#     uvicorn.run(app, host=host, port=config.PORT)


from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import logging
import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from .config import Config, config
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
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
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

    class Config:
        schema_extra = {
            "example": {
                "message": "What documents do you have available?",
                "session_id": "user-123456"
            }
        }


class ChatResponse(BaseModel):
    response: str
    tools_executed: List[str] = []
    intent: Optional[str] = None
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    config: Dict[str, Any]
    kubernetes: Optional[Dict[str, Any]] = None
    components: Dict[str, str] = {}


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


# New models for the testing endpoint
class PromptRequest(BaseModel):
    """Request model for testing endpoint"""
    prompt: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}


class PromptResponse(BaseModel):
    """Response model for testing endpoint"""
    response: str
    tools_executed: List[str] = []
    intent: Optional[str] = None
    confidence: Optional[float] = None
    execution_time: float = 0.0


# New models for MCP Inspector support
class ExecuteToolRequest(BaseModel):
    """Request model for executing a tool via MCP Inspector"""
    tool: str
    params: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

    class Config:
        schema_extra = {
            "example": {
                "tool": "search_docs",
                "params": {"query": "security policy"},
                "context": {"session_id": "inspector-123"}
            }
        }


class ExecuteToolResponse(BaseModel):
    """Response model for tool execution via MCP Inspector"""
    result: Any
    error: Optional[str] = None


class ProcessRequest(BaseModel):
    """Request model for processing messages via MCP Inspector"""
    message: str
    session_id: Optional[str] = None
    system_message: Optional[str] = None
    context: Dict[str, Any] = {}

    class Config:
        schema_extra = {
            "example": {
                "message": "What are the top clients?",
                "session_id": "inspector-session",
                "context": {"debug": True}
            }
        }


class ToolInfo(BaseModel):
    """Information about a tool for MCP Inspector"""
    name: str
    description: str
    parameters: Dict[str, Any] = {}


class StreamRequest(BaseModel):
    """Request model for streaming conversations via MCP Inspector"""
    message: str
    context: Dict[str, Any] = {}

    class Config:
        schema_extra = {
            "example": {
                "message": "Show me client trends",
                "context": {"session_id": "stream-session"}
            }
        }


# Add Kubernetes-specific health probes
@app.get("/livez", status_code=200, tags=["Kubernetes"])
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint

    This endpoint lets Kubernetes know if the application is alive.
    A successful response means the server is running.
    """
    return {"status": "alive"}


@app.get("/readyz", status_code=200, tags=["Kubernetes"])
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint

    This endpoint lets Kubernetes know if the application is ready to receive traffic.
    Checks if the MCP server and other dependencies are available.
    """
    # Import here to avoid circular imports
    from .client import mcp_client

    try:
        # Check if we can connect to MCP server
        mcp_server_port = await mcp_client._discover_mcp_server_port()
        if not mcp_server_port:
            # If MCP server is not available, return 503 Service Unavailable
            return Response(
                content="MCP server not available",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # You could add more dependency checks here (Redis, Cohere, etc.)

        return {"status": "ready", "mcp_server_port": mcp_server_port}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return Response(
            content=f"Readiness check failed: {str(e)}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@app.get("/", tags=["Information"])
async def root():
    """Root endpoint with basic server information"""
    return {
        "message": "MCP API Server",
        "version": "0.1.0",
        "docs_url": "/docs",
        "kubernetes": config.IN_KUBERNETES
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns detailed information about the server's health and configuration.
    """
    # Basic health information
    health_info = {
        "status": "ok",
        "config": config.get_safe_config(),
        "components": {}
    }

    # Add Kubernetes-specific information if running in Kubernetes
    if config.IN_KUBERNETES:
        health_info["kubernetes"] = {
            "pod_name": config.POD_NAME,
            "namespace": config.NAMESPACE,
            "node": os.getenv("NODE_NAME", "unknown")
        }

    # Check MCP server connection
    try:
        from .client import mcp_client
        mcp_server_port = await mcp_client._discover_mcp_server_port()
        health_info["components"]["mcp_server"] = "connected" if mcp_server_port else "disconnected"
    except Exception as e:
        health_info["components"]["mcp_server"] = f"error: {str(e)}"

    # Check other components as needed
    # For example, check Redis connection if you're using it
    if hasattr(config, "REDIS_HOST") and config.REDIS_HOST:
        try:
            import redis
            r = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB
            )
            if r.ping():
                health_info["components"]["redis"] = "connected"
            else:
                health_info["components"]["redis"] = "disconnected"
        except Exception as e:
            health_info["components"]["redis"] = f"error: {str(e)}"

    return health_info


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes messages using the MCP pipeline

    This endpoint:
    1. Takes a message and optional session_id
    2. Routes the request through the MCP bridge
    3. Executes appropriate tools based on the intent
    4. Returns a response with the results
    """
    try:
        # Import here to avoid circular imports
        from .client import mcp_client

        # Add pod name to session_id in Kubernetes for tracing
        session_id = request.session_id
        if config.IN_KUBERNETES and session_id:
            session_id = f"{session_id}-{config.POD_NAME}"

        # Process the message
        result = await mcp_client.process_message(
            message=request.message,
            session_id=session_id
        )

        return result

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# MCP Inspector Support Endpoints

@app.get("/meta", tags=["MCP Inspector"])
async def get_meta():
    """
    MCP Protocol metadata endpoint

    Returns information about the MCP server implementation that
    the MCP Inspector uses to understand available capabilities.
    """
    try:
        # Import the bridge here to avoid circular imports
        from .mcp_bridge import MCPBridge
        bridge = MCPBridge()

        # Get available tools
        tools = await bridge.get_available_tools()

        # Format tools for MCP protocol
        formatted_tools = []
        for namespace, tool_dict in tools.items():
            for name, description in tool_dict.items():
                tool_name = f"{namespace}:{name}" if namespace != "default" else name
                formatted_tools.append({
                    "name": tool_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to process"
                            }
                        },
                        "required": ["message"]
                    }
                })

        # Return metadata in MCP protocol format
        return {
            "protocol": {
                "name": "model-context-protocol",
                "version": "0.1.0"
            },
            "config": {
                "execution": {
                    "concurrency": 1,
                    "tool_timeoutMs": 30000
                },
                "streaming": {
                    "enabled": True
                }
            },
            "tools": formatted_tools,
            "info": {
                "name": config.SERVER_NAME,
                "description": config.SERVER_DESCRIPTION
            }
        }
    except Exception as e:
        logger.error(f"Error getting meta information: {e}")
        return Response(
            content=f"Error getting meta information: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/tools", tags=["MCP Inspector"])
async def get_tools():
    """
    Get all available tools with their descriptions and parameters

    This endpoint follows the MCP protocol format expected by the MCP Inspector.
    """
    try:
        # Import the bridge here to avoid circular imports
        from .mcp_bridge import MCPBridge
        bridge = MCPBridge()

        # Get available tools and format them
        tools = await bridge.get_available_tools()

        # Same formatting as in the meta endpoint
        formatted_tools = []
        for namespace, tool_dict in tools.items():
            for name, description in tool_dict.items():
                tool_name = f"{namespace}:{name}" if namespace != "default" else name
                formatted_tools.append({
                    "name": tool_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to process"
                            }
                        },
                        "required": ["message"]
                    }
                })

        return {"tools": formatted_tools}
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        return Response(
            content=f"Error getting tools: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.post("/execute", tags=["MCP Inspector"])
async def execute_tool_endpoint(request: ExecuteToolRequest):
    """
    Execute a tool directly by name

    This endpoint allows the MCP Inspector to execute tools directly with
    specific parameters, without going through the natural language processing.
    """
    try:
        # Import the bridge here to avoid circular imports
        from .mcp_bridge import MCPBridge
        bridge = MCPBridge()

        # Log the request for debugging
        logger.info(f"MCP Inspector executing tool: {request.tool}")

        # Handle namespaced tools
        tool_name = request.tool

        # Execute the tool through the bridge
        result = await bridge.execute_tool(tool_name, request.params, request.context)

        # Return result in standard format
        return {"result": result, "error": None}
    except Exception as e:
        logger.error(f"Error executing tool {request.tool}: {e}")
        return {"result": None, "error": str(e)}


@app.post("/stream", tags=["MCP Inspector"])
async def stream_conversation(request: StreamRequest):
    """
    Stream a conversation with the MCP server

    This endpoint implements the SSE protocol that MCP Inspector expects
    for streaming interactions.
    """

    async def generate():
        try:
            # Import here to avoid circular imports
            from .client import mcp_client
            from .mcp_bridge import MCPBridge
            bridge = MCPBridge()

            # Start time for measuring performance
            start_time = time.time()

            # Route the message to get intent and tools to execute
            routing = await bridge.route_request(request.message, request.context)

            # Start streaming with thinking indicator
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Processing your request...'})}\n\n"

            # Execute the tools based on routing
            tools_executed = []
            for endpoint in routing.get('endpoints', []):
                if endpoint['type'] == 'tool':
                    tool_name = endpoint['name']
                    params = endpoint['params']

                    try:
                        # Stream that we're executing a tool
                        yield f"data: {json.dumps({'type': 'tool-start', 'name': tool_name, 'params': params})}\n\n"

                        # Execute the tool
                        result = await bridge.execute_tool(tool_name, params, request.context)

                        # Stream the tool result
                        tool_event = {
                            "type": "tool",
                            "name": tool_name,
                            "params": params,
                            "result": result
                        }
                        yield f"data: {json.dumps(tool_event)}\n\n"

                        # Track executed tools for later use
                        tools_executed.append({
                            "name": tool_name,
                            "params": params,
                            "result": result
                        })

                    except Exception as tool_error:
                        # Stream tool error
                        yield f"data: {json.dumps({'type': 'tool-error', 'name': tool_name, 'error': str(tool_error)})}\n\n"

            # Generate the final response
            # For this example, we'll just combine all tool results
            if tools_executed:
                response_text = "Here are the results:\n\n"
                for i, tool in enumerate(tools_executed, 1):
                    response_text += f"{i}. {tool['name']}: {tool['result']}\n"
            else:
                response_text = "I processed your request but didn't execute any tools."

            # Send the final text response
            yield f"data: {json.dumps({'type': 'text', 'content': response_text})}\n\n"

            # Send performance metrics
            processing_time = time.time() - start_time
            yield f"data: {json.dumps({'type': 'metrics', 'processing_time_ms': processing_time * 1000})}\n\n"

            # End the stream
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream: {e}")
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/mcp", tags=["MCP Protocol"])
async def mcp_endpoint(request: Request):
    """
    MCP Protocol endpoint for Chainlit integration

    This endpoint implements the core MCP protocol that Chainlit expects.
    """
    try:
        # Parse the request body
        body = await request.json()
        message = body.get("message", "")
        context = body.get("context", {})

        # Import the bridge
        from .mcp_bridge import MCPBridge
        bridge = MCPBridge()

        # Process the message
        routing = await bridge.route_request(message, context)

        # Execute tools based on routing
        results = []
        for endpoint in routing.get('endpoints', []):
            if endpoint['type'] == 'tool':
                tool_name = endpoint['name']
                params = endpoint['params']

                try:
                    result = await bridge.execute_tool(tool_name, params, context)
                    results.append({
                        "name": tool_name,
                        "params": params,
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    results.append({
                        "name": tool_name,
                        "params": params,
                        "error": str(e)
                    })

        # Format the response
        response = {
            "response": results[0]["result"] if results else "No tools were executed",
            "tools": [r["name"] for r in results],
            "results": results
        }

        return response
    except Exception as e:
        logger.error(f"Error in MCP endpoint: {e}")
        return Response(
            content=f"Error processing MCP request: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

if __name__ == "__main__":
    import uvicorn

    # Use 0.0.0.0 in container environments
    host = "0.0.0.0" if config.IN_KUBERNETES else config.HOST
    uvicorn.run(app, host=host, port=config.PORT)
