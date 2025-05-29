"""
Streamlined MCP Server with enhanced tool discovery, proper table formatting,
mock endpoints, and comprehensive Swagger documentation.

This server handles both stdio and SSE modes, with automatic tool registration
and proper chaining between tools.

Usage:
    - Run in HTTP/SSE mode: python -m app.streamlined_mcp_server --mode http
    - Run in stdio mode: python -m app.streamlined_mcp_server --mode stdio
    - Install with: uvx install
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import inspect
import time
import pandas as pd
from pathlib import Path
from fastmcp import FastMCP, Context
from fastapi import FastAPI, Request, Response, status, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import Dict, Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Set up paths
sys.path.append(str(Path(__file__).parent.parent))
from app.config import config
from app.stdio_handler import run_stdio_mode
from app.sse_handler import sse_endpoint

# Environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("mcp_server")
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# File handler for persistent logs
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / f"mcp_server_{time.strftime('%Y%m%d')}.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# CSV logger for interactions
interaction_logger = logging.getLogger("mcp_interactions")
interaction_logger.setLevel(logging.INFO)
csv_handler = logging.FileHandler(log_dir / f"mcp_interactions_{time.strftime('%Y%m%d')}.csv")
interaction_logger.addHandler(csv_handler)
if not interaction_logger.hasHandlers():
    interaction_logger.info("timestamp,session_id,message,tools_executed,processing_time_ms")

# MCP server
mcp = FastMCP(
    config.SERVER_NAME,
    description=config.SERVER_DESCRIPTION
)

# Pydantic models for API documentation
class ChatRequest(BaseModel):
    message: str = Field(..., description="The message to process")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Show me the top clients",
                "session_id": "user-123456"
            }
        }

class ChatResponse(BaseModel):
    response: str = Field(..., description="The processed response")
    tools_executed: List[str] = Field(default=[], description="List of tools that were executed")
    intent: Optional[str] = Field(None, description="Detected intent of the request")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default={}, description="Parameters for the tool")
    context: Dict[str, Any] = Field(default={}, description="Context for tool execution")
    
    class Config:
        schema_extra = {
            "example": {
                "tool_name": "getTopClients",
                "parameters": {"region": "USA", "currency": "USD", "limit": 10},
                "context": {"session_id": "user-123456"}
            }
        }

class ToolExecutionResponse(BaseModel):
    result: Any = Field(..., description="Result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class ToolInfo(BaseModel):
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    parameters: Dict[str, Any] = Field(default={}, description="Parameters schema")
    namespace: str = Field(default="default", description="Namespace of the tool")

class ToolListResponse(BaseModel):
    tools: List[ToolInfo] = Field(..., description="List of available tools")

class StreamRequest(BaseModel):
    message: str = Field(..., description="The message to process")
    context: Dict[str, Any] = Field(default={}, description="Context for processing")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Show me client trends",
                "context": {"session_id": "stream-session"}
            }
        }

class MockEndpointRequest(BaseModel):
    appCode: str = Field(..., description="Application code")
    values: List[Any] = Field(default=[], description="Values for the request")
    
    class Config:
        schema_extra = {
            "example": {
                "appCode": "client1",
                "values": []
            }
        }

# Register basic tools
@mcp.tool()
async def health_check(ctx: Context) -> str:
    """Health check endpoint"""
    return "MCP Server is healthy"

@mcp.tool()
async def server_info(ctx: Context) -> str:
    """Get server information"""
    tools = await mcp.get_tools()
    
    result = f"""
# MCP Server Information
**Host**: {config.MCP_SERVER_HOST}
**Port**: {config.MCP_SERVER_PORT}
**Version**: {config.VERSION}
**Environment**: {config.ENVIRONMENT}

## Available Tools
"""
    
    for name, tool in tools.items():
        result += f"- **{name}**: {getattr(tool, 'description', 'No description')}\n"
    
    return result

@mcp.tool()
async def list_tools(ctx: Context) -> str:
    """List all available tools and their schemas"""
    try:
        tools = await mcp.get_tools()
        result = "# Available Tools\n\n"

        for name, tool in tools.items():
            result += f"## {name}\n"
            result += f"Description: {getattr(tool, 'description', 'No description')}\n\n"

            # Try to get schema
            schema = None
            for attr in ['input_schema', 'schema', 'parameters']:
                if hasattr(tool, attr):
                    schema = getattr(tool, attr)
                    break

            if schema:
                result += f"Parameters:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            else:
                result += "No parameter schema available.\n\n"

        return result
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return f"Error listing tools: {str(e)}"

# Tool for formatting table outputs
@mcp.tool()
async def format_table(ctx: Context, data: List[Dict], format: str = "markdown") -> str:
    """
    Format a data table in various output formats
    
    Args:
        data: List of dictionaries containing the table data
        format: Output format (markdown, html, csv)
    
    Returns:
        Formatted table string
    """
    if not data:
        return "No data available"
    
    try:
        df = pd.DataFrame(data)
        
        if format.lower() == "html":
            return df.to_html(index=False)
        elif format.lower() == "csv":
            return df.to_csv(index=False)
        elif format.lower() == "json":
            return json.dumps(data, indent=2)
        else:  # Default to markdown
            return df.to_markdown(index=False)
    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return f"Error formatting table: {str(e)}"

# Mock financial tools
@mcp.tool()
async def getTopClients(ctx: Context, region: str = None, currency: str = "USD", limit: int = 10) -> str:
    """
    Get top clients by revenue
    
    Args:
        region: Filter by region (USA, CAN, EUR, APAC, LATAM)
        currency: Currency for revenue values (USD, CAD, EUR)
        limit: Maximum number of clients to return
    
    Returns:
        Table of top clients
    """
    try:
        # Generate mock data
        import random
        random.seed(42)  # For consistent results
        
        regions = ["USA", "CAN", "EUR", "APAC", "LATAM"]
        if region and region not in regions:
            return f"Invalid region: {region}. Valid regions are: {', '.join(regions)}"
        
        clients = []
        for i in range(20):
            client_region = random.choice(regions)
            if region and region != client_region:
                continue
                
            revenue = round(random.uniform(1_000_000, 10_000_000), 2)
            
            # Apply currency conversion if needed
            if currency == "CAD":
                revenue *= 1.35
            elif currency == "EUR":
                revenue *= 0.93
                
            clients.append({
                "ClientName": f"Client {i+1}",
                "ClientID": 1000 + i,
                "Region": client_region,
                "Revenue": revenue,
                "Currency": currency,
                "YoYGrowth": f"{random.uniform(-15, 30):.1f}%"
            })
        
        # Sort by revenue and limit results
        clients.sort(key=lambda x: x["Revenue"], reverse=True)
        clients = clients[:limit]
        
        # Format as markdown table
        df = pd.DataFrame(clients)
        
        # Format currency values
        df["Revenue"] = df["Revenue"].apply(lambda x: f"{x:,.2f}")
        
        return df.to_markdown(index=False)
    except Exception as e:
        logger.error(f"Error in getTopClients: {e}")
        return f"Error retrieving top clients: {str(e)}"

@mcp.tool()
async def getClientRevenue(ctx: Context, client_id: int = None, client_name: str = None, 
                          time_period: str = "YTD") -> str:
    """
    Get revenue details for a specific client
    
    Args:
        client_id: Client ID number
        client_name: Client name (alternative to client_id)
        time_period: Time period for data (YTD, QTD, MTD, PrevYTD)
    
    Returns:
        Revenue details for the specified client
    """
    try:
        if not client_id and not client_name:
            return "Error: Either client_id or client_name must be provided"
            
        # Generate mock data
        import random
        random.seed(42)  # For consistent results
        
        # Find client
        client_found = False
        client_info = {}
        
        for i in range(20):
            current_id = 1000 + i
            current_name = f"Client {i+1}"
            
            if (client_id and current_id == client_id) or (client_name and current_name == client_name):
                client_found = True
                client_info = {
                    "ClientName": current_name,
                    "ClientID": current_id,
                    "Region": random.choice(["USA", "CAN", "EUR", "APAC", "LATAM"])
                }
                break
                
        if not client_found:
            return f"Client not found with ID {client_id}" if client_id else f"Client not found with name {client_name}"
            
        # Generate revenue data
        products = ["Bonds", "Equities", "FX", "Derivatives", "Commodities"]
        revenue_data = []
        
        for product in products:
            base_revenue = random.uniform(100_000, 2_000_000)
            
            # Adjust based on time period
            if time_period == "QTD":
                base_revenue *= 0.25
            elif time_period == "MTD":
                base_revenue *= 0.08
            elif time_period == "PrevYTD":
                base_revenue *= random.uniform(0.8, 1.2)
                
            revenue_data.append({
                "Product": product,
                "Revenue": round(base_revenue, 2),
                "YoYChange": f"{random.uniform(-20, 40):.1f}%",
                "TimePeriod": time_period
            })
            
        # Calculate total
        total_revenue = sum(item["Revenue"] for item in revenue_data)
        revenue_data.append({
            "Product": "TOTAL",
            "Revenue": round(total_revenue, 2),
            "YoYChange": f"{random.uniform(-10, 30):.1f}%",
            "TimePeriod": time_period
        })
        
        # Format response
        result = f"# Revenue for {client_info['ClientName']} ({time_period})\n\n"
        result += f"**Client ID:** {client_info['ClientID']}\n"
        result += f"**Region:** {client_info['Region']}\n\n"
        
        # Format as markdown table
        df = pd.DataFrame(revenue_data)
        
        # Format currency values
        df["Revenue"] = df["Revenue"].apply(lambda x: f"${x:,.2f}")
        
        result += df.to_markdown(index=False)
        return result
    except Exception as e:
        logger.error(f"Error in getClientRevenue: {e}")
        return f"Error retrieving client revenue: {str(e)}"

# Auto-discover and register tools
def autodiscover_tools():
    """Automatically discover and register tools from the tools directory"""
    try:
        tools_dir = Path(__file__).parent / "tools"
        if not tools_dir.exists():
            logger.warning(f"Tools directory not found: {tools_dir}")
            return
            
        logger.info(f"Discovering tools in: {tools_dir}")
        
        # Import all modules in the tools directory
        for file_path in tools_dir.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
                
            module_name = f"app.tools.{file_path.stem}"
            try:
                logger.info(f"Importing module: {module_name}")
                module = __import__(module_name, fromlist=["*"])
                
                # Look for register_tools function
                if hasattr(module, "register_tools"):
                    logger.info(f"Registering tools from {module_name}")
                    module.register_tools(mcp)
                    
                # Look for tool functions
                for name, obj in inspect.getmembers(module):
                    if hasattr(obj, "__mcp_tool__") and obj.__mcp_tool__ is True:
                        logger.info(f"Found MCP tool: {name}")
                        mcp.register_tool(obj)
            except Exception as e:
                logger.error(f"Error importing module {module_name}: {e}")
    except Exception as e:
        logger.error(f"Error discovering tools: {e}")

# Call autodiscover
try:
    autodiscover_tools()
except Exception as e:
    logger.error(f"Tool auto-discovery failed: {e}")

# Setup the bridge
try:
    try:
        from app.langchain_bridge import LangChainBridge
        bridge = LangChainBridge(mcp)  # Pass mcp instance explicitly
        logger.info("LangChain Bridge initialized successfully")
    except ImportError:
        from app.mcp_bridge import MCPBridge
        bridge = MCPBridge(mcp)  # Pass mcp instance explicitly
        logger.info("MCP Bridge initialized successfully")
except Exception as e:
    logger.error(f"Bridge initialization failed: {e}")
    bridge = None

# Process message using the bridge
async def process_message(message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a message using the Bridge"""
    start_time = time.time()
    session_id = context.get("session_id", "unknown") if context else "unknown"
    
    try:
        logger.info(f"Processing message: {message}")
        
        # Route the message using the bridge
        routing_result = await bridge.route_request(message, context or {})
        logger.info(f"Routing result: {routing_result}")
        
        if not routing_result.get("endpoints"):
            logger.info("No tool selected — routing directly to LLM for natural language response.")
            from app.client import mcp_client
            
            try:
                llm_response = await mcp_client.call_llm([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message}
                ])
                
                if isinstance(llm_response, dict) and "choices" in llm_response:
                    response_text = llm_response["choices"][0]["message"]["content"]
                else:
                    response_text = str(llm_response)
                    
                processing_time = (time.time() - start_time) * 1000
                
                # Log the interaction
                interaction_logger.info(f"{time.time()},{session_id},{message},{{}},{processing_time:.2f}")
                
                return {
                    "response": response_text,
                    "tools_executed": [],
                    "intent": None,
                    "processing_time_ms": processing_time
                }
            except Exception as e:
                logger.error(f"LLM fallback failed: {e}")
                return {
                    "response": "I'm sorry, I couldn't process that request at the moment.",
                    "tools_executed": [],
                    "intent": None,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
        
        # Execute tools from endpoints
        results = []
        tools_executed = []
        
        for endpoint in routing_result["endpoints"]:
            if endpoint["type"] == "tool":
                tool_name = endpoint["name"]
                params = endpoint.get("params", {})
                tools_executed.append(tool_name)
                
                logger.info(f"Executing tool: {tool_name} with params: {params}")
                
                # Get tools
                tools = await mcp.get_tools()
                if tool_name in tools:
                    # Create context
                    ctx = Context(context or {})
                    
                    try:
                        # Get the tool function
                        tool_func = tools[tool_name].fn
                        
                        # Try calling with parameters only (context is injected by decorator)
                        try:
                            if params:
                                result = await tool_func(**params)
                            else:
                                result = await tool_func()
                        except TypeError:
                            # If TypeError, try calling with ctx as first parameter
                            if params:
                                result = await tool_func(ctx, **params)
                            else:
                                result = await tool_func(ctx)
                                
                        results.append(result)
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        results.append(error_msg)
                else:
                    logger.warning(f"Tool not found: {tool_name}")
                    results.append(f"Tool not found: {tool_name}")
        
        # Format response based on results
        if results:
            # Process results - check if any result is a dict (structured response)
            processed_results = []
            for result in results:
                if isinstance(result, dict) and "output" in result:
                    # Extract the output field from structured responses
                    processed_results.append(result["output"])
                else:
                    # Use the result as-is
                    processed_results.append(str(result))
            
            if len(processed_results) == 1:
                # Just return the single result directly
                response_text = processed_results[0]
            else:
                # Format multiple results
                response_parts = ["Multiple tool results:"]
                for i, result in enumerate(processed_results, 1):
                    response_parts.append(f"{i}. {result}")
                response_text = "\n\n".join(response_parts)
        else:
            response_text = "I couldn't find any tools to execute for your request."
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log the interaction
        interaction_logger.info(f"{time.time()},{session_id},{message},{','.join(tools_executed)},{processing_time:.2f}")
        
        return {
            "response": response_text,
            "tools_executed": tools_executed,
            "intent": routing_result.get("intent"),
            "processing_time_ms": processing_time
        }
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        processing_time = (time.time() - start_time) * 1000
        
        # Log the error
        interaction_logger.info(f"{time.time()},{session_id},{message},error,{processing_time:.2f}")
        
        return {
            "response": f"Error processing message: {str(e)}",
            "tools_executed": [],
            "intent": None,
            "processing_time_ms": processing_time
        }

def main_factory():
    """Create and return the FastAPI application"""
    app = FastAPI(
        title="Streamlined MCP API Server",
        description="Enhanced MCP Server with tool discovery, table formatting, and comprehensive API documentation",
        version=config.VERSION,
        docs_url=None,  # We'll create a custom docs endpoint
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom Swagger UI with enhanced styling
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - API Documentation",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
            swagger_ui_parameters={"defaultModelsExpandDepth": -1}
        )
    
    # Customize OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
            
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add security schemes if needed
        # openapi_schema["components"]["securitySchemes"] = {...}
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
        
    app.openapi = custom_openapi
    
    # Basic endpoints
    @app.get("/ping", tags=["Health"])
    async def ping():
        """Simple health check endpoint"""
        return {"status": "ok", "server": "MCP Server", "version": config.VERSION}
    
    @app.get("/", tags=["Information"])
    async def root():
        """Root endpoint with basic server information"""
        return {
            "message": "Streamlined MCP API Server",
            "version": config.VERSION,
            "docs_url": "/docs",
            "swagger_url": "/docs",
            "environment": config.ENVIRONMENT
        }
    
    # Main MCP endpoint
    @app.post("/mcp", response_model=ChatResponse, tags=["Chat"])
    async def handle_mcp_request(request: ChatRequest):
        """
        Process a message using the MCP pipeline
        
        This endpoint:
        1. Takes a message and optional session_id
        2. Routes the request through the MCP bridge
        3. Executes appropriate tools based on the intent
        4. Returns a response with the results
        """
        try:
            context = {"session_id": request.session_id} if request.session_id else {}
            
            result = await process_message(request.message, context)
            return result
        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return {
                "response": f"Error processing request: {str(e)}",
                "tools_executed": [],
                "intent": None,
                "processing_time_ms": None
            }
    
    # Tool execution endpoint
    @app.post("/execute", response_model=ToolExecutionResponse, tags=["Tools"])
    async def execute_tool(request: ToolExecutionRequest):
        """
        Execute a specific tool directly
        
        This endpoint allows direct execution of a tool with specific parameters,
        bypassing the natural language processing pipeline.
        """
        start_time = time.time()
        try:
            # Get available tools
            tools = await mcp.get_tools()
            
            if request.tool_name not in tools:
                return {
                    "result": None,
                    "error": f"Tool not found: {request.tool_name}",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            
            # Create context
            ctx = Context(request.context)
            
            # Get the tool function
            tool_func = tools[request.tool_name].fn
            
            # Try calling with parameters only first
            try:
                if request.parameters:
                    result = await tool_func(**request.parameters)
                else:
                    result = await tool_func()
            except TypeError:
                # If TypeError, try calling with ctx as first parameter
                if request.parameters:
                    result = await tool_func(ctx, **request.parameters)
                else:
                    result = await tool_func(ctx)
            
            # Process structured responses
            if isinstance(result, dict) and "output" in result:
                result = result["output"]
                
            return {
                "result": result,
                "error": None,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return {
                "result": None,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    # List available tools
    @app.get("/tools", response_model=ToolListResponse, tags=["Tools"])
    async def list_available_tools():
        """
        List all available tools with their descriptions and parameter schemas
        """
        try:
            tools = await mcp.get_tools()
            
            tool_list = []
            for name, tool in tools.items():
                # Get input schema
                input_schema = getattr(tool, 'input_schema', {})
                
                # Handle possible function call schema format
                if not input_schema and hasattr(tool, 'fn'):
                    # Try to extract parameter info from function
                    import inspect
                    sig = inspect.signature(tool.fn)
                    input_schema = {
                        "type": "object",
                        "properties": {}
                    }
                    
                    # Skip first parameter (usually ctx/context)
                    params = list(sig.parameters.items())
                    if params and params[0][0] in ('ctx', 'context'):
                        params = params[1:]  # Skip first parameter
                    
                    for param_name, param in params:
                        param_info = {
                            "type": "string"  # Default to string
                        }
                        # Get annotation if available
                        if param.annotation != inspect.Parameter.empty:
                            if param.annotation == str:
                                param_info["type"] = "string"
                            elif param.annotation == int:
                                param_info["type"] = "integer"
                            elif param.annotation == float:
                                param_info["type"] = "number"
                            elif param.annotation == bool:
                                param_info["type"] = "boolean"
                        
                        # Get default if available
                        if param.default != inspect.Parameter.empty:
                            param_info["default"] = param.default
                        
                        input_schema["properties"][param_name] = param_info
                
                # Get namespace
                namespace = getattr(tool, 'namespace', 'default')
                
                tool_list.append({
                    "name": name,
                    "description": getattr(tool, 'description', 'No description'),
                    "parameters": input_schema,
                    "namespace": namespace
                })
            
            return {"tools": tool_list}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")
    
    # SSE streaming endpoint
    @app.get("/sse", tags=["Streaming"])
    async def handle_sse(request: Request):
        """
        Server-Sent Events (SSE) endpoint for streaming conversations
        
        This endpoint implements the SSE protocol for streaming interactions
        with the MCP server.
        """
        return await sse_endpoint(request, process_message, get_tools_func=mcp.get_tools)
    
    # Streaming endpoint
    @app.post("/stream", tags=["Streaming"])
    async def stream_conversation(request: StreamRequest):
        """
        Stream a conversation with the MCP server
        
        This endpoint implements the SSE protocol for streaming interactions
        with the MCP server based on a POST request.
        """
        async def generate():
            try:
                # Start time for measuring performance
                start_time = time.time()
                
                # Route the message
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
                            tools = await mcp.get_tools()
                            if tool_name in tools:
                                ctx = Context(request.context)
                                tool_func = tools[tool_name].fn
                                
                                try:
                                    if params:
                                        result = await tool_func(**params)
                                    else:
                                        result = await tool_func()
                                except TypeError:
                                    if params:
                                        result = await tool_func(ctx, **params)
                                    else:
                                        result = await tool_func(ctx)
                                
                                # Process structured responses
                                if isinstance(result, dict) and "output" in result:
                                    result = result["output"]
                                
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
                            else:
                                # Stream tool not found error
                                yield f"data: {json.dumps({'type': 'tool-error', 'name': tool_name, 'error': 'Tool not found'})}\n\n"
                        except Exception as tool_error:
                            # Stream tool error
                            yield f"data: {json.dumps({'type': 'tool-error', 'name': tool_name, 'error': str(tool_error)})}\n\n"
                
                # Generate the final response
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
    
    # Mock financial endpoints
    @app.post("/mock/getTopClients", tags=["Mock Endpoints"])
    async def mock_get_top_clients(request: MockEndpointRequest):
        """
        Mock endpoint for getTopClients financial data
        
        This endpoint simulates the response from a financial data service.
        """
        import random
        random.seed(42)  # For consistent results
        
        data = []
        for i in range(20):
            data.append({
                "ClientName": f"Client {i+1}",
                "ClientID": 1000 + i,
                "Region": random.choice(["USA", "CAN", "EUR", "APAC", "LATAM"]),
                "Revenue": round(random.uniform(1_000_000, 10_000_000), 2),
                "YoYGrowth": f"{random.uniform(-15, 30):.1f}%"
            })
        
        return {"status": "success", "data": data}
    
    @app.post("/mock/getClientRevenue", tags=["Mock Endpoints"])
    async def mock_get_client_revenue(request: MockEndpointRequest):
        """
        Mock endpoint for getClientRevenue financial data
        
        This endpoint simulates the response from a financial data service.
        """
        import random
        random.seed(42)  # For consistent results
        
        products = ["Bonds", "Equities", "FX", "Derivatives", "Commodities"]
        data = []
        
        for product in products:
            data.append({
                "Product": product,
                "Revenue": round(random.uniform(100_000, 2_000_000), 2),
                "YoYChange": f"{random.uniform(-20, 40):.1f}%",
                "TimePeriod": "YTD"
            })
        
        return {"status": "success", "data": data}
    
    @app.post("/procedure/memsql__client1__getTopClients", tags=["Mock Endpoints"])
    async def mock_procedure_top_clients(request: MockEndpointRequest):
        """Mock financial procedure endpoint for getTopClients"""
        import random
        random.seed(42)
        
        regions = ["CAN", "USA", "EUR", "APAC", "LATAM", "OTHER"]
        focus_lists = ["Focus40", "FS30", "Corp100"]
        
        data = []
        for i in range(150):
            data.append({
                "ClientName": f"Client {i+1}",
                "ClientCDRID": 1000 + i,
                "RevenueYTD": round(random.uniform(1_000_000, 10_000_000), 2),
                "RegionName": random.choice(regions),
                "FocusList": random.choice(focus_lists),
                "InteractionCMOCYTD": random.randint(0, 20),
                "InteractionGMOCYTD": random.randint(0, 20),
                "InteractionYTD": random.randint(0, 40),
                "InteractionCMOCPrevYTD": random.randint(0, 20),
                "InteractionGMOCPrevYTD": random.randint(0, 20),
                "InteractionPrevYTD": random.randint(0, 40),
            })
        
        return {"status": "success", "data": data}
    
    @app.post("/procedure/memsql__client1__getRevenueTotalByTimePeriod", tags=["Mock Endpoints"])
    async def mock_procedure_revenue_by_time(request: MockEndpointRequest):
        """Mock financial procedure endpoint for getRevenueTotalByTimePeriod"""
        import random
        random.seed(42)
        
        data = []
        for i in range(150):
            data.append({
                "ClientName": f"Client {i+1}",
                "ClientCDRID": 1000 + i,
                "RevenueYTD": round(random.uniform(1_000_000, 10_000_000), 2),
                "RevenuePrevYTD": round(random.uniform(500_000, 9_000_000), 2),
                "InteractionCMOCYTD": random.randint(0, 20),
                "InteractionGMOCYTD": random.randint(0, 20),
                "InteractionYTD": random.randint(0, 40),
                "TimePeriodList": [2023, 2024, 2025],
                "TimePeriodCategory": random.choice(["FY", "CY"]),
            })
        
        return {"status": "success", "data": data}
    
    @app.post("/procedure/memsql__client1__getClientValueRevenueByProduct", tags=["Mock Endpoints"])
    async def mock_procedure_client_value_by_product(request: MockEndpointRequest):
        """Mock financial procedure endpoint for getClientValueRevenueByProduct"""
        import random
        random.seed(42)
        
        products = ["Bonds", "Equities", "FX", "Derivatives", "Commodities"]
        
        data = []
        for i in range(150):
            data.append({
                "ProductName": random.choice(products),
                "RevenueYTD": round(random.uniform(500_000, 5_000_000), 2),
                "RevenuePrevYTD": round(random.uniform(500_000, 5_000_000), 2),
                "ProductID": 2000 + i,
                "ProductHierarchyDepth": random.randint(1, 3),
                "ParentProductID": random.randint(1000, 1999),
                "TimePeriodList": [2023, 2024, 2025],
            })
        
        return {"status": "success", "data": data}
    
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined MCP Server")
    parser.add_argument(
        "--mode", "--server_type",
        choices=["http", "stdio"],
        default="http",
        help="Server mode: 'http' for HTTP/SSE or 'stdio' for STDIO transport"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to"
    )
    args = parser.parse_args()
    
    if args.mode == "stdio":
        # Run in STDIO mode
        logger.info("Starting MCP server in stdio mode")
        asyncio.run(run_stdio_mode(mcp, process_message))
    else:
        # Run in HTTP/SSE mode
        import uvicorn
        logger.info(f"Starting MCP server in HTTP mode on {args.host}:{args.port}")
        uvicorn.run(main_factory(), host=args.host, port=args.port)

# App instance for ASGI servers
app = main_factory()
