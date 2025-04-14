"""MCP (Model Context Protocol) server integration."""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from fastapi import Depends, APIRouter

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from mcp.database import get_session, models
from mcp.database.models import Model as MCPModel, DataSource as MCPDataSource
from mcp.storage import create_storage_backend
from mcp.storage.s3 import S3Storage
from mcp.storage.azure import AzureStorage
from mcp.storage.local import LocalStorage
from mcp.storage.snowflake import SnowflakeStorage

from app.core.config import get_settings
from app.core.logger import logger
from app.core.model_client import ModelClientFactory, ModelClient

# Get settings
settings = get_settings()

# Initialize MCP server
mcp_server = FastMCP("MCP Model Registry")

# Store clients by model ID
model_clients: Dict[str, ModelClient] = {}


async def get_model_client(model_id: str) -> ModelClient:
    """Get or create a model client for the specified model ID."""
    if model_id in model_clients:
        return model_clients[model_id]
    
    # Get model from database using MCP's database functionality
    async with get_session() as session:
        model = await models.get_model(session, model_id)
    
    if not model:
        raise ValueError(f"Model {model_id} not found")
    
    # Create client
    client_factory = ModelClientFactory()
    client = client_factory.create_client_from_mcp_model(model)
    
    # Store for reuse
    model_clients[model_id] = client
    
    return client


# Set up FastAPI router-based implementation for mock
router = APIRouter(prefix="/api/mcp", tags=["MCP"])

# Mapping for resource and tool handlers
resource_handlers = {}
tool_handlers = {}
prompt_handlers = {}


@router.get("/models")
async def list_models_resource():
    """List all registered models as a resource."""
    try:
        # Use MCP's database functionality
        async with get_session() as session:
            models_list = await models.get_models(session)
        
        model_list = []
        for model in models_list:
            model_list.append({
                "model_id": model.id,
                "name": model.name,
                "description": model.description,
                "backend": model.backend,
                "version": model.version,
                "is_active": model.is_active,
                "total_requests": model.metrics.get("total_requests", 0) if model.metrics else 0,
                "successful_requests": model.metrics.get("successful_requests", 0) if model.metrics else 0
            })
        
        return model_list
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return {"error": str(e)}


@router.get("/models/{model_id}")
async def get_model_resource(model_id: str):
    """Get details for a specific model."""
    try:
        # Use MCP's database functionality
        async with get_session() as session:
            model = await models.get_model(session, model_id)
        
        if not model:
            return {"error": f"Model {model_id} not found"}
        
        model_data = {
            "model_id": model.id,
            "name": model.name,
            "description": model.description,
            "backend": model.backend,
            "version": model.version,
            "api_base": model.api_base,
            "is_active": model.is_active,
            "total_requests": model.metrics.get("total_requests", 0) if model.metrics else 0,
            "successful_requests": model.metrics.get("successful_requests", 0) if model.metrics else 0,
            "failed_requests": model.metrics.get("failed_requests", 0) if model.metrics else 0,
            "total_tokens": model.metrics.get("total_tokens", 0) if model.metrics else 0,
            "average_latency": model.metrics.get("average_latency", 0.0) if model.metrics else 0.0,
            "config": model.configuration
        }
        
        return model_data
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        return {"error": str(e)}


@router.get("/sources")
async def list_sources_resource():
    """List all registered data sources as a resource."""
    try:
        # Use MCP's database functionality
        async with get_session() as session:
            sources_list = await models.get_data_sources(session)
        
        source_list = []
        for source in sources_list:
            source_list.append({
                "name": source.name,
                "type": source.type,
                "description": source.description,
                "is_active": source.is_active
            })
        
        return source_list
    except Exception as e:
        logger.error(f"Error listing data sources: {str(e)}")
        return {"error": str(e)}


@router.get("/snowflake/{source_name}/{path:path}")
async def get_snowflake_resource(source_name: str, path: str):
    """Get data from Snowflake."""
    try:
        # Use MCP's database functionality to get data source
        async with get_session() as session:
            source = await models.get_data_source_by_name(session, source_name)
        
        if not source or source.type != "snowflake":
            return {"error": f"Snowflake data source {source_name} not found"}
        
        # Use MCP's storage functionality
        storage = SnowflakeStorage(source.configuration)
        data = await storage.read_file(path)
        
        return {"data": data.decode("utf-8")}
    except Exception as e:
        logger.error(f"Error accessing Snowflake data {path} from {source_name}: {str(e)}")
        return {"error": str(e)}


@router.get("/azure/{source_name}/{path:path}")
async def get_azure_resource(source_name: str, path: str):
    """Get data from Azure Storage."""
    try:
        # Use MCP's database functionality to get data source
        async with get_session() as session:
            source = await models.get_data_source_by_name(session, source_name)
        
        if not source or source.type != "azure":
            return {"error": f"Azure data source {source_name} not found"}
        
        # Use MCP's storage functionality
        storage = AzureStorage(source.configuration)
        data = await storage.read_file(path)
        
        return {"data": data.decode("utf-8")}
    except Exception as e:
        logger.error(f"Error accessing Azure data {path} from {source_name}: {str(e)}")
        return {"error": str(e)}


@router.get("/s3/{source_name}/{path:path}")
async def get_s3_resource(source_name: str, path: str):
    """Get data from S3 Storage."""
    try:
        # Use MCP's database functionality to get data source
        async with get_session() as session:
            source = await models.get_data_source_by_name(session, source_name)
        
        if not source or source.type != "s3":
            return {"error": f"S3 data source {source_name} not found"}
        
        # Use MCP's storage functionality
        storage = S3Storage(source.configuration)
        data = await storage.read_file(path)
        
        return {"data": data.decode("utf-8")}
    except Exception as e:
        logger.error(f"Error accessing S3 data {path} from {source_name}: {str(e)}")
        return {"error": str(e)}


@router.post("/tools/query_snowflake")
async def query_snowflake(source_name: str, query: str):
    """Execute a query against a Snowflake data source."""
    try:
        # Use MCP's database functionality to get data source
        async with get_session() as session:
            source = await models.get_data_source_by_name(session, source_name)
        
        if not source or source.type != "snowflake":
            return {"error": f"Snowflake data source {source_name} not found"}
        
        # Use MCP's storage functionality
        storage = SnowflakeStorage(source.configuration)
        result = await storage.execute_query(query)
        
        return result
    except Exception as e:
        logger.error(f"Error executing Snowflake query on {source_name}: {str(e)}")
        return {"error": str(e)}


@router.post("/tools/generate_with_model")
async def generate_with_model(model_id: str, prompt: str):
    """Generate text using a model."""
    try:
        client = await get_model_client(model_id)
        result = await client.generate(prompt)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error generating with model {model_id}: {str(e)}")
        return {"error": f"Error generating with model {model_id}: {str(e)}"}


# Mock implementation of resource/tool registration for compatibility
def resource(path):
    """Decorator to register a resource handler."""
    def decorator(func):
        resource_handlers[path] = func
        return func
    return decorator

def tool():
    """Decorator to register a tool handler."""
    def decorator(func):
        tool_handlers[func.__name__] = func
        return func
    return decorator

def prompt(name):
    """Decorator to register a prompt template."""
    def decorator(func):
        prompt_handlers[name] = func
        return func
    return decorator

# Add decorator methods to mcp_server for compatibility
mcp_server.resource = resource
mcp_server.tool = tool
mcp_server.prompt = prompt

# Function to get the router for integration with FastAPI
def get_router():
    """Get the FastAPI router with MCP endpoints."""
    return router

def get_mcp_server():
    """Return the MCP server instance."""
    return mcp_server

def create_mcp_app():
    """Create a FastAPI app with the MCP server."""
    return mcp_server.mount_to_fastapi() 