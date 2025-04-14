#!/usr/bin/env python3
"""
Simple MCP Server

This is a simplified MCP server that doesn't require database access.
It uses in-memory storage for models and data sources.
"""

import os
import json
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, APIRouter, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mcp_server")

# Load sample data
storage_dir = Path("storage")
models_file = storage_dir / "models.json"
sources_file = storage_dir / "data_sources.json"

# Load models
if models_file.exists():
    with open(models_file, "r") as f:
        models = json.load(f)
else:
    models = [
        {
            "id": "openai-gpt4",
            "name": "OpenAI GPT-4",
            "description": "GPT-4 model from OpenAI",
            "backend": "openai",
            "version": "1.0.0",
            "api_base": "https://api.openai.com/v1",
            "configuration": {"model_name": "gpt-4"},
            "is_active": True,
            "metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "average_latency": 0.0
            }
        },
        {
            "id": "anthropic-claude",
            "name": "Anthropic Claude",
            "description": "Claude model from Anthropic",
            "backend": "anthropic",
            "version": "1.0.0",
            "api_base": "https://api.anthropic.com/v1",
            "configuration": {"model_name": "claude-2"},
            "is_active": True,
            "metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "average_latency": 0.0
            }
        }
    ]

# Load data sources
if sources_file.exists():
    with open(sources_file, "r") as f:
        data_sources = json.load(f)
else:
    data_sources = [
        {
            "id": "snowflake-demo",
            "name": "Snowflake Demo",
            "type": "snowflake",
            "description": "Demo Snowflake data source",
            "configuration": {
                "account": "your-account",
                "user": "your-user",
                "password": "your-password",
                "warehouse": "your-warehouse",
                "database": "your-database",
                "schema": "your-schema"
            },
            "is_active": True
        },
        {
            "id": "azure-demo",
            "name": "Azure Demo",
            "type": "azure",
            "description": "Demo Azure Storage data source",
            "configuration": {
                "account": "your-storage-account",
                "key": "your-storage-key",
                "container": "your-container-name"
            },
            "is_active": True
        },
        {
            "id": "s3-demo",
            "name": "S3 Demo",
            "type": "s3",
            "description": "Demo S3 data source",
            "configuration": {
                "access_key": "your-access-key",
                "secret_key": "your-secret-key",
                "region": "your-region",
                "bucket": "your-bucket"
            },
            "is_active": True
        }
    ]

# Create FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Simple MCP Server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
router = APIRouter(prefix="/api/mcp")

# Routes for models
@router.get("/models")
async def list_models():
    """List all models."""
    logger.info("Listing models")
    return models

@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get a model by ID."""
    logger.info(f"Getting model: {model_id}")
    for model in models:
        if model["id"] == model_id:
            return model
    return {"error": f"Model {model_id} not found"}

# Routes for data sources
@router.get("/sources")
async def list_sources():
    """List all data sources."""
    logger.info("Listing data sources")
    return data_sources

@router.get("/snowflake/{source_name}/{path:path}")
async def get_snowflake_data(source_name: str, path: str):
    """Get data from Snowflake."""
    logger.info(f"Getting Snowflake data: {source_name}/{path}")
    # In a real implementation, this would query Snowflake
    return {
        "data": f"Sample data from Snowflake source {source_name} at path {path}",
        "columns": ["col1", "col2", "col3"],
        "rows": [
            {"col1": "value1", "col2": "value2", "col3": "value3"},
            {"col1": "value4", "col2": "value5", "col3": "value6"}
        ]
    }

@router.get("/azure/{source_name}/{path:path}")
async def get_azure_data(source_name: str, path: str):
    """Get data from Azure Storage."""
    logger.info(f"Getting Azure data: {source_name}/{path}")
    # In a real implementation, this would fetch from Azure
    return {
        "data": f"Sample data from Azure source {source_name} at path {path}",
        "content_type": "text/csv",
        "size": 1024
    }

@router.get("/s3/{source_name}/{path:path}")
async def get_s3_data(source_name: str, path: str):
    """Get data from S3."""
    logger.info(f"Getting S3 data: {source_name}/{path}")
    # In a real implementation, this would fetch from S3
    return {
        "data": f"Sample data from S3 source {source_name} at path {path}",
        "content_type": "text/csv",
        "size": 2048
    }

# Tool endpoints
@router.post("/tools/query_snowflake")
async def query_snowflake(request: Request):
    """Execute a query against Snowflake."""
    data = await request.json()
    source_name = data.get("source_name")
    query = data.get("query")
    logger.info(f"Executing Snowflake query on {source_name}: {query}")
    # In a real implementation, this would execute a query
    return {
        "status": "success",
        "columns": ["col1", "col2", "col3"],
        "rows": [
            {"col1": "value1", "col2": "value2", "col3": "value3"},
            {"col1": "value4", "col2": "value5", "col3": "value6"}
        ],
        "row_count": 2,
        "execution_time": 0.5
    }

@router.post("/tools/generate_with_model")
async def generate_with_model(request: Request):
    """Generate text using a model."""
    data = await request.json()
    model_id = data.get("model_id")
    prompt = data.get("prompt")
    logger.info(f"Generating with model {model_id}: {prompt[:50]}...")
    # In a real implementation, this would call the model API
    return {
        "result": f"This is a mock response from model {model_id} to the prompt: '{prompt[:50]}...'"
    }

# Add router to app
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "MCP Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "models": "/api/mcp/models",
            "sources": "/api/mcp/sources",
            "tools": {
                "query_snowflake": "/api/mcp/tools/query_snowflake",
                "generate_with_model": "/api/mcp/tools/generate_with_model"
            }
        }
    }

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting MCP server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port) 