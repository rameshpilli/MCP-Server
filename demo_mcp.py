#!/usr/bin/env python3
"""
MCP Demo Script

This script demonstrates how to use the MCP (Model Context Protocol) implementation
for registering models and data sources, and making basic requests.

Usage:
    python demo_mcp.py
"""

import asyncio
import os
import sys
from pathlib import Path
import json
import uuid

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import MCP components
from mcp.database import init_database, get_session, close_database
from mcp.database.models import Model, DataSource, create_model, create_data_source
from app.core.config import get_settings
from app.core.logger import logger

# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Get settings
settings = get_settings()

async def demo_register_model():
    """Demonstrate how to register a model with MCP."""
    logger.info("=== Demo: Register a model with MCP ===")
    
    # Create a sample model
    model_id = f"demo-model-{uuid.uuid4().hex[:8]}"
    model = Model(
        id=model_id,
        name="Demo GPT-4 Model",
        description="A demonstration model for MCP",
        backend="openai",
        version="1.0",
        api_base="https://api.openai.com/v1",
        configuration={
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        is_active=True,
        metrics={
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "average_latency": 0.0
        }
    )
    
    # Register the model
    async with get_session() as session:
        await create_model(session, model)
    
    logger.info(f"Model registered with ID: {model_id}")
    return model_id

async def demo_register_data_sources():
    """Demonstrate how to register different data sources with MCP."""
    logger.info("=== Demo: Register data sources with MCP ===")
    
    # Create sample data sources
    sources = []
    
    # Snowflake data source
    snowflake_source = DataSource(
        id=f"snowflake-{uuid.uuid4().hex[:8]}",
        name="Demo Snowflake",
        type="snowflake",
        description="A demonstration Snowflake data source",
        configuration={
            "account": os.getenv("SNOWFLAKE_ACCOUNT", "your-account"),
            "user": os.getenv("SNOWFLAKE_USER", "your-user"),
            "password": os.getenv("SNOWFLAKE_PASSWORD", "your-password"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "your-warehouse"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "your-database"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "your-schema")
        },
        is_active=True
    )
    sources.append(snowflake_source)
    
    # S3 data source
    s3_source = DataSource(
        id=f"s3-{uuid.uuid4().hex[:8]}",
        name="Demo S3",
        type="s3",
        description="A demonstration S3 data source",
        configuration={
            "bucket": os.getenv("S3_BUCKET", "your-bucket"),
            "region": os.getenv("AWS_REGION", "us-east-1"),
            "access_key": os.getenv("AWS_ACCESS_KEY_ID", "your-access-key"),
            "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", "your-secret-key")
        },
        is_active=True
    )
    sources.append(s3_source)
    
    # Azure data source
    azure_source = DataSource(
        id=f"azure-{uuid.uuid4().hex[:8]}",
        name="Demo Azure",
        type="azure",
        description="A demonstration Azure Blob Storage data source",
        configuration={
            "connection_string": os.getenv("AZURE_CONNECTION_STRING", "your-connection-string"),
            "container": os.getenv("AZURE_CONTAINER", "your-container")
        },
        is_active=True
    )
    sources.append(azure_source)
    
    # Register the data sources
    async with get_session() as session:
        for source in sources:
            await create_data_source(session, source)
            logger.info(f"Data source registered: {source.name} ({source.type})")
    
    return [source.name for source in sources]

async def demo_query_database():
    """Demonstrate how to query for registered models and data sources."""
    logger.info("=== Demo: Query for registered resources ===")
    
    async with get_session() as session:
        # Query for models
        from mcp.database.models import get_models
        models = await get_models(session)
        
        logger.info(f"Found {len(models)} registered models:")
        for model in models:
            logger.info(f"  - {model.name} ({model.id}): {model.description}")
        
        # Query for data sources
        from mcp.database.models import get_data_sources
        sources = await get_data_sources(session)
        
        logger.info(f"Found {len(sources)} registered data sources:")
        for source in sources:
            logger.info(f"  - {source.name} ({source.type}): {source.description}")

async def demo_http_client():
    """Demonstrate how to use the MCP server through HTTP requests."""
    logger.info("=== Demo: Using MCP server through HTTP ===")
    
    import httpx
    
    # Assuming the MCP server is running on localhost:8000
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # List models
        logger.info("Fetching models from MCP server...")
        response = await client.get(f"{base_url}/mcp/models")
        
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Found {len(models)} models from the MCP server")
            if models:
                logger.info(f"First model: {json.dumps(models[0], indent=2)}")
        else:
            logger.error(f"Failed to fetch models: {response.status_code} {response.text}")
        
        # List data sources
        logger.info("Fetching data sources from MCP server...")
        response = await client.get(f"{base_url}/mcp/sources")
        
        if response.status_code == 200:
            sources = response.json()
            logger.info(f"Found {len(sources)} data sources from the MCP server")
            if sources:
                logger.info(f"First data source: {json.dumps(sources[0], indent=2)}")
        else:
            logger.error(f"Failed to fetch data sources: {response.status_code} {response.text}")

async def run_demo():
    """Run the full MCP demo."""
    logger.info("Starting MCP demo...")
    
    try:
        # Initialize the database
        db_config = settings.get_mcp_database_config()
        await init_database(db_config)
        
        # Run demo components
        model_id = await demo_register_model()
        sources = await demo_register_data_sources()
        await demo_query_database()
        
        # The HTTP client demo requires the server to be running
        logger.info("""
To test the HTTP client demo:
1. Run the MCP server in a separate terminal: python mcp_server.py
2. Then run: python demo_mcp.py --http-client
        """)
        
        # Only run the HTTP client demo if explicitly requested
        if "--http-client" in sys.argv:
            await demo_http_client()
        
        logger.info("Demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise
    finally:
        # Close database connections
        await close_database()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo()) 