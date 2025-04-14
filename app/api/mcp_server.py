"""MCP (Model Context Protocol) server integration."""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from app.core.database import get_db
from app.core.models import ModelRecord, APIKey, DataSource
from app.core.security import APIKeyManager
from app.core.storage import S3StorageBackend, AzureStorageBackend, LocalStorageBackend, SnowflakeStorageBackend
from app.core.config import get_settings
from app.core.logger import logger
from app.core.model_client import ModelClientFactory, ModelClient

# Get settings
settings = get_settings()

# Initialize MCP server
mcp_server = FastMCP("MCP Model Registry")

# Store clients by model ID
model_clients: Dict[str, ModelClient] = {}


async def get_model_client(model_id: str, db: AsyncSession) -> ModelClient:
    """Get or create a model client for the specified model ID."""
    if model_id in model_clients:
        return model_clients[model_id]
    
    # Get model from database
    result = await db.execute(
        select(ModelRecord).where(ModelRecord.model_id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise ValueError(f"Model {model_id} not found")
    
    # Create client
    client_factory = ModelClientFactory()
    client = client_factory.create_client(model)
    
    # Store for reuse
    model_clients[model_id] = client
    
    return client


# ---------------------------
# Resource handlers
# ---------------------------

@mcp_server.resource("models://list")
async def list_models_resource(db: AsyncSession = Depends(get_db)) -> str:
    """List all registered models as a resource."""
    try:
        result = await db.execute(select(ModelRecord))
        models = result.scalars().all()
        
        model_list = []
        for model in models:
            model_list.append({
                "model_id": model.model_id,
                "name": model.name,
                "description": model.description,
                "backend": model.backend,
                "version": model.version,
                "is_active": model.is_active,
                "total_requests": model.total_requests,
                "successful_requests": model.successful_requests
            })
        
        return json.dumps(model_list, indent=2)
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp_server.resource("models://{model_id}")
async def get_model_resource(model_id: str, db: AsyncSession = Depends(get_db)) -> str:
    """Get details for a specific model."""
    try:
        result = await db.execute(
            select(ModelRecord).where(ModelRecord.model_id == model_id)
        )
        model = result.scalar_one_or_none()
        
        if not model:
            return json.dumps({"error": f"Model {model_id} not found"})
        
        model_data = {
            "model_id": model.model_id,
            "name": model.name,
            "description": model.description,
            "backend": model.backend,
            "version": model.version,
            "api_base": model.api_base,
            "is_active": model.is_active,
            "total_requests": model.total_requests,
            "successful_requests": model.successful_requests,
            "failed_requests": model.failed_requests,
            "total_tokens": model.total_tokens,
            "average_latency": model.average_latency,
            "config": model.config
        }
        
        return json.dumps(model_data, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp_server.resource("sources://list")
async def list_data_sources_resource(db: AsyncSession = Depends(get_db)) -> str:
    """List all data sources as a resource."""
    try:
        result = await db.execute(select(DataSource))
        sources = result.scalars().all()
        
        source_list = []
        for source in sources:
            source_list.append({
                "id": source.id,
                "name": source.name,
                "source_type": source.source_type,
                "is_active": source.is_active,
                "is_healthy": source.is_healthy,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "last_health_check": source.last_health_check.isoformat() if source.last_health_check else None
            })
        
        return json.dumps(source_list, indent=2)
    except Exception as e:
        logger.error(f"Error listing data sources: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp_server.resource("snowflake://{source_name}/{path}")
async def get_snowflake_data(source_name: str, path: str, db: AsyncSession = Depends(get_db)) -> str:
    """Get data from Snowflake data source."""
    try:
        # Get the data source from database
        result = await db.execute(
            select(DataSource).where(
                DataSource.name == source_name,
                DataSource.source_type == "snowflake",
                DataSource.is_active == True
            )
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return json.dumps({"error": f"Snowflake data source {source_name} not found or inactive"})
        
        # Connect to Snowflake
        connection_params = source.config.get("connection", {})
        snowflake_backend = SnowflakeStorageBackend(connection_params)
        
        # Read the data
        data = snowflake_backend.read_file(path)
        return data.decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading from Snowflake {source_name}/{path}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp_server.resource("azure://{source_name}/{path}")
async def get_azure_data(source_name: str, path: str, db: AsyncSession = Depends(get_db)) -> str:
    """Get data from Azure Blob Storage."""
    try:
        # Get the data source from database
        result = await db.execute(
            select(DataSource).where(
                DataSource.name == source_name,
                DataSource.source_type == "azure",
                DataSource.is_active == True
            )
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return json.dumps({"error": f"Azure data source {source_name} not found or inactive"})
        
        # Connect to Azure
        connection_string = source.config.get("connection_string", "")
        container = source.config.get("container", "")
        azure_backend = AzureStorageBackend(connection_string, container)
        
        # Read the data
        data = azure_backend.read_file(path)
        return data.decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading from Azure {source_name}/{path}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp_server.resource("s3://{source_name}/{path}")
async def get_s3_data(source_name: str, path: str, db: AsyncSession = Depends(get_db)) -> str:
    """Get data from S3 storage."""
    try:
        # Get the data source from database
        result = await db.execute(
            select(DataSource).where(
                DataSource.name == source_name,
                DataSource.source_type == "s3",
                DataSource.is_active == True
            )
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return json.dumps({"error": f"S3 data source {source_name} not found or inactive"})
        
        # Connect to S3
        bucket = source.config.get("bucket", "")
        endpoint_url = source.config.get("endpoint_url", None)
        aws_access_key_id = source.config.get("aws_access_key_id", None)
        aws_secret_access_key = source.config.get("aws_secret_access_key", None)
        
        s3_backend = S3StorageBackend(
            bucket=bucket,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Read the data
        data = s3_backend.read_file(path)
        return data.decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading from S3 {source_name}/{path}: {str(e)}")
        return json.dumps({"error": str(e)})


# ---------------------------
# Tool handlers
# ---------------------------

@mcp_server.tool()
async def generate_with_model(model_id: str, prompt: str, db: AsyncSession = Depends(get_db)) -> str:
    """Generate text using a specific model."""
    try:
        client = await get_model_client(model_id, db)
        response = await client.generate(prompt, {})
        return response
    except Exception as e:
        logger.error(f"Error generating with model {model_id}: {str(e)}")
        return f"Error: {str(e)}"


@mcp_server.tool()
async def query_snowflake(source_name: str, query: str, db: AsyncSession = Depends(get_db)) -> str:
    """Execute a query against a Snowflake data source."""
    try:
        # Get the data source from database
        result = await db.execute(
            select(DataSource).where(
                DataSource.name == source_name,
                DataSource.source_type == "snowflake",
                DataSource.is_active == True
            )
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return f"Error: Snowflake data source {source_name} not found or inactive"
        
        # Connect to Snowflake
        connection_params = source.config.get("connection", {})
        conn = source.get_connection()
        
        # Execute query safely (no INSERTs, UPDATEs, DELETEs allowed)
        query = query.strip().upper()
        if any(keyword in query for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]):
            return "Error: Only SELECT queries are allowed"
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Fetch results
        results = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        
        # Convert to list of dicts
        rows = []
        for row in results:
            rows.append(dict(zip(columns, row)))
        
        # Format as pretty JSON
        return json.dumps(rows, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error querying Snowflake {source_name}: {str(e)}")
        return f"Error: {str(e)}"


@mcp_server.tool()
async def list_storage_files(source_name: str, path: str = "", db: AsyncSession = Depends(get_db)) -> str:
    """List files in a storage path."""
    try:
        # Get the data source from database
        result = await db.execute(
            select(DataSource).where(
                DataSource.name == source_name,
                DataSource.is_active == True
            )
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return f"Error: Data source {source_name} not found or inactive"
        
        # Choose backend based on source type
        backend = None
        if source.source_type == "s3":
            bucket = source.config.get("bucket", "")
            endpoint_url = source.config.get("endpoint_url", None)
            aws_access_key_id = source.config.get("aws_access_key_id", None)
            aws_secret_access_key = source.config.get("aws_secret_access_key", None)
            
            backend = S3StorageBackend(
                bucket=bucket,
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        elif source.source_type == "azure":
            connection_string = source.config.get("connection_string", "")
            container = source.config.get("container", "")
            backend = AzureStorageBackend(connection_string, container)
        elif source.source_type == "snowflake":
            connection_params = source.config.get("connection", {})
            backend = SnowflakeStorageBackend(connection_params)
        elif source.source_type == "local":
            base_path = source.config.get("base_path", "./data")
            backend = LocalStorageBackend(base_path)
        else:
            return f"Error: Unsupported source type {source.source_type}"
        
        # List files
        files = backend.list_files(path)
        return json.dumps(files, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error listing files from {source_name}/{path}: {str(e)}")
        return f"Error: {str(e)}"


@mcp_server.tool()
async def register_model(
    model_id: str, 
    name: str, 
    backend: str, 
    description: str = None, 
    api_base: str = None, 
    version: str = None,
    db: AsyncSession = Depends(get_db)
) -> str:
    """Register a new model with the model registry."""
    try:
        # Check if model already exists
        existing = await db.execute(
            select(ModelRecord).where(ModelRecord.model_id == model_id)
        )
        if existing.scalar_one_or_none():
            return f"Error: Model with ID {model_id} already exists"
        
        # Validate backend
        try:
            from app.core.config import ModelBackend
            backend_enum = ModelBackend(backend.lower())
        except ValueError:
            backends = [b.value for b in ModelBackend]
            return f"Error: Invalid backend. Must be one of: {', '.join(backends)}"
        
        # Create new model
        from datetime import datetime, UTC
        model = ModelRecord(
            model_id=model_id,
            name=name,
            description=description,
            version=version,
            api_base=api_base,
            backend=backend_enum,
            config={},
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_tokens=0,
            average_latency=0.0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        
        db.add(model)
        await db.flush()
        
        # Generate API key
        api_key_manager = APIKeyManager()
        api_key = await api_key_manager.create_key(
            db=db,
            owner=name,
            expires_in_days=365,
            permissions=["model:access"],
            rate_limit="100/minute"
        )
        
        await db.commit()
        
        # Return success message with the API key
        return (
            f"Model {model_id} registered successfully!\n"
            f"API Key: {api_key.key}\n"
            "Make sure to save this API key as it won't be shown again."
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Error registering model: {str(e)}")
        return f"Error: {str(e)}"


# ---------------------------
# Prompt handlers
# ---------------------------

@mcp_server.prompt()
def data_analysis_prompt(data: str, question: str = None) -> str:
    """Create a prompt for data analysis."""
    prompt_text = f"""Analyze the following data and provide insights:

```
{data}
```

"""
    if question:
        prompt_text += f"Specifically answer this question: {question}\n"
    
    prompt_text += """
Provide your analysis in a clear, structured format with:
1. Summary of the data
2. Key insights and patterns
3. Actionable recommendations
"""
    return prompt_text


@mcp_server.prompt()
def query_generator_prompt(table_description: str, question: str) -> str:
    """Generate a query based on a natural language question."""
    return f"""Generate a SQL query to answer the following question:

Question: {question}

Table information:
```
{table_description}
```

Return only the SQL query without any explanations or markdown formatting."""


def get_mcp_server():
    """Return the MCP server instance."""
    return mcp_server


def create_mcp_app():
    """Create a FastAPI app with the MCP server."""
    return mcp_server.mount_to_fastapi() 