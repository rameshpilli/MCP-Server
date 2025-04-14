#!/usr/bin/env python3
"""
Database Initialization Script

This script initializes the MCP database with some sample data.
"""

import asyncio
import os
import sys
import json
import secrets
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.absolute()))

# Load environment variables
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value

# Import MCP components
from mcp.database import init_database, get_session, close_database
from mcp.database.models import Model, DataSource, APIKey, create_model, create_data_source
from mcp.database.config import DatabaseConfig
from app.core.config import get_settings
from app.core.logger import logger

# Get settings
settings = get_settings()

async def initialize_database():
    """Initialize the MCP database with sample data."""
    try:
        # Initialize the database
        logger.info("Initializing MCP database...")
        db_config = settings.get_mcp_database_config()
        await init_database(db_config)
        
        # Create sample models
        logger.info("Creating sample models...")
        sample_models = [
            Model(
                id="openai-gpt4",
                name="OpenAI GPT-4",
                description="GPT-4 model from OpenAI",
                backend="openai",
                version="1.0.0",
                api_base="https://api.openai.com/v1",
                configuration={"model_name": "gpt-4"},
                is_active=True
            ),
            Model(
                id="anthropic-claude",
                name="Anthropic Claude",
                description="Claude model from Anthropic",
                backend="anthropic",
                version="1.0.0",
                api_base="https://api.anthropic.com/v1",
                configuration={"model_name": "claude-2"},
                is_active=True
            )
        ]
        
        # Save the models
        async with get_session() as session:
            for model in sample_models:
                await create_model(session, model)
                logger.info(f"Created model: {model.name}")
                
                # Create API key for the model
                api_key = APIKey(
                    id=f"key_{secrets.token_hex(4)}",
                    key=secrets.token_urlsafe(32),
                    owner=model.name,
                    model_id=model.id,
                    rate_limit="100/minute"
                )
                session.add(api_key)
                await session.commit()
                logger.info(f"Created API key for {model.name}: {api_key.key}")
        
        # Create sample data sources
        logger.info("Creating sample data sources...")
        sample_sources = [
            DataSource(
                id="snowflake-demo",
                name="Snowflake Demo",
                type="snowflake",
                description="Demo Snowflake data source",
                configuration={
                    "account": "your-account",
                    "user": "your-user",
                    "password": "your-password",
                    "warehouse": "your-warehouse",
                    "database": "your-database",
                    "schema": "your-schema"
                },
                is_active=True
            ),
            DataSource(
                id="azure-demo",
                name="Azure Demo",
                type="azure",
                description="Demo Azure Storage data source",
                configuration={
                    "account": "your-storage-account",
                    "key": "your-storage-key",
                    "container": "your-container-name"
                },
                is_active=True
            ),
            DataSource(
                id="s3-demo",
                name="S3 Demo",
                type="s3",
                description="Demo S3 data source",
                configuration={
                    "access_key": "your-access-key",
                    "secret_key": "your-secret-key",
                    "region": "your-region",
                    "bucket": "your-bucket"
                },
                is_active=True
            )
        ]
        
        # Save the data sources
        async with get_session() as session:
            for source in sample_sources:
                session.add(source)
                await session.commit()
                logger.info(f"Created data source: {source.name}")
        
        logger.info("✓ Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
        raise
    finally:
        # Close database connections
        await close_database()

if __name__ == "__main__":
    # Run the initialization
    asyncio.run(initialize_database()) 