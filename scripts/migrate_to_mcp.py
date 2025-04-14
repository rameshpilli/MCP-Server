#!/usr/bin/env python
"""
Data Migration Script: Custom Database to MCP

This script migrates data from our custom database structure
to the MCP package's built-in database format.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import legacy database models
from app.core.models import ModelRecord, APIKey, DataSource
from app.core.database import get_db, init_db
from sqlalchemy import select

# Import MCP models
from mcp.database import init_database, get_session, models
from mcp.database.models import Model as MCPModel, DataSource as MCPDataSource
from app.core.config import get_settings
from app.core.logger import logger

settings = get_settings()

async def migrate_models():
    """Migrate models from the legacy database to MCP format."""
    logger.info("Migrating models...")
    
    # Initialize both databases
    logger.info("Initializing legacy database...")
    await init_db()
    
    logger.info("Initializing MCP database...")
    db_config = settings.get_mcp_database_config()
    await init_database(db_config)
    
    # Get models from legacy database
    async for db in get_db():
        result = await db.execute(select(ModelRecord))
        legacy_models = result.scalars().all()
        logger.info(f"Found {len(legacy_models)} models in legacy database")
        
        # Migrate each model to MCP format
        for model in legacy_models:
            logger.info(f"Migrating model: {model.model_id}")
            
            # Convert to MCP format
            mcp_model = MCPModel(
                id=model.model_id,
                name=model.name,
                description=model.description,
                backend=model.backend,
                version=model.version,
                api_base=model.api_base,
                configuration=model.config or {},
                is_active=model.is_active,
                metrics={
                    "total_requests": model.total_requests,
                    "successful_requests": model.successful_requests,
                    "failed_requests": model.failed_requests,
                    "total_tokens": model.total_tokens,
                    "average_latency": model.average_latency
                }
            )
            
            # Save to MCP database
            async with get_session() as session:
                try:
                    # Check if model already exists
                    existing = await models.get_model(session, model.model_id)
                    if existing:
                        logger.info(f"Model {model.model_id} already exists in MCP database, updating...")
                        await models.update_model(session, mcp_model)
                    else:
                        logger.info(f"Creating model {model.model_id} in MCP database...")
                        await models.create_model(session, mcp_model)
                except Exception as e:
                    logger.error(f"Error migrating model {model.model_id}: {str(e)}")
        
        break  # Exit the async generator

async def migrate_api_keys():
    """Migrate API keys from the legacy database to MCP format."""
    logger.info("Migrating API keys...")
    
    # Get API keys from legacy database
    async for db in get_db():
        result = await db.execute(select(APIKey))
        legacy_keys = result.scalars().all()
        logger.info(f"Found {len(legacy_keys)} API keys in legacy database")
        
        # Migrate each key to MCP format
        for key in legacy_keys:
            logger.info(f"Migrating API key for: {key.owner}")
            
            # Convert to MCP format
            async with get_session() as session:
                try:
                    # Create new API key in MCP format
                    await models.create_api_key_with_value(
                        session,
                        key=key.key,  # Reuse the same key value
                        owner=key.owner,
                        model_id=key.owner,  # Use owner as model_id if applicable
                        expires_at=key.expires_at,
                        rate_limit=key.rate_limit or "100/minute"
                    )
                    logger.info(f"Migrated API key for {key.owner}")
                except Exception as e:
                    logger.error(f"Error migrating API key for {key.owner}: {str(e)}")
        
        break  # Exit the async generator

async def migrate_data_sources():
    """Migrate data sources from the legacy database to MCP format."""
    logger.info("Migrating data sources...")
    
    # Get data sources from legacy database
    async for db in get_db():
        result = await db.execute(select(DataSource))
        legacy_sources = result.scalars().all()
        logger.info(f"Found {len(legacy_sources)} data sources in legacy database")
        
        # Migrate each source to MCP format
        for source in legacy_sources:
            logger.info(f"Migrating data source: {source.name}")
            
            # Convert to MCP format
            mcp_source = MCPDataSource(
                name=source.name,
                type=source.source_type,
                description=source.description or f"{source.source_type} data source",
                configuration=source.config or {},
                is_active=source.is_active
            )
            
            # Save to MCP database
            async with get_session() as session:
                try:
                    # Check if source already exists
                    existing = await models.get_data_source_by_name(session, source.name)
                    if existing:
                        logger.info(f"Data source {source.name} already exists in MCP database, updating...")
                        await models.update_data_source(session, mcp_source)
                    else:
                        logger.info(f"Creating data source {source.name} in MCP database...")
                        await models.create_data_source(session, mcp_source)
                except Exception as e:
                    logger.error(f"Error migrating data source {source.name}: {str(e)}")
        
        break  # Exit the async generator

async def migrate_all():
    """Run all migration functions."""
    try:
        # Create backup of old data
        await create_backup()
        
        # Run migrations
        await migrate_models()
        await migrate_api_keys()
        await migrate_data_sources()
        
        logger.info("✓ Migration completed successfully")
    except Exception as e:
        logger.error(f"✗ Migration failed: {str(e)}")
        raise

async def create_backup():
    """Create a backup of the legacy database."""
    logger.info("Creating backup of legacy database...")
    
    backup_dir = project_root / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"legacy_db_backup_{timestamp}.json"
    
    backup_data = {
        "models": [],
        "api_keys": [],
        "data_sources": []
    }
    
    # Get models
    async for db in get_db():
        # Backup models
        result = await db.execute(select(ModelRecord))
        models = result.scalars().all()
        for model in models:
            backup_data["models"].append({
                "model_id": model.model_id,
                "name": model.name,
                "description": model.description,
                "backend": model.backend,
                "version": model.version,
                "api_base": model.api_base,
                "config": model.config,
                "is_active": model.is_active,
                "total_requests": model.total_requests,
                "successful_requests": model.successful_requests,
                "failed_requests": model.failed_requests,
                "total_tokens": model.total_tokens,
                "average_latency": model.average_latency,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "updated_at": model.updated_at.isoformat() if model.updated_at else None
            })
        
        # Backup API keys
        result = await db.execute(select(APIKey))
        api_keys = result.scalars().all()
        for key in api_keys:
            backup_data["api_keys"].append({
                "id": key.id,
                "key": key.key,
                "owner": key.owner,
                "created_at": key.created_at.isoformat() if key.created_at else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "rate_limit": key.rate_limit,
                "permissions": key.permissions
            })
        
        # Backup data sources
        result = await db.execute(select(DataSource))
        data_sources = result.scalars().all()
        for source in data_sources:
            backup_data["data_sources"].append({
                "id": source.id,
                "name": source.name,
                "source_type": source.source_type,
                "description": source.description,
                "config": source.config,
                "is_active": source.is_active,
                "is_healthy": source.is_healthy,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "last_health_check": source.last_health_check.isoformat() if source.last_health_check else None
            })
        
        break  # Exit the async generator
    
    # Write backup to file
    with open(backup_file, "w") as f:
        json.dump(backup_data, f, indent=2)
    
    logger.info(f"Backup created at {backup_file}")
    return backup_file

if __name__ == "__main__":
    asyncio.run(migrate_all()) 