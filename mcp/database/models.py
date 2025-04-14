"""
MCP Database Models

This module provides SQLAlchemy models for the MCP database.
"""

from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
from sqlalchemy import Column, String, Boolean, Integer, Float, ForeignKey, DateTime, JSON, select
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

# Create base class for models
Base = declarative_base()

class Model(Base):
    """Model record for storing model information."""
    __tablename__ = "mcp_models"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    version = Column(String, nullable=True)
    api_base = Column(String, nullable=True)
    backend = Column(String, nullable=False)
    configuration = Column(JSON, nullable=True, default={})
    is_active = Column(Boolean, default=True)
    
    # Metrics
    metrics = Column(JSON, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

class DataSource(Base):
    """Data source model."""
    __tablename__ = "mcp_data_sources"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    type = Column(String, nullable=False)
    description = Column(String, nullable=True)
    configuration = Column(JSON, nullable=True, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

class APIKey(Base):
    """API key model."""
    __tablename__ = "mcp_api_keys"
    
    id = Column(String, primary_key=True)
    key = Column(String, nullable=False, unique=True, index=True)
    owner = Column(String, nullable=False)
    model_id = Column(String, ForeignKey("mcp_models.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    expires_at = Column(DateTime, nullable=True)
    rate_limit = Column(String, nullable=True)
    permissions = Column(JSON, nullable=True, default=["model:access"])

# Simple CRUD operations for models
async def get_model(session, model_id):
    """Get a model by ID."""
    model = await session.get(Model, model_id)
    return model

async def get_models(session):
    """Get all models."""
    result = await session.execute(select(Model))
    return result.scalars().all()

async def create_model(session, model):
    """Create a new model."""
    session.add(model)
    await session.commit()
    return model

async def update_model(session, model):
    """Update an existing model."""
    session.add(model)
    await session.commit()
    return model

async def delete_model(session, model_id):
    """Delete a model."""
    model = await get_model(session, model_id)
    if model:
        await session.delete(model)
        await session.commit()
        return True
    return False

# Simple CRUD operations for data sources
async def get_data_source(session, source_id):
    """Get a data source by ID."""
    source = await session.get(DataSource, source_id)
    return source

async def get_data_source_by_name(session, name):
    """Get a data source by name."""
    result = await session.execute(select(DataSource).where(DataSource.name == name))
    return result.scalars().first()

async def get_data_sources(session):
    """Get all data sources."""
    result = await session.execute(select(DataSource))
    return result.scalars().all()

async def create_data_source(session, data_source):
    """Create a new data source."""
    session.add(data_source)
    await session.commit()
    return data_source

async def update_data_source(session, data_source):
    """Update an existing data source."""
    session.add(data_source)
    await session.commit()
    return data_source

# Helper function for API keys
async def create_api_key_with_value(session, key, owner, model_id=None, expires_at=None, rate_limit=None):
    """Create a new API key with a specific value."""
    api_key = APIKey(
        id=f"key_{key[:8]}",
        key=key,
        owner=owner,
        model_id=model_id,
        expires_at=expires_at,
        rate_limit=rate_limit or "100/minute"
    )
    session.add(api_key)
    await session.commit()
    return api_key 