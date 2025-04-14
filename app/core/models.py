"""Database models for compatibility with MCP.

This module defines models that maintain compatibility
with the existing application while utilizing the MCP database.
"""

from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
from app.core.database import Base
from app.core.config import ModelBackend

class ModelRecord:
    """Model record for storing model information."""
    
    def __init__(self, model_id, name, description=None, version=None, api_base=None, 
                 backend=None, config=None, is_active=True, total_requests=0,
                 successful_requests=0, failed_requests=0, total_tokens=0, average_latency=0.0,
                 created_at=None, updated_at=None, last_used=None):
        self.model_id = model_id
        self.name = name
        self.description = description
        self.version = version
        self.api_base = api_base
        self.backend = backend
        self.config = config or {}
        self.is_active = is_active
        
        # Metrics
        self.total_requests = total_requests
        self.successful_requests = successful_requests
        self.failed_requests = failed_requests
        self.total_tokens = total_tokens
        self.average_latency = average_latency
        
        # Timestamps
        self.created_at = created_at or datetime.now(UTC)
        self.updated_at = updated_at or datetime.now(UTC)
        self.last_used = last_used

class APIKey:
    """API key model for authentication."""
    
    def __init__(self, id, key, owner, created_at=None, expires_at=None, 
                 rate_limit=None, permissions=None):
        self.id = id
        self.key = key
        self.owner = owner
        self.created_at = created_at or datetime.now(UTC)
        self.expires_at = expires_at
        self.rate_limit = rate_limit  # Format: "100/minute", "1000/day", etc.
        self.permissions = permissions or ["model:access"]

class DataSource:
    """Data source model for external data connections."""
    
    def __init__(self, id, name, source_type, description=None, config=None, 
                 is_active=True, is_healthy=True, created_at=None, last_health_check=None):
        self.id = id
        self.name = name
        self.source_type = source_type  # "snowflake", "azure_blob", "s3", etc.
        self.description = description
        self.config = config or {}
        self.is_active = is_active
        self.is_healthy = is_healthy
        self.created_at = created_at or datetime.now(UTC)
        self.last_health_check = last_health_check

class QueryHistory:
    """Query history model for tracking data queries."""
    
    def __init__(self, id, api_key, natural_query=None, sql_query=None, data_source_id=None,
                 execution_time=None, row_count=None, status=None, error=None, 
                 response=None, timestamp=None):
        self.id = id
        self.api_key = api_key
        self.natural_query = natural_query
        self.sql_query = sql_query
        self.data_source_id = data_source_id
        self.execution_time = execution_time  # In seconds
        self.row_count = row_count
        self.status = status  # "success", "error", "timeout", etc.
        self.error = error
        self.response = response
        self.timestamp = timestamp or datetime.now(UTC) 