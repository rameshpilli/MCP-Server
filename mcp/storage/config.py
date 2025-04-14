"""
MCP Storage Configuration

This module provides the configuration classes for MCP storage backends.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class StorageConfig(BaseModel):
    """Storage backend configuration."""
    type: str = "local"  # local, azure, s3, snowflake
    path: Optional[str] = None
    connection_string: Optional[str] = None
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    
    # Authentication/credentials
    account: Optional[str] = None
    key: Optional[str] = None
    secret: Optional[str] = None
    
    # Additional options
    options: Dict[str, Any] = {} 