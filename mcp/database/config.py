"""
MCP Database Configuration

This module provides the configuration classes for the MCP database.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any

class DatabaseConfig(BaseModel):
    """Database configuration for MCP."""
    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    connect_args: Dict[str, Any] = {} 