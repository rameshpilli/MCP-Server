"""
MCP Server Configuration

This module provides the configuration classes for the MCP server.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ServerConfig(BaseModel):
    """Server configuration for MCP."""
    name: str = "MCP Server"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = ["*"]
    log_level: str = "info"
    enable_metrics: bool = True
    middleware: List[str] = []
    max_request_size: int = 10 * 1024 * 1024  # 10 MB
    timeout: int = 60 