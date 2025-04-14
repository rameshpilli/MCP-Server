"""
MCP Server Module

This module provides the server components for the MCP package.
"""

from .config import ServerConfig

def create_mcp_router():
    """Create a FastAPI router for MCP endpoints."""
    # This is a mock implementation for compatibility
    from fastapi import APIRouter
    router = APIRouter(prefix="/mcp")
    return router 