"""
FastMCP Server Integration

This module provides FastAPI integration for the MCP package.
"""

from fastapi import APIRouter, FastAPI
from typing import Dict, Any, List, Optional, Callable

class FastMCP:
    """
    FastMCP integrates MCP functionality with FastAPI.
    """
    
    def __init__(self, name: str = "MCP Server"):
        """Initialize the FastMCP server."""
        self.name = name
        self.router = APIRouter()
        self.handlers = {}
    
    def add_resource_handler(self, resource_type: str, handler: Callable):
        """Register a resource handler."""
        self.handlers[resource_type] = handler
    
    def get_router(self) -> APIRouter:
        """Get the FastAPI router with registered endpoints."""
        return self.router 