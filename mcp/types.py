"""
MCP Type Definitions

This module provides type definitions for MCP.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

class TextContent(BaseModel):
    """Text content type for MCP."""
    text: str
    
class ImageContent(BaseModel):
    """Image content type for MCP."""
    url: str
    alt_text: Optional[str] = None
    
class DataContent(BaseModel):
    """Data content type for MCP."""
    format: str  # csv, json, etc.
    data: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    
class Resource(BaseModel):
    """Resource model for MCP."""
    id: str
    type: str
    attributes: Dict[str, Any] = {}
    
class QueryResult(BaseModel):
    """Query result for MCP."""
    data: Union[List[Dict[str, Any]], Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    status: str = "success"
    error: Optional[str] = None 