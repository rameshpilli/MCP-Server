"""
MCP Bridge Module

This module provides the bridge between the MCP server and external tools.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from fastmcp import FastMCP, Context
from app.config import config

# Setup logging
logger = logging.getLogger('mcp_bridge')

class MCPBridge:
    def __init__(self):
        """Initialize the MCP Bridge"""
        # Lazy import to avoid circular dependency
        from .mcp_server import mcp
        self.mcp = mcp
        logger.info("MCP Bridge initialized")

    async def route_request(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a request to appropriate tools based on the message content.
        
        Args:
            message: The user's message
            context: Optional context information
            
        Returns:
            Dict containing routing information and intent
        """
        try:
            # Use the MCP server's routing capabilities
            routing_result = await self.mcp.route(message, context or {})
            return routing_result
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return {
                "intent": "unknown",
                "endpoints": []
            }

    async def execute_tool(self, tool_name: str, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a specific tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            context: Optional context information
            
        Returns:
            Result from tool execution
        """
        try:
            return await self.mcp.execute_tool(tool_name, params, context or {})
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise

    async def generate_response(self, query: str, results: List[Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response based on tool execution results.
        
        Args:
            query: Original user query
            results: List of results from tool executions
            context: Optional context information
            
        Returns:
            Generated response text
        """
        try:
            # Format results for response generation
            formatted_results = []
            for result in results:
                if isinstance(result, str):
                    formatted_results.append(result)
                else:
                    formatted_results.append(str(result))
            
            # Use the MCP server's response generation
            response = await self.mcp.generate_response(query, formatted_results, context or {})
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while processing your request."

    async def get_available_tools(self) -> Dict[str, Any]:
        """
        Get list of available tools.
        
        Returns:
            Dictionary of available tools and their descriptions
        """
        try:
            return await self.mcp.get_tools()
        except Exception as e:
            logger.error(f"Error getting available tools: {e}")
            return {}