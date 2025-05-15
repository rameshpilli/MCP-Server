"""
Search Tools

This module contains tools for searching through documents.
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add root path to system path
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui.mcp.doc_reader import doc_reader

# Get logger
logger = logging.getLogger('mcp_server.tools.search')

async def search_docs(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Search through local documentation.

    Call this tool when you need to find specific information in local documents based on a search query.

    Args:
        query: The search query for local documents.
        context: Optional context information.
    
    Returns:
        A formatted string with search results.
    """
    try:
        results = doc_reader.search_documents(query)
        
        if not results:
            return f"No documents found matching the query: '{query}'"
            
        formatted_results = []
        for result in results:
            formatted_results.append(f"Document: {result['document']}\n")
            formatted_results.append(f"Preview: {result['preview']}\n")
            
        return "Search Results:\n\n" + "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"Error searching documents: {e}"

def register_tools(mcp):
    """Register search tools with the MCP server"""
    @mcp.tool()
    async def search_docs(ctx: Context, query: str) -> str:
        """
        Search through local documentation.

        Call this tool when you need to find specific information in local documents based on a search query.

        Args:
            ctx: The MCP server provided context.
            query: The search query for local documents.
        
        Returns:
            A formatted string with search results.
        """
        try:
            results = doc_reader.search_documents(query)
            
            if not results:
                return f"No documents found matching the query: '{query}'"
                
            formatted_results = []
            for result in results:
                formatted_results.append(f"Document: {result['document']}\n")
                formatted_results.append(f"Preview: {result['preview']}\n")
                
            return "Search Results:\n\n" + "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return f"Error searching documents: {e}" 